enable f16;

const ENCODED_DIM = 27u;
const MAX_HIDDEN = 128u;
const BOX_HALF_EXTENTS = vec3<f32>(0.6, 0.25, 0.6);
const BOX_SHARPNESS = 14.0;
const DENSITY_WRITE_CUTOFF = 0.18;

struct MLPMetadata {
  inputDim: u32,
  hidden0: u32,
  hidden1: u32,
  w0Offset: u32,
  b0Offset: u32,
  w1Offset: u32,
  b1Offset: u32,
  w2Offset: u32,
  b2Offset: u32,
  fourierLevels: u32,
  _pad0: u32,
  _pad1: u32,
}

struct FillParams {
  detailParams: vec4f, // x=scale, y=erosionStrength, z=edgeStart, w=edgeEnd
  flags: vec4f, // x=enableDetail
}

@group(0) @binding(0) var volumeOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(1) var<storage, read> mlpWeights: array<f16>;
@group(0) @binding(2) var<uniform> mlpMeta: MLPMetadata;
@group(0) @binding(3) var detailNoiseTex: texture_3d<f32>;
@group(0) @binding(4) var detailNoiseSampler: sampler;
@group(0) @binding(5) var<uniform> fillParams: FillParams;

fn encodePosition(p: vec3<f32>) -> array<f32, ENCODED_DIM> {
  var out: array<f32, ENCODED_DIM>;
  var idx = 0u;

  out[idx] = p.x;
  idx++;
  out[idx] = p.y;
  idx++;
  out[idx] = p.z;
  idx++;

  for (var i = 0u; i < mlpMeta.fourierLevels; i++) {
    let freq = pow(2.0, f32(i));

    // Must match training order exactly:
    // [sin(freq*x), sin(freq*y), sin(freq*z), cos(freq*x), cos(freq*y), cos(freq*z)]
    out[idx] = sin(freq * p.x);
    idx++;
    out[idx] = sin(freq * p.y);
    idx++;
    out[idx] = sin(freq * p.z);
    idx++;
    out[idx] = cos(freq * p.x);
    idx++;
    out[idx] = cos(freq * p.y);
    idx++;
    out[idx] = cos(freq * p.z);
    idx++;
  }

  return out;
}

fn linearFromEncoded(
  input: array<f32, ENCODED_DIM>,
  inputDim: u32,
  outputDim: u32,
  wOffset: u32,
  bOffset: u32
) -> array<f32, MAX_HIDDEN> {
  var out: array<f32, MAX_HIDDEN>;

  for (var i = 0u; i < outputDim; i++) {
    var sum = f32(mlpWeights[bOffset + i]);
    for (var j = 0u; j < inputDim; j++) {
      let w = f32(mlpWeights[wOffset + i * inputDim + j]);
      sum += w * input[j];
    }
    out[i] = max(sum, 0.0);
  }

  return out;
}

fn linearFromHidden(
  input: array<f32, MAX_HIDDEN>,
  inputDim: u32,
  outputDim: u32,
  wOffset: u32,
  bOffset: u32
) -> array<f32, MAX_HIDDEN> {
  var out: array<f32, MAX_HIDDEN>;

  for (var i = 0u; i < outputDim; i++) {
    var sum = f32(mlpWeights[bOffset + i]);
    for (var j = 0u; j < inputDim; j++) {
      let w = f32(mlpWeights[wOffset + i * inputDim + j]);
      sum += w * input[j];
    }
    out[i] = max(sum, 0.0);
  }

  return out;
}

fn mlpResidual(p: vec3<f32>) -> f32 {
  let encoded = encodePosition(p);
  let h0 = linearFromEncoded(
    encoded,
    mlpMeta.inputDim,
    mlpMeta.hidden0,
    mlpMeta.w0Offset,
    mlpMeta.b0Offset
  );
  let h1 = linearFromHidden(
    h0,
    mlpMeta.hidden0,
    mlpMeta.hidden1,
    mlpMeta.w1Offset,
    mlpMeta.b1Offset
  );

  var sum = f32(mlpWeights[mlpMeta.b2Offset]);
  for (var i = 0u; i < mlpMeta.hidden1; i++) {
    let w = f32(mlpWeights[mlpMeta.w2Offset + i]);
    sum += w * h1[i];
  }

  return sum;
}

fn sdBox(p: vec3<f32>, halfExtents: vec3<f32>) -> f32 {
  let q = abs(p) - halfExtents;
  let outside = length(max(q, vec3<f32>(0.0)));
  let inside = min(max(max(q.x, q.y), q.z), 0.0);
  return outside + inside;
}

fn smoothBoxBaseline(p: vec3<f32>) -> f32 {
  let sdf = sdBox(p, BOX_HALF_EXTENTS);
  return 1.0 / (1.0 + exp(BOX_SHARPNESS * sdf));
}

fn erodeWithDetail(rawDensity: f32, p: vec3<f32>) -> f32 {
  if (fillParams.flags.x < 0.5) {
    return rawDensity;
  }
  let detailScale = max(fillParams.detailParams.x, 0.001);
  let erosionStrength = clamp(fillParams.detailParams.y, 0.0, 2.0);
  let edgeStart = clamp(fillParams.detailParams.z, 0.0, 1.0);
  let edgeEnd = clamp(fillParams.detailParams.w, edgeStart + 1.0e-4, 1.0);

  let uvw = fract((p * 0.5 + vec3<f32>(0.5)) * detailScale);
  let edgeMask = pow(1.0 - smoothstep(edgeStart, edgeEnd, rawDensity), 1.45);

  // Domain-warp detail coordinates near edges so breakup does not read as uniform lumps.
  let baseNoise = textureSampleLevel(detailNoiseTex, detailNoiseSampler, uvw, 0.0);
  let warp = (baseNoise.rgb * 2.0 - vec3<f32>(1.0)) * (0.11 * edgeMask);
  let warpedUvw = fract(uvw + warp);

  let noise0 = textureSampleLevel(detailNoiseTex, detailNoiseSampler, warpedUvw, 0.0);
  let noise1 = textureSampleLevel(
    detailNoiseTex,
    detailNoiseSampler,
    fract(warpedUvw * 2.07 + vec3<f32>(0.17, 0.31, 0.47)),
    0.0
  );
  let noise2 = textureSampleLevel(
    detailNoiseTex,
    detailNoiseSampler,
    fract(warpedUvw * 4.19 + vec3<f32>(0.43, 0.13, 0.73)),
    0.0
  );

  let worley = clamp(noise0.g * 0.55 + noise1.g * 0.30 + noise2.g * 0.15, 0.0, 1.0);
  let turbulence = clamp(noise1.b * 0.65 + noise2.b * 0.35, 0.0, 1.0);
  let perlin = noise0.r;

  // Modulate edge threshold with detail (signed) rather than pure subtraction.
  let cauliflowerNoise = clamp((worley * 0.72) + ((1.0 - turbulence) * 0.20) + (perlin * 0.08), 0.0, 1.0);
  let detailSigned = (cauliflowerNoise - 0.5) * 2.0;
  let thresholdShift = detailSigned * erosionStrength * edgeMask * 0.22;
  let shiftedDensity = clamp(rawDensity + thresholdShift, 0.0, 1.0);

  // High-contrast cellular carving concentrated at boundaries.
  let carveMask = smoothstep(0.34, 0.76, cauliflowerNoise);
  let carveAmount = edgeMask * erosionStrength * 0.85;
  let carvedDensity = shiftedDensity * mix(1.0, carveMask, carveAmount);
  return clamp(carvedDensity, 0.0, 1.0);
}

@compute @workgroup_size(4, 4, 4)
fn csMain(@builtin(global_invocation_id) id: vec3<u32>) {
  let dims = textureDimensions(volumeOut);
  if (any(id >= dims)) {
    return;
  }

  let uvw = vec3<f32>(id) / vec3<f32>(dims - 1u);
  let p = uvw * 2.0 - 1.0;

  let macroDensity = smoothBoxBaseline(p);
  let residual = mlpResidual(p); // keep signed so negative values can carve holes
  let baseDensity = clamp(macroDensity + residual, 0.0, 1.0);
  let detailedDensity = erodeWithDetail(baseDensity, p);
  let shapedDensity = pow(detailedDensity, 0.82);
  let density = select(0.0, shapedDensity, shapedDensity >= DENSITY_WRITE_CUTOFF);

  textureStore(volumeOut, vec3<i32>(id), vec4f(density, 0.0, 0.0, 0.0));
}
