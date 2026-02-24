enable f16;

const ENCODED_DIM = 27u;
const MAX_HIDDEN = 64u;
const BOX_HALF_EXTENTS = vec3<f32>(0.6, 0.25, 0.6);
const BOX_SHARPNESS = 14.0;
const DENSITY_WRITE_CUTOFF = 0.14;

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

@group(0) @binding(0) var volumeOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(1) var<storage, read> mlpWeights: array<f16>;
@group(0) @binding(2) var<uniform> mlpMeta: MLPMetadata;

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
  let rawDensity = clamp(macroDensity + residual, 0.0, 1.0);
  let density = select(0.0, rawDensity, rawDensity >= DENSITY_WRITE_CUTOFF);

  textureStore(volumeOut, vec3<i32>(id), vec4f(density, 0.0, 0.0, 0.0));
}
