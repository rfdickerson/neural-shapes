enable f16;

const ENCODED_DIM = 51u;
const MAX_HIDDEN = 256u;
const EDGE_RECON_MIN = 0.035;
const EDGE_RECON_MAX = 0.42;
const EDGE_SUBVOXEL_SCALE = 0.35;

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
  outputDim: u32,
  _pad0: u32, // inference mode: 2=detail_only_density
}

struct FillParams {
  baselineHalfExtents: vec4<f32>,
  params0: vec4<f32>,
  params1: vec4<f32>, // z=detail fade start, w=detail fade end
  params2: vec4<f32>, // w=detail amplitude
}

@group(0) @binding(0) var volumeOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(1) var featureFieldTex: texture_3d<f32>;
@group(0) @binding(2) var<storage, read> mlpWeights: array<f16>;
@group(0) @binding(3) var<uniform> mlpMeta: MLPMetadata;
@group(0) @binding(4) var<uniform> fillParams: FillParams;

fn sampleFeatureField(p: vec3<f32>) -> f32 {
  let dims = textureDimensions(featureFieldTex);
  let uvw = clamp(p * 0.5 + 0.5, vec3<f32>(0.0), vec3<f32>(1.0));
  let pTex = uvw * vec3<f32>(dims - vec3<u32>(1u));
  let i0 = vec3<u32>(floor(pTex));
  let i1 = min(i0 + vec3<u32>(1u), dims - vec3<u32>(1u));
  let t = pTex - vec3<f32>(i0);

  let c000 = textureLoad(featureFieldTex, vec3<i32>(i0), 0).x;
  let c100 = textureLoad(featureFieldTex, vec3<i32>(vec3<u32>(i1.x, i0.y, i0.z)), 0).x;
  let c010 = textureLoad(featureFieldTex, vec3<i32>(vec3<u32>(i0.x, i1.y, i0.z)), 0).x;
  let c110 = textureLoad(featureFieldTex, vec3<i32>(vec3<u32>(i1.x, i1.y, i0.z)), 0).x;
  let c001 = textureLoad(featureFieldTex, vec3<i32>(vec3<u32>(i0.x, i0.y, i1.z)), 0).x;
  let c101 = textureLoad(featureFieldTex, vec3<i32>(vec3<u32>(i1.x, i0.y, i1.z)), 0).x;
  let c011 = textureLoad(featureFieldTex, vec3<i32>(vec3<u32>(i0.x, i1.y, i1.z)), 0).x;
  let c111 = textureLoad(featureFieldTex, vec3<i32>(i1), 0).x;

  let c00 = mix(c000, c100, t.x);
  let c10 = mix(c010, c110, t.x);
  let c01 = mix(c001, c101, t.x);
  let c11 = mix(c011, c111, t.x);
  let c0 = mix(c00, c10, t.y);
  let c1 = mix(c01, c11, t.y);
  return mix(c0, c1, t.z);
}

fn encodeFourierInput(p: vec3<f32>) -> array<f32, ENCODED_DIM> {
  var out = array<f32, ENCODED_DIM>();
  var idx = 0u;

  out[idx] = p.x;
  idx++;
  out[idx] = p.y;
  idx++;
  out[idx] = p.z;
  idx++;

  for (var i = 0u; i < mlpMeta.fourierLevels; i++) {
    let freq = pow(2.0, f32(i));
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

fn mlpDetail(p: vec3<f32>) -> f32 {
  let encoded = encodeFourierInput(p);
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
  return tanh(sum);
}

fn evalRawDensity(p: vec3<f32>) -> f32 {
  let coarse = clamp(sampleFeatureField(p), 0.0, 1.0);
  let detail = mlpDetail(p);
  let fadeStart = clamp(fillParams.params1.z, 0.0, 1.0);
  let fadeEnd = clamp(fillParams.params1.w, fadeStart + 1.0e-4, 1.0);
  let detailFade = smoothstep(fadeStart, fadeEnd, coarse);
  let detailAmp = max(fillParams.params2.w, 0.0);
  return clamp(coarse + detailAmp * detail * detailFade, 0.0, 1.0);
}

@compute @workgroup_size(4, 4, 4)
fn csMain(@builtin(global_invocation_id) id: vec3<u32>) {
  let dims = textureDimensions(volumeOut);
  if (any(id >= dims)) {
    return;
  }

  let uvw = vec3<f32>(id) / vec3<f32>(dims - 1u);
  let p = uvw * 2.0 - 1.0;
  var rawDensity = evalRawDensity(p);

  // Spend extra MLP work only around the boundary shell where voxel quantization is most visible.
  if (rawDensity > EDGE_RECON_MIN && rawDensity < EDGE_RECON_MAX) {
    let voxelSpan = (vec3<f32>(2.0) / vec3<f32>(dims - 1u)) * EDGE_SUBVOXEL_SCALE;
    let o0 = vec3<f32>( voxelSpan.x,  voxelSpan.y,  voxelSpan.z);
    let o1 = vec3<f32>( voxelSpan.x, -voxelSpan.y, -voxelSpan.z);
    let o2 = vec3<f32>(-voxelSpan.x,  voxelSpan.y, -voxelSpan.z);
    let o3 = vec3<f32>(-voxelSpan.x, -voxelSpan.y,  voxelSpan.z);
    let p0 = clamp(p + o0, vec3<f32>(-1.0), vec3<f32>(1.0));
    let p1 = clamp(p + o1, vec3<f32>(-1.0), vec3<f32>(1.0));
    let p2 = clamp(p + o2, vec3<f32>(-1.0), vec3<f32>(1.0));
    let p3 = clamp(p + o3, vec3<f32>(-1.0), vec3<f32>(1.0));
    rawDensity = (rawDensity + evalRawDensity(p0) + evalRawDensity(p1) + evalRawDensity(p2) + evalRawDensity(p3)) * 0.2;
  }

  let density = clamp(rawDensity, 0.0, 1.0);
  textureStore(volumeOut, vec3<i32>(id), vec4f(density, 0.0, 0.0, 0.0));
}
