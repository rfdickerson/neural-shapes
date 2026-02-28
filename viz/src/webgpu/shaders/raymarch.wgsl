enable f16;

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

struct CameraUniform {
  position: vec4f,
  focus: vec4f,
  up: vec4f,
  params: vec4f, // x: aspect, y: tanHalfFov, z: near, w: far
  renderParams: vec4f, // x: mode, y: sigma, z: phase g, w: density source (0 texture, 1 neural)
  sunDirectionIntensity: vec4f, // xyz: sun direction, w: intensity
}

const kCloudWorldHalfExtents = vec3f(0.8112, 0.5515, 1.0);
const kCloudWorldMin = -kCloudWorldHalfExtents;
const kCloudWorldMax = kCloudWorldHalfExtents;
const ENCODED_DIM = 51u;
const MAX_HIDDEN = 256u;

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
  _pad0: u32,
}

struct FillParams {
  baselineHalfExtents: vec4<f32>, // xyz = half extents, w = baseline scale
  params0: vec4<f32>, // x=baseline sharpness, yzw=reserved (w=representation mode)
  params1: vec4<f32>, // x=levelset decode k, y=iso logit, z=iso value, w=low residual scale
  params2: vec4<f32>, // x=high residual scale, y=baseline mode, z=shell threshold, w=support threshold
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var volumeSampler: sampler;
@group(0) @binding(2) var volumeTex: texture_3d<f32>;
@group(0) @binding(3) var sunTransmittanceTex: texture_3d<f32>;
@group(0) @binding(4) var multiScatterTex: texture_3d<f32>;
@group(0) @binding(5) var<storage, read> mlpWeights: array<f16>;
@group(0) @binding(6) var<uniform> mlpMeta: MLPMetadata;
@group(0) @binding(7) var<uniform> fillParams: FillParams;
@group(0) @binding(8) var blueNoiseTex: texture_2d<f32>;
@group(0) @binding(9) var baselinePhiTex: texture_3d<f32>;

fn worldToLocal(pWorld: vec3f) -> vec3f {
  return pWorld / kCloudWorldHalfExtents;
}

fn localToUvw(pLocal: vec3f) -> vec3f {
  return pLocal * 0.5 + vec3f(0.5);
}

fn samplePackedVolume(p: vec3f) -> vec4f {
  let uvw = localToUvw(worldToLocal(p));
  if (any(uvw <= vec3f(0.0)) || any(uvw >= vec3f(1.0))) {
    return vec4f(0.0);
  }
  return textureSampleLevel(volumeTex, volumeSampler, uvw, 0.0);
}

@vertex
fn vsMain(@builtin(vertex_index) vid: u32) -> VSOut {
  var positions = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f(1.0, -1.0),
    vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0),
    vec2f(1.0, 1.0)
  );
  let p = positions[vid];
  var out: VSOut;
  out.position = vec4f(p, 0.0, 1.0);
  out.uv = p * 0.5 + vec2f(0.5);
  return out;
}

fn sdBox(p: vec3<f32>, halfExtents: vec3<f32>) -> f32 {
  let q = abs(p) - halfExtents;
  let outside = length(max(q, vec3<f32>(0.0)));
  let inside = min(max(max(q.x, q.y), q.z), 0.0);
  return outside + inside;
}

fn smoothBoxBaseline(p: vec3<f32>) -> f32 {
  let sdf = sdBox(p, fillParams.baselineHalfExtents.xyz);
  return 1.0 / (1.0 + exp(fillParams.params0.x * sdf));
}

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

fn densityTexture(p: vec3f) -> f32 {
  return clamp(samplePackedVolume(p).a, 0.0, 1.0);
}

fn sampleSunTransmittance(p: vec3f) -> f32 {
  let uvw = localToUvw(worldToLocal(p));
  if (any(uvw <= vec3f(0.0)) || any(uvw >= vec3f(1.0))) {
    return 1.0;
  }
  return clamp(textureSampleLevel(sunTransmittanceTex, volumeSampler, uvw, 0.0).r, 0.0, 1.0);
}

fn sampleMultiScatter(p: vec3f) -> vec3f {
  let uvw = localToUvw(worldToLocal(p));
  if (any(uvw <= vec3f(0.0)) || any(uvw >= vec3f(1.0))) {
    return vec3f(0.0);
  }
  return max(textureSampleLevel(multiScatterTex, volumeSampler, uvw, 0.0).xyz, vec3f(0.0));
}

fn sampleMacroSdf(p: vec3f) -> f32 {
  let uvw = localToUvw(worldToLocal(p));
  if (any(uvw <= vec3f(0.0)) || any(uvw >= vec3f(1.0))) {
    return max(fillParams.params2.z * 4.0, 0.25);
  }

  let dims = textureDimensions(baselinePhiTex);
  let pTex = uvw * vec3f(dims - vec3<u32>(1u));
  let i0 = vec3<u32>(floor(pTex));
  let i1 = min(i0 + vec3<u32>(1u), dims - vec3<u32>(1u));
  let t = pTex - vec3f(i0);

  let c000 = textureLoad(baselinePhiTex, vec3<i32>(i0), 0).x;
  let c100 = textureLoad(baselinePhiTex, vec3<i32>(vec3<u32>(i1.x, i0.y, i0.z)), 0).x;
  let c010 = textureLoad(baselinePhiTex, vec3<i32>(vec3<u32>(i0.x, i1.y, i0.z)), 0).x;
  let c110 = textureLoad(baselinePhiTex, vec3<i32>(vec3<u32>(i1.x, i1.y, i0.z)), 0).x;
  let c001 = textureLoad(baselinePhiTex, vec3<i32>(vec3<u32>(i0.x, i0.y, i1.z)), 0).x;
  let c101 = textureLoad(baselinePhiTex, vec3<i32>(vec3<u32>(i1.x, i0.y, i1.z)), 0).x;
  let c011 = textureLoad(baselinePhiTex, vec3<i32>(vec3<u32>(i0.x, i1.y, i1.z)), 0).x;
  let c111 = textureLoad(baselinePhiTex, vec3<i32>(i1), 0).x;

  let c00 = mix(c000, c100, t.x);
  let c10 = mix(c010, c110, t.x);
  let c01 = mix(c001, c101, t.x);
  let c11 = mix(c011, c111, t.x);
  let c0 = mix(c00, c10, t.y);
  let c1 = mix(c01, c11, t.y);
  let gridValue = mix(c0, c1, t.z);
  let baselineMode = fillParams.params2.y;
  if (baselineMode > 1.5) {
    // support_mask mode stores support density [0,1], not signed distance.
    let support = clamp(gridValue, 0.0, 1.0);
    let supportThreshold = clamp(fillParams.params2.w, 1.0e-4, 1.0);
    let pseudoPhi = (supportThreshold - support) / supportThreshold;
    return fillParams.baselineHalfExtents.w * pseudoPhi;
  }
  return fillParams.baselineHalfExtents.w * gridValue;
}

fn intersectAabb(rayOrigin: vec3f, rayDir: vec3f, bmin: vec3f, bmax: vec3f) -> vec2f {
  let dirSign = select(vec3f(1.0), vec3f(-1.0), rayDir < vec3f(0.0));
  let invDir = dirSign / max(abs(rayDir), vec3f(1e-6));
  let t0 = (bmin - rayOrigin) * invDir;
  let t1 = (bmax - rayOrigin) * invDir;
  let tsmaller = min(t0, t1);
  let tbigger = max(t0, t1);
  let tEnter = max(max(tsmaller.x, tsmaller.y), tsmaller.z);
  let tExit = min(min(tbigger.x, tbigger.y), tbigger.z);
  return vec2f(tEnter, tExit);
}

fn schlickPhase(cosTheta: f32, g: f32) -> f32 {
  // Fast approximation to HG with comparable visual response for cloud scattering.
  let k = 1.55 * g - 0.55 * g * g * g;
  let k2 = k * k;
  let d = 1.0 + k * cosTheta;
  let d2 = max(d * d, 1.0e-4);
  return (1.0 - k2) / (4.0 * 3.1415926535 * d2);
}

fn skyColor(dir: vec3f, sunDir: vec3f, sunIntensity: f32) -> vec3f {
  let viewDir = normalize(dir);
  let t = clamp(viewDir.y * 0.5 + 0.5, 0.0, 1.0);
  let cosTheta = max(dot(viewDir, sunDir), 0.0);
  let sunset = 1.0 - smoothstep(0.02, 0.55, sunDir.y);
  let daylight = smoothstep(-0.14, 0.12, sunDir.y);

  let dayHorizon = vec3f(0.22, 0.37, 0.68);
  let dayZenith = vec3f(0.02, 0.08, 0.28);
  let sunsetHorizon = vec3f(1.10, 0.45, 0.18);
  let sunsetZenith = vec3f(0.46, 0.15, 0.52);
  let horizon = mix(dayHorizon, sunsetHorizon, sunset);
  let zenith = mix(dayZenith, sunsetZenith, sunset);

  var sky = mix(horizon, zenith, pow(t, 0.58));
  let hazeColorDay = vec3f(0.55, 0.67, 0.90);
  let hazeColorSunset = vec3f(1.15, 0.58, 0.30);
  let hazeColor = mix(hazeColorDay, hazeColorSunset, sunset);
  let mieG = mix(0.74, 0.90, sunset);
  let miePhase = schlickPhase(cosTheta, mieG);
  let mieStrength = mix(0.010, 0.060, sunset);
  sky += hazeColor * miePhase * sunIntensity * mieStrength;

  let purpleBand = vec3f(0.60, 0.22, 0.64) * sunset * exp(-pow(max(viewDir.y, 0.0) * 3.2, 2.0)) * 0.28;
  sky += purpleBand;

  let sunDisk = pow(cosTheta, mix(420.0, 220.0, sunset));
  let sunGlow = pow(cosTheta, mix(46.0, 10.0, sunset));
  let sunColorDay = vec3f(1.0, 0.962, 0.885);
  let sunColorSunset = vec3f(1.45, 0.58, 0.24);
  let sunColor = mix(sunColorDay, sunColorSunset, sunset);
  sky += sunColor * sunIntensity * (sunDisk + sunGlow * (0.05 + 0.20 * sunset));

  let nightSky = vec3f(0.01, 0.02, 0.05);
  return mix(nightSky, sky, daylight);
}

fn toneMapACES(x: vec3f) -> vec3f {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}

@fragment
fn fsMain(in: VSOut) -> @location(0) vec4f {
  let camPos = camera.position.xyz;
  let camTarget = camera.focus.xyz;
  let worldUp = normalize(camera.up.xyz);
  let aspect = camera.params.x;
  let tanHalfFov = camera.params.y;
  let nearT = camera.params.z;
  let farT = camera.params.w;
  let modeFlag = camera.renderParams.x;
  let sigma = max(camera.renderParams.y, 0.01);
  let basePhaseG = clamp(camera.renderParams.z, 0.0, 0.95);
  let sunDir = normalize(camera.sunDirectionIntensity.xyz);
  let sunIntensity = max(camera.sunDirectionIntensity.w, 0.01);

  let forward = normalize(camTarget - camPos);
  let right = normalize(cross(forward, worldUp));
  let up = cross(right, forward);

  // Keep image upright (sun at top): y should increase toward top in view space.
  let ndc = vec2f(in.uv.x * 2.0 - 1.0, in.uv.y * 2.0 - 1.0);
  let rayDir = normalize(forward + right * ndc.x * aspect * tanHalfFov + up * ndc.y * tanHalfFov);

  let boundsMin = kCloudWorldMin;
  let boundsMax = kCloudWorldMax;
  let hit = intersectAabb(camPos, rayDir, boundsMin, boundsMax);
  let tStart = max(nearT, hit.x);
  let tEnd = min(farT, hit.y);
  if (tEnd <= tStart) {
    if (modeFlag > 0.5) {
      return vec4f(toneMapACES(skyColor(rayDir, sunDir, sunIntensity)), 1.0);
    }
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }

  var transmittance = 1.0;
  var accum = 0.0;
  var cloudAccum = vec3f(0.0);
  let maxSteps = 192u;
  let totalDist = tEnd - tStart;
  let stepSize = totalDist / f32(maxSteps);
  let noiseDims = textureDimensions(blueNoiseTex);
  let pixelCoord = vec2u(u32(max(i32(in.position.x), 0)), u32(max(i32(in.position.y), 0)));
  let noiseCoord = vec2u(pixelCoord.x % noiseDims.x, pixelCoord.y % noiseDims.y);
  let noiseSample = textureLoad(blueNoiseTex, vec2i(i32(noiseCoord.x), i32(noiseCoord.y)), 0);
  let jitterFactor = noiseSample.r;
  let sunset = 1.0 - smoothstep(0.02, 0.55, sunDir.y);
  let autoPhaseG = clamp(mix(basePhaseG, min(basePhaseG + 0.08, 0.84), sunset), 0.0, 0.90);
  let sunViewCos = clamp(dot(rayDir, sunDir), -1.0, 1.0);
  let sunPhase = 1.5 * schlickPhase(sunViewCos, autoPhaseG);
  let forwardScatterBoost = pow(max(sunViewCos, 0.0), 4.0);
  let sunColor = mix(vec3f(1.0, 0.962, 0.885), vec3f(1.35, 0.58, 0.25), sunset);
  let sunRadiance = sunColor * sunIntensity;
  let cloudAlbedo = 1.0;
  let ambientSky = skyColor(vec3f(0.0, 1.0, 0.0), sunDir, sunIntensity);
  let ambientBase = mix(vec3f(0.08, 0.10, 0.14), vec3f(0.18, 0.10, 0.16), sunset);
  let ambientTerm = mix(ambientBase, ambientSky, 0.35) * cloudAlbedo;

  let useSdfGuidance = fillParams.params2.y > 0.5;
  let sdfShell = max(fillParams.params2.z, 0.03);
  let bigStep = stepSize * 4.0;
  let maxSdfSkip = stepSize * 24.0;
  var t = tStart + jitterFactor * stepSize;
  for (var i = 0u; i < maxSteps; i++) {
    if (transmittance < 0.01 || t > tEnd) {
      break;
    }
    let p = camPos + rayDir * t;
    var localStep = stepSize;
    if (useSdfGuidance) {
      let distToCloud = sampleMacroSdf(p);
      if (distToCloud > sdfShell) {
        // SDF-guided empty-space skipping (sphere-tracing style).
        // Move close to the shell with a conservative factor to avoid thin-feature misses.
        let distanceToShell = max(distToCloud - (sdfShell * 0.5), 0.0);
        let sdfSkip = max(stepSize, distanceToShell * 0.9);
        let skipStep = min(maxSdfSkip, max(bigStep, sdfSkip));
        t += skipStep;
        continue;
      }
      let nearBoundary = 1.0 - smoothstep(0.0, sdfShell, abs(distToCloud));
      let stepJitter = mix(0.92, 1.08, fract(noiseSample.g + f32(i) * 0.754877666));
      localStep = mix(stepSize * 1.5, stepSize * 0.45, nearBoundary) * stepJitter;
    } else {
      let d = densityTexture(p);
      let edgeBandIn = smoothstep(0.02, 0.20, d);
      let edgeBandOut = 1.0 - smoothstep(0.36, 0.75, d);
      let edgeBand = clamp(edgeBandIn * edgeBandOut, 0.0, 1.0);
      let stepJitter = mix(0.92, 1.08, fract(noiseSample.g + f32(i) * 0.754877666));
      localStep = mix(stepSize * 2.2, stepSize * 0.45, edgeBand) * stepJitter;
    }
    let pMid = camPos + rayDir * (t + localStep * 0.5);
    let packedMid = samplePackedVolume(pMid);
    let dMid = clamp(packedMid.a, 0.0, 1.0);
    if (dMid <= 1e-4) {
      t += localStep;
      continue;
    }
    let densityShaped = mix(dMid, smoothstep(0.02, 0.50, dMid), 0.35);
    let sigmaT = densityShaped * sigma;
    let segmentTransmittance = exp(-sigmaT * localStep);
    let contrib = transmittance * (1.0 - segmentTransmittance);
    if (modeFlag > 0.5) {
      // Depth-based culling for expensive volume lookups:
      // near shell (high transmittance) use full-quality lighting,
      // deep interior smoothly fades to a cheap approximation.
      let detailFade = smoothstep(0.3, 0.5, transmittance);
      let useExpensive = detailFade > 1.0e-3;
      var sunTr = 0.35;
      var indirectMs = vec3f(0.0);
      if (useExpensive) {
        let sunTrFull = sampleSunTransmittance(pMid);
        sunTr = mix(0.35, sunTrFull, detailFade);
        indirectMs = sampleMultiScatter(pMid) * detailFade;
      }
      let nLocal = packedMid.rgb;
      let nLen2 = dot(nLocal, nLocal);
      let nWorld = select(
        vec3f(0.0),
        normalize(vec3f(
          nLocal.x / kCloudWorldHalfExtents.x,
          nLocal.y / kCloudWorldHalfExtents.y,
          nLocal.z / kCloudWorldHalfExtents.z
        )),
        nLen2 > 1.0e-8
      );
      let ndotl = max(dot(nWorld, sunDir), 0.0);
      let normalBoost = mix(1.0, 0.92 + 0.08 * ndotl, select(0.0, 1.0, nLen2 > 1.0e-8));
      let direct = sunRadiance * sunPhase * sunTr * cloudAlbedo;
      let indirect = indirectMs * densityShaped * 0.8;
      let edgeAccent = 1.0 + forwardScatterBoost * 0.45;
      let scattering = (ambientTerm * 0.88 + direct + indirect) * normalBoost;
      cloudAccum += scattering * contrib * edgeAccent;
    } else {
      accum += contrib;
    }
    transmittance *= segmentTransmittance;
    t += localStep;
  }

  if (modeFlag > 0.5) {
    let backgroundSky = skyColor(rayDir, sunDir, sunIntensity);
    let finalCloud = cloudAccum + backgroundSky * transmittance;
    let mapped = toneMapACES(finalCloud);
    return vec4f(mapped, 1.0);
  }

  let gray = clamp(accum, 0.0, 1.0);
  return vec4f(vec3f(gray), 1.0);
}
