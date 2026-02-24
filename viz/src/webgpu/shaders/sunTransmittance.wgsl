const kCloudAabbMin = vec3f(-1.0, -1.0, -1.0);
const kCloudAabbMax = vec3f(1.0, 1.0, 1.0);

struct LightingParams {
  sunDirectionIntensity: vec4f,
  mediumParams: vec4f, // x=densityScale, y=g, z=albedo, w=extinctionCoeff
  solveParams: vec4f, // x=volumeDim, y=lambda, z=stepDistance, w=sunMarchSteps
}

@group(0) @binding(0) var densityVolume: texture_3d<f32>;
@group(0) @binding(2) var sunTransmittanceOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var<uniform> params: LightingParams;

fn hash3(p: vec3u) -> u32 {
  var h = p.x * 747796405u;
  h = h ^ (p.y * 2891336453u);
  h = h ^ (p.z * 1181783497u);
  h = h ^ (h >> 16u);
  h = h * 2246822519u;
  h = h ^ (h >> 13u);
  h = h * 3266489917u;
  h = h ^ (h >> 16u);
  return h;
}

fn sampleDensity(p: vec3f) -> f32 {
  let uvw = (p - kCloudAabbMin) / (kCloudAabbMax - kCloudAabbMin);
  if (any(uvw < vec3f(0.0)) || any(uvw > vec3f(1.0))) {
    return 0.0;
  }
  let dims = textureDimensions(densityVolume, 0);
  let coord = min(vec3u(uvw * vec3f(dims)), dims - vec3u(1u));
  return clamp(textureLoad(densityVolume, vec3i(coord), 0).r, 0.0, 1.0);
}

fn intersectAabb(rayOrigin: vec3f, rayDir: vec3f, bmin: vec3f, bmax: vec3f) -> vec2f {
  let invDir = 1.0 / rayDir;
  let t0 = (bmin - rayOrigin) * invDir;
  let t1 = (bmax - rayOrigin) * invDir;
  let tSmall = min(t0, t1);
  let tLarge = max(t0, t1);
  let tMin = max(max(tSmall.x, tSmall.y), max(tSmall.z, 0.0));
  let tMax = min(min(tLarge.x, tLarge.y), tLarge.z);
  return vec2f(tMin, tMax);
}

@compute @workgroup_size(4, 4, 4)
fn csMain(@builtin(global_invocation_id) id: vec3u) {
  let dim = textureDimensions(sunTransmittanceOut);
  if (any(id >= dim)) {
    return;
  }

  let uvw = (vec3f(id) + vec3f(0.5)) / vec3f(dim);
  let p = mix(kCloudAabbMin, kCloudAabbMax, uvw);
  let sunDir = normalize(params.sunDirectionIntensity.xyz);

  let hit = intersectAabb(p + sunDir * 0.01, sunDir, kCloudAabbMin, kCloudAabbMax);
  if (hit.y <= hit.x) {
    textureStore(sunTransmittanceOut, vec3i(id), vec4f(1.0, 0.0, 0.0, 1.0));
    return;
  }

  let steps = u32(max(params.solveParams.w, 1.0));
  let segmentLen = hit.y - hit.x;
  let stepSize = segmentLen / f32(max(steps, 1u));
  if (stepSize <= 0.0) {
    textureStore(sunTransmittanceOut, vec3i(id), vec4f(1.0, 0.0, 0.0, 1.0));
    return;
  }

  let densityScale = max(params.mediumParams.x, 0.01);
  let extinctionCoeff = max(params.mediumParams.w, 0.01);
  let jitter = f32(hash3(id)) * (1.0 / 4294967295.0);

  var opticalDepth = 0.0;
  var t = hit.x + fract(jitter + 0.5) * stepSize;
  for (var i = 0u; i < steps; i++) {
    let x = p + sunDir * t;
    let sigmaT = sampleDensity(x) * densityScale * extinctionCoeff;
    opticalDepth += sigmaT * stepSize;
    if (opticalDepth > 12.0) {
      break;
    }
    t += stepSize;
  }

  let transmittance = exp(-opticalDepth);
  textureStore(sunTransmittanceOut, vec3i(id), vec4f(transmittance, 0.0, 0.0, 1.0));
}
