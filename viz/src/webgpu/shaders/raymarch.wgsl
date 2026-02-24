struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

struct CameraUniform {
  position: vec4f,
  focus: vec4f,
  up: vec4f,
  params: vec4f, // x: aspect, y: tanHalfFov, z: near, w: far
  renderParams: vec4f, // x: mode (0 fogOnly, 1 cloudSky), y: sigma, z: phase g
  sunDirectionIntensity: vec4f, // xyz: sun direction, w: intensity
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var volumeSampler: sampler;
@group(0) @binding(2) var volumeTex: texture_3d<f32>;
@group(0) @binding(3) var sunTransmittanceTex: texture_3d<f32>;
@group(0) @binding(4) var multiScatterTex: texture_3d<f32>;

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

fn density(p: vec3f) -> f32 {
  let uvw = p * 0.5 + vec3f(0.5);
  if (any(uvw < vec3f(0.0)) || any(uvw > vec3f(1.0))) {
    return 0.0;
  }
  return clamp(textureSampleLevel(volumeTex, volumeSampler, uvw, 0.0).r, 0.0, 1.0);
}

fn sampleSunTransmittance(p: vec3f) -> f32 {
  let uvw = p * 0.5 + vec3f(0.5);
  if (any(uvw < vec3f(0.0)) || any(uvw > vec3f(1.0))) {
    return 1.0;
  }
  return clamp(textureSampleLevel(sunTransmittanceTex, volumeSampler, uvw, 0.0).r, 0.0, 1.0);
}

fn sampleMultiScatter(p: vec3f) -> vec3f {
  let uvw = p * 0.5 + vec3f(0.5);
  if (any(uvw < vec3f(0.0)) || any(uvw > vec3f(1.0))) {
    return vec3f(0.0);
  }
  return max(textureSampleLevel(multiScatterTex, volumeSampler, uvw, 0.0).xyz, vec3f(0.0));
}

fn estimateNormal(p: vec3f, epsScalar: f32) -> vec3f {
  let eps = vec3f(epsScalar, 0.0, 0.0);
  let gx = density(p + eps.xyy) - density(p - eps.xyy);
  let gy = density(p + eps.yxy) - density(p - eps.yxy);
  let gz = density(p + eps.yyx) - density(p - eps.yyx);
  let g = vec3f(gx, gy, gz);
  let len2 = dot(g, g);
  if (len2 < 1e-8) {
    return vec3f(0.0, 0.0, 1.0);
  }
  return normalize(g);
}

fn intersectAabb(rayOrigin: vec3f, rayDir: vec3f, bmin: vec3f, bmax: vec3f) -> vec2f {
  let invDir = 1.0 / rayDir;
  let t0 = (bmin - rayOrigin) * invDir;
  let t1 = (bmax - rayOrigin) * invDir;
  let tsmaller = min(t0, t1);
  let tbigger = max(t0, t1);
  let tEnter = max(max(tsmaller.x, tsmaller.y), tsmaller.z);
  let tExit = min(min(tbigger.x, tbigger.y), tbigger.z);
  return vec2f(tEnter, tExit);
}

fn hash(p: vec2f) -> f32 {
  let h = dot(p, vec2f(127.1, 311.7));
  return fract(sin(h) * 43758.5453123);
}

fn henyeyGreenstein(cosTheta: f32, g: f32) -> f32 {
  let g2 = g * g;
  let d = max(1.0 + g2 - 2.0 * g * cosTheta, 1e-4);
  return (1.0 - g2) / (4.0 * 3.1415926535 * pow(d, 1.5));
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
  let miePhase = henyeyGreenstein(cosTheta, mieG);
  let mieStrength = mix(0.010, 0.060, sunset);
  sky += hazeColor * miePhase * sunIntensity * mieStrength;

  let purpleBand = vec3f(0.60, 0.22, 0.64) * sunset * exp(-pow(max(viewDir.y, 0.0) * 3.2, 2.0)) * 0.28;
  sky += purpleBand;

  let sunDisk = pow(cosTheta, mix(420.0, 220.0, sunset));
  let sunGlow = pow(cosTheta, mix(46.0, 10.0, sunset));
  let sunColorDay = vec3f(1.0, 0.97, 0.92);
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

  let boundsMin = vec3f(-1.0);
  let boundsMax = vec3f(1.0);
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
  let jitterFactor = hash(in.uv);
  let sunset = 1.0 - smoothstep(0.02, 0.55, sunDir.y);
  let autoPhaseG = clamp(mix(basePhaseG, min(basePhaseG + 0.08, 0.84), sunset), 0.0, 0.90);
  let sunViewCos = clamp(dot(rayDir, sunDir), -1.0, 1.0);
  let sunPhase = 1.5 * henyeyGreenstein(sunViewCos, autoPhaseG);
  let forwardScatterBoost = pow(max(sunViewCos, 0.0), 4.0);
  let sunColor = mix(vec3f(1.0, 0.97, 0.92), vec3f(1.35, 0.58, 0.25), sunset);
  let sunRadiance = sunColor * sunIntensity;
  let cloudAlbedo = 0.92;
  let ambientSky = skyColor(vec3f(0.0, 1.0, 0.0), sunDir, sunIntensity);
  let ambientBase = mix(vec3f(0.08, 0.10, 0.14), vec3f(0.18, 0.10, 0.16), sunset);
  let ambientTerm = mix(ambientBase, ambientSky, 0.35) * cloudAlbedo;

  var t = tStart + jitterFactor * stepSize;
  for (var i = 0u; i < maxSteps; i++) {
    if (transmittance < 0.005 || t > tEnd) {
      break;
    }
    let p = camPos + rayDir * t;
    let d = density(p);
    let densityMask = smoothstep(0.02, 0.24, d);
    let localStep = mix(stepSize * 2.0, stepSize * 0.4, densityMask);
    if (d <= 1e-4) {
      t += localStep;
      continue;
    }
    let sigmaT = d * sigma;
    let segmentTransmittance = exp(-sigmaT * localStep);
    let contrib = transmittance * (1.0 - segmentTransmittance);
    if (modeFlag > 0.5) {
      let sunTr = sampleSunTransmittance(p);
      let n = estimateNormal(p, localStep * 0.75);
      let ndotl = max(dot(n, sunDir), 0.0);
      let viewFacing = max(dot(n, -rayDir), 0.0);
      let rim = pow(1.0 - viewFacing, 2.0);

      let direct = sunRadiance * sunPhase * sunTr * cloudAlbedo * (0.90 + 0.10 * ndotl);
      let indirect = sampleMultiScatter(p) * d * 0.8;
      let edgeAccent = 1.0 + rim * 0.4 + forwardScatterBoost * 0.6;
      let scattering = ambientTerm * 0.88 + direct + indirect;
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
