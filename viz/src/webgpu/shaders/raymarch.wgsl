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
  let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
  let horizon = vec3f(0.72, 0.80, 0.95);
  let zenith = vec3f(0.12, 0.33, 0.72);
  let sky = mix(horizon, zenith, pow(t, 0.6));
  let sunDisk = pow(max(dot(dir, sunDir), 0.0), 256.0);
  let sunGlow = pow(max(dot(dir, sunDir), 0.0), 32.0) * 0.08;
  let sunColor = vec3f(1.0, 0.95, 0.85) * sunIntensity * (sunDisk + sunGlow);
  return sky + sunColor;
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
  let phaseG = clamp(camera.renderParams.z, 0.0, 0.95);
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
      return vec4f(skyColor(rayDir, sunDir, sunIntensity), 1.0);
    }
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }

  var transmittance = 1.0;
  var accum = 0.0;
  var cloudAccum = vec3f(0.0);
  let maxSteps = 128u;
  let totalDist = tEnd - tStart;
  let stepSize = totalDist / f32(maxSteps);
  let jitterFactor = hash(in.uv);
  let sunPhase = henyeyGreenstein(clamp(dot(rayDir, sunDir), -1.0, 1.0), phaseG);
  let sunRadiance = vec3f(1.0, 0.97, 0.92) * sunIntensity;
  let ambientSky = skyColor(vec3f(0.0, 1.0, 0.0), sunDir, sunIntensity);
  let ambientTerm = mix(vec3f(0.08, 0.10, 0.13), ambientSky, 0.35);

  for (var i = 0u; i < maxSteps; i++) {
    if (transmittance < 0.01) {
      break;
    }
    let t = tStart + (f32(i) + jitterFactor) * stepSize;
    let p = camPos + rayDir * t;
    let d = clamp(density(p), 0.0, 1.0);
    let alpha = 1.0 - exp(-d * sigma * stepSize);
    let contrib = alpha * transmittance;
    if (modeFlag > 0.5) {
      let sunTr = sampleSunTransmittance(p);
      let direct = sunRadiance * sunPhase * sunTr * 2.2;
      let indirect = sampleMultiScatter(p) * 0.9;
      let scattering = ambientTerm + direct + indirect;
      cloudAccum += scattering * contrib;
    } else {
      accum += contrib;
    }
    transmittance *= (1.0 - alpha);
  }

  if (modeFlag > 0.5) {
    let backgroundSky = skyColor(rayDir, sunDir, sunIntensity);
    let finalCloud = cloudAccum + backgroundSky * transmittance;
    return vec4f(clamp(finalCloud, vec3f(0.0), vec3f(1.0)), 1.0);
  }

  let gray = clamp(accum, 0.0, 1.0);
  return vec4f(vec3f(gray), 1.0);
}
