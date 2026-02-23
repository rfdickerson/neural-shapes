struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

struct CameraUniform {
  position: vec4f,
  focus: vec4f,
  up: vec4f,
  params: vec4f, // x: aspect, y: tanHalfFov, z: near, w: far
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var volumeSampler: sampler;
@group(0) @binding(2) var volumeTex: texture_3d<f32>;

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
  return textureSampleLevel(volumeTex, volumeSampler, uvw, 0.0).r;
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

@fragment
fn fsMain(in: VSOut) -> @location(0) vec4f {
  let camPos = camera.position.xyz;
  let camTarget = camera.focus.xyz;
  let worldUp = normalize(camera.up.xyz);
  let aspect = camera.params.x;
  let tanHalfFov = camera.params.y;
  let nearT = camera.params.z;
  let farT = camera.params.w;

  let forward = normalize(camTarget - camPos);
  let right = normalize(cross(forward, worldUp));
  let up = cross(right, forward);

  let ndc = vec2f(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
  let rayDir = normalize(forward + right * ndc.x * aspect * tanHalfFov + up * ndc.y * tanHalfFov);

  let boundsMin = vec3f(-1.0);
  let boundsMax = vec3f(1.0);
  let hit = intersectAabb(camPos, rayDir, boundsMin, boundsMax);
  let tStart = max(nearT, hit.x);
  let tEnd = min(farT, hit.y);
  if (tEnd <= tStart) {
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }

  var transmittance = 1.0;
  var accum = 0.0;
  let voxelSize = 2.0 / 32.0;
  let stepSize = voxelSize * 0.75;
  let sigma = 2.5;
  let lightDir = normalize(vec3f(0.6, 0.7, 0.35));
  let seed = fract(sin(dot(in.uv, vec2f(12.9898, 78.233))) * 43758.5453);
  var t = tStart + seed * stepSize;

  loop {
    if (t > tEnd || transmittance < 0.01) {
      break;
    }

    let p = camPos + rayDir * t;
    let d = density(p);
    let n = estimateNormal(p, voxelSize * 0.5);
    let light = max(dot(n, lightDir), 0.0);
    let shaded = d * (0.15 + 0.85 * light);
    let attenuation = exp(-d * sigma * stepSize);
    accum += transmittance * shaded * stepSize;
    transmittance *= attenuation;
    t += stepSize;
  }

  let gray = clamp(accum, 0.0, 1.0);
  return vec4f(vec3f(gray), 1.0);
}
