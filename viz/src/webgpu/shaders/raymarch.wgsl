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

fn sampleDensity(p: vec3f) -> f32 {
  let uvw = p * 0.5 + vec3f(0.5);
  if (any(uvw < vec3f(0.0)) || any(uvw > vec3f(1.0))) {
    return 0.0;
  }
  return textureSampleLevel(volumeTex, volumeSampler, uvw, 0.0).r;
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

  var t = nearT;
  var transmittance = 1.0;
  var accum = 0.0;
  let stepSize = 0.02;

  loop {
    if (t > farT || transmittance < 0.01) {
      break;
    }

    let p = camPos + rayDir * t;
    let d = sampleDensity(p);
    let alpha = clamp(d * stepSize * 2.0, 0.0, 1.0);
    let contrib = transmittance * alpha;
    accum += contrib;
    transmittance *= (1.0 - alpha);
    t += stepSize;
  }

  let gray = clamp(accum, 0.0, 1.0);
  return vec4f(vec3f(gray), 1.0);
}
