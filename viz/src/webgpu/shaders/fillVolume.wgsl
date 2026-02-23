@group(0) @binding(0) var volumeOut: texture_storage_3d<rgba8unorm, write>;

fn densitySphere(p: vec3f) -> f32 {
  if (length(p) < 0.5) {
    return 1.0;
  }
  return 0.0;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(volumeOut);
  if (any(id >= dims)) {
    return;
  }

  let uvw = (vec3f(id) + vec3f(0.5)) / vec3f(dims);
  let p = uvw * 2.0 - vec3f(1.0);
  let d = densitySphere(p);
  textureStore(volumeOut, vec3i(id), vec4f(d, d, d, 1.0));
}
