@group(0) @binding(0) var volumeOut: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn csMain(@builtin(global_invocation_id) id: vec3<u32>) {
  let dims = vec3<u32>(64u, 64u, 64u);
  if (any(id >= dims)) {
    return;
  }

  let uvw = vec3<f32>(id) / vec3<f32>(dims - 1u);
  let p = uvw * 2.0 - 1.0;
  let r = length(p);
  var d = clamp(1.0 - r / 0.5, 0.0, 1.0);
  d *= 0.5 + 0.5 * sin(p.x * 10.0);
  textureStore(volumeOut, vec3<i32>(id), vec4f(d, 0.0, 0.0, 0.0));
}
