@group(0) @binding(0) var densityVolume: texture_3d<f32>;
@group(0) @binding(1) var densitySampler: sampler;
@group(0) @binding(2) var packedOut: texture_storage_3d<rgba16float, write>;

fn sampleDensity(uvw: vec3<f32>) -> f32 {
  if (any(uvw <= vec3<f32>(0.0)) || any(uvw >= vec3<f32>(1.0))) {
    return 0.0;
  }
  return clamp(textureSampleLevel(densityVolume, densitySampler, uvw, 0.0).r, 0.0, 1.0);
}

@compute @workgroup_size(4, 4, 4)
fn csMain(@builtin(global_invocation_id) id: vec3<u32>) {
  let dims = textureDimensions(packedOut);
  if (any(id >= dims)) {
    return;
  }

  let invDims = 1.0 / vec3<f32>(dims);
  let uvw = (vec3<f32>(id) + vec3<f32>(0.5)) * invDims;
  let eps = invDims;

  let d = sampleDensity(uvw);
  let gx = sampleDensity(uvw + vec3<f32>(eps.x, 0.0, 0.0)) - sampleDensity(uvw - vec3<f32>(eps.x, 0.0, 0.0));
  let gy = sampleDensity(uvw + vec3<f32>(0.0, eps.y, 0.0)) - sampleDensity(uvw - vec3<f32>(0.0, eps.y, 0.0));
  let gz = sampleDensity(uvw + vec3<f32>(0.0, 0.0, eps.z)) - sampleDensity(uvw - vec3<f32>(0.0, 0.0, eps.z));
  let g = vec3<f32>(gx, gy, gz);
  let n = select(vec3<f32>(0.0), normalize(-g), dot(g, g) > 1.0e-8);

  textureStore(packedOut, vec3<i32>(id), vec4<f32>(n, d));
}
