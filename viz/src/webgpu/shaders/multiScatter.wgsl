const kPi = 3.1415926535;

struct LightingParams {
  sunDirectionIntensity: vec4f,
  mediumParams: vec4f, // x=densityScale, y=g, z=albedo, w=extinctionCoeff
  solveParams: vec4f, // x=volumeDim, y=lambda, z=stepDistance, w=iteration
}

@group(0) @binding(0) var multiScatterOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(1) var multiScatterIn: texture_3d<f32>;
@group(0) @binding(2) var sunTransmittanceVolume: texture_3d<f32>;
@group(0) @binding(3) var densityVolume: texture_3d<f32>;
@group(0) @binding(4) var volumeSampler: sampler;
@group(0) @binding(5) var<uniform> params: LightingParams;

fn sampleMs(texel: vec3i, dim: vec3i) -> vec3f {
  let p = clamp(texel, vec3i(0), dim - vec3i(1));
  return textureLoad(multiScatterIn, p, 0).xyz;
}

@compute @workgroup_size(4, 4, 4)
fn csMain(@builtin(global_invocation_id) id: vec3u) {
  let dim = textureDimensions(multiScatterOut);
  if (any(id >= dim)) {
    return;
  }

  let uvw = (vec3f(id) + vec3f(0.5)) / vec3f(dim);
  let densityDims = textureDimensions(densityVolume, 0);
  let densityCoord = min(vec3u(uvw * vec3f(densityDims)), densityDims - vec3u(1u));
  let density = clamp(textureLoad(densityVolume, vec3i(densityCoord), 0).r, 0.0, 1.0);
  let sunTr = clamp(textureSampleLevel(sunTransmittanceVolume, volumeSampler, uvw, 0.0).r, 0.0, 1.0);

  let densityScale = max(params.mediumParams.x, 0.01);
  let albedo = clamp(params.mediumParams.z, 0.0, 1.0);
  let extinctionCoeff = max(params.mediumParams.w, 0.01);
  let lambda = clamp(params.solveParams.y, 0.0, 1.0);
  let stepDistance = max(params.solveParams.z, 1e-3);
  let iteration = params.solveParams.w;

  let sunDir = normalize(params.sunDirectionIntensity.xyz);
  let sunIntensity = max(params.sunDirectionIntensity.w, 0.01);
  let sunNoon = vec3f(1.0, 0.962, 0.885);
  let sunSet = vec3f(1.0, 0.35, 0.05);
  let dynamicSun = mix(sunSet, sunNoon, smoothstep(-0.05, 0.2, sunDir.y));
  let extinction = smoothstep(-0.15, 0.15, sunDir.y);
  let sunRadiance = dynamicSun * sunIntensity * extinction;

  let sigmaT = density * densityScale * extinctionCoeff;
  let sigmaS = sigmaT * albedo;
  let source = sigmaS * sunTr * sunRadiance * (1.0 / (4.0 * kPi));

  let d = vec3i(dim);
  let p = vec3i(id);

  var propagated = vec3f(0.0);
  if (iteration > 0.5) {
    let prev = sampleMs(p, d);
    let msXp = sampleMs(p + vec3i(1, 0, 0), d);
    let msXn = sampleMs(p + vec3i(-1, 0, 0), d);
    let msYp = sampleMs(p + vec3i(0, 1, 0), d);
    let msYn = sampleMs(p + vec3i(0, -1, 0), d);
    let msZp = sampleMs(p + vec3i(0, 0, 1), d);
    let msZn = sampleMs(p + vec3i(0, 0, -1), d);

    let bias = 0.25;
    let wXp = 0.5 - bias * sunDir.x;
    let wXn = 0.5 + bias * sunDir.x;
    let wYp = 0.5 - bias * sunDir.y;
    let wYn = 0.5 + bias * sunDir.y;
    let wZp = 0.5 - bias * sunDir.z;
    let wZn = 0.5 + bias * sunDir.z;

    var blur = (msXp * wXp + msXn * wXn) + (msYp * wYp + msYn * wYn) + (msZp * wZp + msZn * wZn);
    blur *= (1.0 / 3.0);
    propagated = mix(prev, blur, lambda);
  }

  let transmission = exp(-sigmaT * stepDistance);
  let outE = max(source + (propagated * transmission), vec3f(0.0));

  textureStore(multiScatterOut, p, vec4f(outE, 1.0));
}
