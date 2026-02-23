export type Vec3 = [number, number, number];

export function vec3(x: number, y: number, z: number): Vec3 {
  return [x, y, z];
}

export function normalize(v: Vec3): Vec3 {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}
