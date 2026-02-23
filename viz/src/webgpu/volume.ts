export interface VolumeTextureData {
  readonly size: number;
  readonly bytesPerRow: number;
  readonly rowsPerImage: number;
  readonly data: Uint8Array<ArrayBuffer>;
}

export function createSphereVolumeData(size = 32): VolumeTextureData {
  const voxelCount = size * size * size;
  const raw = new Uint8Array(new ArrayBuffer(voxelCount));
  let idx = 0;

  for (let z = 0; z < size; z++) {
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const px = ((x + 0.5) / size) * 2.0 - 1.0;
        const py = ((y + 0.5) / size) * 2.0 - 1.0;
        const pz = ((z + 0.5) / size) * 2.0 - 1.0;
        const d = Math.hypot(px, py, pz) < 0.5 ? 1 : 0;
        raw[idx++] = d * 255;
      }
    }
  }

  const bytesPerVoxel = 1;
  const rowStride = size * bytesPerVoxel;
  const bytesPerRow = Math.ceil(rowStride / 256) * 256;
  const rowsPerImage = size;
  const padded = new Uint8Array(new ArrayBuffer(bytesPerRow * rowsPerImage * size));

  for (let z = 0; z < size; z++) {
    for (let y = 0; y < size; y++) {
      const srcOffset = z * size * size + y * size;
      const dstOffset = z * bytesPerRow * rowsPerImage + y * bytesPerRow;
      padded.set(raw.subarray(srcOffset, srcOffset + size), dstOffset);
    }
  }

  return {
    size,
    bytesPerRow,
    rowsPerImage,
    data: padded
  };
}
