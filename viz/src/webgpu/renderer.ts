import shaderSource from "./shaders/raymarch.wgsl?raw";
import fillVolumeSource from "./shaders/fillVolume.wgsl?raw";
import sunTransmittanceSource from "./shaders/sunTransmittance.wgsl?raw";
import multiScatterSource from "./shaders/multiScatter.wgsl?raw";
import type { OrbitCamera } from "./orbitCamera";

const CAMERA_UNIFORM_BYTES = 96;
const VOLUME_SIZE = 128;
const SHADER_MAX_INPUT_DIM = 27;
const SHADER_MAX_HIDDEN = 128;
const MLP_META_URL = "/mlp/residual_mlp_metadata.json";
const MLP_WEIGHTS_URL = "/mlp/residual_mlp_weights.bin";
const EXPECTED_ENCODING_ORDER = "input_xyz_then_per_level_sin_xyz_cos_xyz";
const DEFAULT_SIGMA = 4.0;
const DEFAULT_PHASE_G = 0.72;
const DEFAULT_SUN_INTENSITY = 2.4;
const DEFAULT_ALBEDO = 0.92;
const LIGHT_MARCH_STEPS = 96;
const MULTISCATTER_ITERS = 8;
const MULTISCATTER_LAMBDA = 0.58;
const DETAIL_TEXTURE_SIZE = 64;
const DETAIL_SCALE = 3.25;
const DETAIL_EROSION_STRENGTH = 0.32;
const DETAIL_EDGE_START = 0.08;
const DETAIL_EDGE_END = 0.58;

export type RenderMode = "cloudSky" | "fogOnly";

interface ExportMetadata {
  layout: string;
  dtype: string;
  encoding: {
    levels: number;
    include_input?: boolean;
    frequency_base?: number;
    uses_two_pi?: boolean;
    ordering?: string;
    axis_order?: string;
    input_dims?: number;
    encoded_dims: number;
  };
  network: {
    layers: number[];
  };
  offsets: Record<string, number>;
  total_floats: number;
}

interface LoadedMlpData {
  weightsFp16: Uint16Array<ArrayBuffer>;
  metaUniform: Uint32Array<ArrayBuffer>;
}

const f32Scratch = new Float32Array(1);
const i32Scratch = new Int32Array(f32Scratch.buffer);

function float32ToFloat16Bits(value: number): number {
  f32Scratch[0] = value;
  const x = i32Scratch[0];

  let bits = (x >> 16) & 0x8000;
  let mantissa = (x >> 12) & 0x07ff;
  const exponent = (x >> 23) & 0xff;

  if (exponent < 103) {
    return bits;
  }

  if (exponent > 142) {
    bits |= 0x7c00;
    bits |= ((exponent === 255 ? 0 : 1) && (x & 0x007fffff)) ? 1 : 0;
    return bits;
  }

  if (exponent < 113) {
    mantissa |= 0x0800;
    bits |= (mantissa >> (114 - exponent)) + ((mantissa >> (113 - exponent)) & 1);
    return bits;
  }

  bits |= ((exponent - 112) << 10) | (mantissa >> 1);
  bits += mantissa & 1;
  return bits;
}

function packFloat32ToFloat16(data: Float32Array<ArrayBufferLike>): Uint16Array<ArrayBuffer> {
  const out = new Uint16Array<ArrayBuffer>(new ArrayBuffer(data.length * 2));
  for (let i = 0; i < data.length; i++) {
    out[i] = float32ToFloat16Bits(data[i]);
  }
  return out;
}

function readOffset(record: Record<string, number>, key: string): number {
  const value = record[key];
  if (!Number.isFinite(value)) {
    throw new Error(`Missing or invalid offset '${key}' in MLP metadata.`);
  }
  const offset = Math.trunc(value);
  if (offset < 0) {
    throw new Error(`Offset '${key}' must be non-negative.`);
  }
  return offset;
}

async function loadExportedMlpData(): Promise<LoadedMlpData> {
  const metaResponse = await fetch(MLP_META_URL);
  if (!metaResponse.ok) {
    throw new Error(
      `Failed to load '${MLP_META_URL}'. Copy exported files into 'viz/public/mlp/' and retry.`
    );
  }
  const meta = (await metaResponse.json()) as Partial<ExportMetadata>;
  if (!meta.layout || !meta.dtype || !meta.encoding || !meta.network || !meta.offsets) {
    throw new Error("MLP metadata is missing required fields.");
  }
  if (meta.layout !== "[W0][b0][W1][b1][W2][b2]") {
    throw new Error(`Unexpected weight layout '${meta.layout}'.`);
  }
  if (meta.dtype !== "float32") {
    throw new Error(`Expected float32 exported weights, got '${meta.dtype}'.`);
  }

  const layers = meta.network.layers ?? [];
  if (!Array.isArray(layers) || layers.length !== 4) {
    throw new Error("MLP metadata network.layers must be [inputDim, hidden0, hidden1, outputDim].");
  }
  if (!layers.every((x) => Number.isFinite(x))) {
    throw new Error("MLP metadata network.layers contains non-numeric values.");
  }
  const inputDim = Math.trunc(layers[0]);
  const hidden0 = Math.trunc(layers[1]);
  const hidden1 = Math.trunc(layers[2]);
  const outputDim = Math.trunc(layers[3]);
  const fourierLevels = Math.trunc(meta.encoding.levels);
  const encodedDims = Math.trunc(meta.encoding.encoded_dims);
  const includeInput = meta.encoding.include_input ?? true;
  const frequencyBase = meta.encoding.frequency_base ?? 2.0;
  const usesTwoPi = meta.encoding.uses_two_pi ?? false;
  const axisOrder = meta.encoding.axis_order ?? "xyz";
  const ordering = meta.encoding.ordering ?? EXPECTED_ENCODING_ORDER;
  const inputDims = Math.trunc(meta.encoding.input_dims ?? 3);
  if (!Number.isFinite(fourierLevels) || !Number.isFinite(encodedDims)) {
    throw new Error("MLP metadata encoding.levels / encoding.encoded_dims must be numeric.");
  }
  if (!Number.isFinite(inputDims)) {
    throw new Error("MLP metadata encoding.input_dims must be numeric.");
  }
  if (!includeInput) {
    throw new Error("Renderer expects Fourier encoding metadata include_input=true.");
  }
  if (Math.abs(frequencyBase - 2.0) > 1e-6) {
    throw new Error(`Renderer expects frequency_base=2.0, got ${frequencyBase}.`);
  }
  if (usesTwoPi) {
    throw new Error("Renderer expects uses_two_pi=false (training uses sin(freq*x), not sin(2*pi*freq*x)).");
  }
  if (axisOrder !== "xyz") {
    throw new Error(`Renderer expects axis_order='xyz', got '${axisOrder}'.`);
  }
  if (ordering !== EXPECTED_ENCODING_ORDER) {
    throw new Error(`Renderer expects encoding ordering '${EXPECTED_ENCODING_ORDER}', got '${ordering}'.`);
  }
  if (inputDims !== 3) {
    throw new Error(`Renderer expects encoding.input_dims=3, got ${inputDims}.`);
  }
  if (fourierLevels < 1) {
    throw new Error(`Renderer requires fourierLevels >= 1, got ${fourierLevels}.`);
  }
  if (fourierLevels > 4) {
    throw new Error(`fourierLevels=${fourierLevels} exceeds shader max 4.`);
  }
  const expectedEncodedDims = 3 + 6 * fourierLevels;
  if (encodedDims !== expectedEncodedDims) {
    throw new Error(
      `encoded_dims=${encodedDims} does not match expected ${expectedEncodedDims} for fourierLevels=${fourierLevels}.`
    );
  }

  if (outputDim !== 1) {
    throw new Error(`Expected output_dim=1, got ${outputDim}.`);
  }
  if (inputDim !== encodedDims) {
    throw new Error(`Input dim mismatch: network input ${inputDim}, encoded dims ${encodedDims}.`);
  }
  if (inputDim > SHADER_MAX_INPUT_DIM) {
    throw new Error(`inputDim=${inputDim} exceeds shader max ${SHADER_MAX_INPUT_DIM}.`);
  }
  if (hidden0 > SHADER_MAX_HIDDEN || hidden1 > SHADER_MAX_HIDDEN) {
    throw new Error(`Hidden size exceeds shader max ${SHADER_MAX_HIDDEN}.`);
  }

  const w0Offset = readOffset(meta.offsets, "W0");
  const b0Offset = readOffset(meta.offsets, "b0");
  const w1Offset = readOffset(meta.offsets, "W1");
  const b1Offset = readOffset(meta.offsets, "b1");
  const w2Offset = readOffset(meta.offsets, "W2");
  const b2Offset = readOffset(meta.offsets, "b2");

  const weightsResponse = await fetch(MLP_WEIGHTS_URL);
  if (!weightsResponse.ok) {
    throw new Error(
      `Failed to load '${MLP_WEIGHTS_URL}'. Copy exported files into 'viz/public/mlp/' and retry.`
    );
  }
  const weightsBuffer = await weightsResponse.arrayBuffer();
  if (weightsBuffer.byteLength % 4 !== 0) {
    throw new Error("Weight blob byte size must be divisible by 4 for FP32.");
  }
  const fp32Weights = new Float32Array(weightsBuffer as ArrayBuffer);
  const totalFloats = Math.trunc(meta.total_floats ?? -1);
  if (totalFloats <= 0 || fp32Weights.length !== totalFloats) {
    throw new Error(
      `Weight count mismatch: metadata total_floats=${totalFloats}, blob floats=${fp32Weights.length}.`
    );
  }

  const paddedCount = (fp32Weights.length + 1) & ~1; // storage buffers must stay 4-byte aligned
  const padded = new Float32Array(new ArrayBuffer(paddedCount * 4));
  padded.set(fp32Weights);
  const weightsFp16 = packFloat32ToFloat16(padded);

  const metaUniform = new Uint32Array<ArrayBuffer>(new ArrayBuffer(12 * 4));
  metaUniform[0] = inputDim;
  metaUniform[1] = hidden0;
  metaUniform[2] = hidden1;
  metaUniform[3] = w0Offset;
  metaUniform[4] = b0Offset;
  metaUniform[5] = w1Offset;
  metaUniform[6] = b1Offset;
  metaUniform[7] = w2Offset;
  metaUniform[8] = b2Offset;
  metaUniform[9] = fourierLevels;
  metaUniform[10] = 0;
  metaUniform[11] = 0;

  return { weightsFp16, metaUniform };
}

function validateLoadedModel(metaUniform: Uint32Array<ArrayBuffer>): void {
  if (metaUniform[0] === 0 || metaUniform[1] === 0 || metaUniform[2] === 0) {
    throw new Error("Loaded model metadata contains zero dimensions.");
  }
}

function normalize3(x: number, y: number, z: number): [number, number, number] {
  const len = Math.hypot(x, y, z) || 1;
  return [x / len, y / len, z / len];
}

function createLightingParamsBufferData(
  sunDirection: [number, number, number],
  sunIntensity: number,
  sigma: number,
  phaseG: number,
  albedo: number,
  solveDim: number,
  lambda: number,
  stepDistance: number,
  solveW: number
): Float32Array<ArrayBuffer> {
  const data = new Float32Array<ArrayBuffer>(new ArrayBuffer(12 * 4)); // 3 vec4
  data[0] = sunDirection[0];
  data[1] = sunDirection[1];
  data[2] = sunDirection[2];
  data[3] = sunIntensity;
  data[4] = 1.0; // density scale
  data[5] = phaseG;
  data[6] = albedo;
  data[7] = sigma; // extinction coefficient
  data[8] = solveDim;
  data[9] = lambda;
  data[10] = stepDistance;
  data[11] = solveW; // reused as march steps or iteration index depending on pass
  return data;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const denom = edge1 - edge0;
  if (Math.abs(denom) < 1e-6) {
    return x >= edge1 ? 1 : 0;
  }
  const t = clamp01((x - edge0) / denom);
  return t * t * (3 - 2 * t);
}

function fract(value: number): number {
  return value - Math.floor(value);
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function wrapIndex(value: number, size: number): number {
  const r = value % size;
  return r < 0 ? r + size : r;
}

function hash3ToUnit(x: number, y: number, z: number, seed: number): number {
  let h = (Math.imul(x, 374761393) ^ Math.imul(y, 668265263) ^ Math.imul(z, 2246822519) ^ seed) >>> 0;
  h = (h ^ (h >>> 13)) >>> 0;
  h = Math.imul(h, 1274126177) >>> 0;
  h = (h ^ (h >>> 16)) >>> 0;
  return h / 4294967295;
}

function valueNoise3D(x: number, y: number, z: number, period: number, seed: number): number {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const z0 = Math.floor(z);
  const x1 = x0 + 1;
  const y1 = y0 + 1;
  const z1 = z0 + 1;
  const tx = x - x0;
  const ty = y - y0;
  const tz = z - z0;
  const ux = smoothstep(0, 1, tx);
  const uy = smoothstep(0, 1, ty);
  const uz = smoothstep(0, 1, tz);

  const v000 = hash3ToUnit(wrapIndex(x0, period), wrapIndex(y0, period), wrapIndex(z0, period), seed);
  const v100 = hash3ToUnit(wrapIndex(x1, period), wrapIndex(y0, period), wrapIndex(z0, period), seed);
  const v010 = hash3ToUnit(wrapIndex(x0, period), wrapIndex(y1, period), wrapIndex(z0, period), seed);
  const v110 = hash3ToUnit(wrapIndex(x1, period), wrapIndex(y1, period), wrapIndex(z0, period), seed);
  const v001 = hash3ToUnit(wrapIndex(x0, period), wrapIndex(y0, period), wrapIndex(z1, period), seed);
  const v101 = hash3ToUnit(wrapIndex(x1, period), wrapIndex(y0, period), wrapIndex(z1, period), seed);
  const v011 = hash3ToUnit(wrapIndex(x0, period), wrapIndex(y1, period), wrapIndex(z1, period), seed);
  const v111 = hash3ToUnit(wrapIndex(x1, period), wrapIndex(y1, period), wrapIndex(z1, period), seed);

  const x00 = lerp(v000, v100, ux);
  const x10 = lerp(v010, v110, ux);
  const x01 = lerp(v001, v101, ux);
  const x11 = lerp(v011, v111, ux);
  const y0v = lerp(x00, x10, uy);
  const y1v = lerp(x01, x11, uy);
  return lerp(y0v, y1v, uz);
}

function fbmTileable(x: number, y: number, z: number, basePeriod: number): number {
  let sum = 0;
  let norm = 0;
  let amplitude = 0.5;
  for (let octave = 0; octave < 4; octave++) {
    const frequency = 1 << octave;
    const period = basePeriod * frequency;
    sum += amplitude * valueNoise3D(x * frequency, y * frequency, z * frequency, period, 1337 + octave * 911);
    norm += amplitude;
    amplitude *= 0.5;
  }
  return norm > 0 ? sum / norm : 0;
}

function worleyF1Tileable(ux: number, uy: number, uz: number, cellCount: number): number {
  const px = ux * cellCount;
  const py = uy * cellCount;
  const pz = uz * cellCount;
  const baseX = Math.floor(px);
  const baseY = Math.floor(py);
  const baseZ = Math.floor(pz);
  let minDistSq = 1e9;

  for (let dz = -1; dz <= 1; dz++) {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const cx = baseX + dx;
        const cy = baseY + dy;
        const cz = baseZ + dz;
        const wx = wrapIndex(cx, cellCount);
        const wy = wrapIndex(cy, cellCount);
        const wz = wrapIndex(cz, cellCount);
        const fx = hash3ToUnit(wx, wy, wz, 401);
        const fy = hash3ToUnit(wx, wy, wz, 997);
        const fz = hash3ToUnit(wx, wy, wz, 1597);
        const sx = cx + fx;
        const sy = cy + fy;
        const sz = cz + fz;
        const dxp = sx - px;
        const dyp = sy - py;
        const dzp = sz - pz;
        const distSq = dxp * dxp + dyp * dyp + dzp * dzp;
        if (distSq < minDistSq) {
          minDistSq = distSq;
        }
      }
    }
  }

  return Math.sqrt(minDistSq);
}

function createDetailNoiseVolumeData(size: number): Uint8Array<ArrayBuffer> {
  const voxelCount = size * size * size;
  const data = new Uint8Array<ArrayBuffer>(new ArrayBuffer(voxelCount * 4));
  const perlinPeriod = 8;
  const worleyCells = 7;
  let index = 0;

  for (let z = 0; z < size; z++) {
    const uz = (z + 0.5) / size;
    for (let y = 0; y < size; y++) {
      const uy = (y + 0.5) / size;
      for (let x = 0; x < size; x++) {
        const ux = (x + 0.5) / size;

        const perlin = fbmTileable(ux * perlinPeriod, uy * perlinPeriod, uz * perlinPeriod, perlinPeriod);
        const perlinShaped = smoothstep(0.22, 0.82, perlin);

        const f1 = worleyF1Tileable(ux, uy, uz, worleyCells);
        const worley = 1.0 - clamp01(f1 / 1.7320508075688772);
        const worleyShaped = smoothstep(0.08, 0.92, worley);

        data[index++] = Math.round(clamp01(perlinShaped) * 255);
        data[index++] = Math.round(clamp01(worleyShaped) * 255);
        data[index++] = 0;
        data[index++] = 255;
      }
    }
  }

  return data;
}

function createFillParamsBufferData(enableDetail: boolean): Float32Array<ArrayBuffer> {
  const data = new Float32Array<ArrayBuffer>(new ArrayBuffer(8 * 4)); // 2 vec4
  data[0] = DETAIL_SCALE;
  data[1] = DETAIL_EROSION_STRENGTH;
  data[2] = DETAIL_EDGE_START;
  data[3] = DETAIL_EDGE_END;
  data[4] = enableDetail ? 1 : 0;
  data[5] = 0;
  data[6] = 0;
  data[7] = 0;
  return data;
}

export class WebGPURenderer {
  private renderMode: RenderMode = "cloudSky";
  private sigma = DEFAULT_SIGMA;
  private readonly phaseG = DEFAULT_PHASE_G;
  private readonly sunIntensity = DEFAULT_SUN_INTENSITY;
  private sunDirection = normalize3(0.5, 0.78, 0.37);
  private detailNoiseEnabled = true;
  private volumeDirty = false;
  private lightingDirty = false;

  private constructor(
    private readonly device: GPUDevice,
    private readonly context: GPUCanvasContext,
    private readonly format: GPUTextureFormat,
    private readonly pipeline: GPURenderPipeline,
    private readonly bindGroup: GPUBindGroup,
    private readonly cameraBuffer: GPUBuffer,
    private readonly fillPipeline: GPUComputePipeline,
    private readonly fillBindGroup: GPUBindGroup,
    private readonly fillParamsBuffer: GPUBuffer,
    private readonly fillParamsData: Float32Array<ArrayBuffer>,
    private readonly sunTransmittancePipeline: GPUComputePipeline,
    private readonly sunTransmittanceBindGroup: GPUBindGroup,
    private readonly multiScatterPipeline: GPUComputePipeline,
    private readonly multiScatterBindGroupA: GPUBindGroup,
    private readonly multiScatterBindGroupB: GPUBindGroup,
    private readonly sunPassParamsBuffer: GPUBuffer,
    private readonly sunPassParamsData: Float32Array<ArrayBuffer>,
    private readonly multiScatterParamsBuffer: GPUBuffer,
    private readonly multiScatterParamsData: Float32Array<ArrayBuffer>
  ) {}

  static async create(canvas: HTMLCanvasElement): Promise<WebGPURenderer> {
    if (!navigator.gpu) {
      throw new Error("WebGPU not supported by this browser.");
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("Failed to get GPU adapter.");
    }
    if (!adapter.features.has("shader-f16")) {
      throw new Error("This GPU adapter does not support shader-f16, required for FP16 MLP compute.");
    }
    const device = await adapter.requestDevice({
      requiredFeatures: ["shader-f16"]
    });
    const context = canvas.getContext("webgpu");
    if (!context) {
      throw new Error("Failed to get WebGPU canvas context.");
    }
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
      device,
      format,
      alphaMode: "opaque"
    });

    const shader = device.createShaderModule({
      code: shaderSource
    });
    const fillVolumeShader = device.createShaderModule({
      code: fillVolumeSource
    });
    const sunTransmittanceShader = device.createShaderModule({
      code: sunTransmittanceSource
    });
    const multiScatterShader = device.createShaderModule({
      code: multiScatterSource
    });

    const cameraBuffer = device.createBuffer({
      size: CAMERA_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    const loadedMlp = await loadExportedMlpData();
    validateLoadedModel(loadedMlp.metaUniform);

    const volumeTexture = device.createTexture({
      size: {
        width: VOLUME_SIZE,
        height: VOLUME_SIZE,
        depthOrArrayLayers: VOLUME_SIZE
      },
      dimension: "3d",
      format: "rgba16float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
    });
    const volumeView = volumeTexture.createView({ dimension: "3d" });

    const sunTransmittanceTexture = device.createTexture({
      size: {
        width: VOLUME_SIZE,
        height: VOLUME_SIZE,
        depthOrArrayLayers: VOLUME_SIZE
      },
      dimension: "3d",
      format: "rgba16float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
    });
    const sunTransmittanceView = sunTransmittanceTexture.createView({ dimension: "3d" });

    const multiScatterTextureA = device.createTexture({
      size: {
        width: VOLUME_SIZE,
        height: VOLUME_SIZE,
        depthOrArrayLayers: VOLUME_SIZE
      },
      dimension: "3d",
      format: "rgba16float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
    });
    const multiScatterTextureB = device.createTexture({
      size: {
        width: VOLUME_SIZE,
        height: VOLUME_SIZE,
        depthOrArrayLayers: VOLUME_SIZE
      },
      dimension: "3d",
      format: "rgba16float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
    });
    const multiScatterViewA = multiScatterTextureA.createView({ dimension: "3d" });
    const multiScatterViewB = multiScatterTextureB.createView({ dimension: "3d" });

    const mlpWeightsBuffer = device.createBuffer({
      size: loadedMlp.weightsFp16.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(mlpWeightsBuffer, 0, loadedMlp.weightsFp16);

    const mlpMetaBuffer = device.createBuffer({
      size: loadedMlp.metaUniform.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(mlpMetaBuffer, 0, loadedMlp.metaUniform);

    const volumeSampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
      addressModeW: "clamp-to-edge"
    });

    const detailNoiseData = createDetailNoiseVolumeData(DETAIL_TEXTURE_SIZE);
    const detailNoiseTexture = device.createTexture({
      size: {
        width: DETAIL_TEXTURE_SIZE,
        height: DETAIL_TEXTURE_SIZE,
        depthOrArrayLayers: DETAIL_TEXTURE_SIZE
      },
      dimension: "3d",
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
    });
    device.queue.writeTexture(
      { texture: detailNoiseTexture },
      detailNoiseData,
      {
        bytesPerRow: DETAIL_TEXTURE_SIZE * 4,
        rowsPerImage: DETAIL_TEXTURE_SIZE
      },
      {
        width: DETAIL_TEXTURE_SIZE,
        height: DETAIL_TEXTURE_SIZE,
        depthOrArrayLayers: DETAIL_TEXTURE_SIZE
      }
    );
    const detailNoiseView = detailNoiseTexture.createView({ dimension: "3d" });
    const detailNoiseSampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "repeat",
      addressModeW: "repeat"
    });

    const fillParamsData = createFillParamsBufferData(true);
    const fillParamsBuffer = device.createBuffer({
      size: fillParamsData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(fillParamsBuffer, 0, fillParamsData);

    const lightingStepDistance = 2.0 / VOLUME_SIZE;
    const sunPassParamsData = createLightingParamsBufferData(
      normalize3(0.5, 0.78, 0.37),
      DEFAULT_SUN_INTENSITY,
      DEFAULT_SIGMA,
      DEFAULT_PHASE_G,
      DEFAULT_ALBEDO,
      VOLUME_SIZE,
      MULTISCATTER_LAMBDA,
      lightingStepDistance,
      LIGHT_MARCH_STEPS
    );
    const sunPassParamsBuffer = device.createBuffer({
      size: sunPassParamsData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(sunPassParamsBuffer, 0, sunPassParamsData);

    const multiScatterParamsData = createLightingParamsBufferData(
      normalize3(0.5, 0.78, 0.37),
      DEFAULT_SUN_INTENSITY,
      DEFAULT_SIGMA,
      DEFAULT_PHASE_G,
      DEFAULT_ALBEDO,
      VOLUME_SIZE,
      MULTISCATTER_LAMBDA,
      lightingStepDistance,
      0
    );
    const multiScatterParamsBuffer = device.createBuffer({
      size: multiScatterParamsData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(multiScatterParamsBuffer, 0, multiScatterParamsData);

    const fillPipeline = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: fillVolumeShader,
        entryPoint: "csMain"
      }
    });
    const fillBindGroup = device.createBindGroup({
      layout: fillPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: volumeView
        },
        {
          binding: 1,
          resource: { buffer: mlpWeightsBuffer }
        },
        {
          binding: 2,
          resource: { buffer: mlpMetaBuffer }
        },
        {
          binding: 3,
          resource: detailNoiseView
        },
        {
          binding: 4,
          resource: detailNoiseSampler
        },
        {
          binding: 5,
          resource: { buffer: fillParamsBuffer }
        }
      ]
    });

    const sunTransmittancePipeline = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: sunTransmittanceShader,
        entryPoint: "csMain"
      }
    });
    const sunTransmittanceBindGroup = device.createBindGroup({
      layout: sunTransmittancePipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: volumeView
        },
        {
          binding: 2,
          resource: sunTransmittanceView
        },
        {
          binding: 3,
          resource: { buffer: sunPassParamsBuffer }
        }
      ]
    });

    const multiScatterPipeline = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: multiScatterShader,
        entryPoint: "csMain"
      }
    });
    const multiScatterBindGroupA = device.createBindGroup({
      layout: multiScatterPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: multiScatterViewA
        },
        {
          binding: 1,
          resource: multiScatterViewB
        },
        {
          binding: 2,
          resource: sunTransmittanceView
        },
        {
          binding: 3,
          resource: volumeView
        },
        {
          binding: 4,
          resource: volumeSampler
        },
        {
          binding: 5,
          resource: { buffer: multiScatterParamsBuffer }
        }
      ]
    });
    const multiScatterBindGroupB = device.createBindGroup({
      layout: multiScatterPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: multiScatterViewB
        },
        {
          binding: 1,
          resource: multiScatterViewA
        },
        {
          binding: 2,
          resource: sunTransmittanceView
        },
        {
          binding: 3,
          resource: volumeView
        },
        {
          binding: 4,
          resource: volumeSampler
        },
        {
          binding: 5,
          resource: { buffer: multiScatterParamsBuffer }
        }
      ]
    });

    const precomputeEncoder = device.createCommandEncoder();

    const fillPass = precomputeEncoder.beginComputePass();
    fillPass.setPipeline(fillPipeline);
    fillPass.setBindGroup(0, fillBindGroup);
    fillPass.dispatchWorkgroups(VOLUME_SIZE / 4, VOLUME_SIZE / 4, VOLUME_SIZE / 4);
    fillPass.end();

    const sunPass = precomputeEncoder.beginComputePass();
    sunPass.setPipeline(sunTransmittancePipeline);
    sunPass.setBindGroup(0, sunTransmittanceBindGroup);
    sunPass.dispatchWorkgroups(VOLUME_SIZE / 4, VOLUME_SIZE / 4, VOLUME_SIZE / 4);
    sunPass.end();

    // First multiscatter iteration seeds the field from source only.
    multiScatterParamsData[11] = 0;
    device.queue.writeBuffer(multiScatterParamsBuffer, 0, multiScatterParamsData);
    const msSeedPass = precomputeEncoder.beginComputePass();
    msSeedPass.setPipeline(multiScatterPipeline);
    msSeedPass.setBindGroup(0, multiScatterBindGroupA);
    msSeedPass.dispatchWorkgroups(VOLUME_SIZE / 4, VOLUME_SIZE / 4, VOLUME_SIZE / 4);
    msSeedPass.end();

    device.queue.submit([precomputeEncoder.finish()]);

    for (let i = 1; i < MULTISCATTER_ITERS; i++) {
      multiScatterParamsData[11] = i;
      device.queue.writeBuffer(multiScatterParamsBuffer, 0, multiScatterParamsData);
      const msIterEncoder = device.createCommandEncoder();
      const msIterPass = msIterEncoder.beginComputePass();
      msIterPass.setPipeline(multiScatterPipeline);
      msIterPass.setBindGroup(0, (i & 1) === 0 ? multiScatterBindGroupA : multiScatterBindGroupB);
      msIterPass.dispatchWorkgroups(VOLUME_SIZE / 4, VOLUME_SIZE / 4, VOLUME_SIZE / 4);
      msIterPass.end();
      device.queue.submit([msIterEncoder.finish()]);
    }

    const finalMultiScatterView = (MULTISCATTER_ITERS & 1) === 0 ? multiScatterViewB : multiScatterViewA;

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" }
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: "filtering" }
        },
        {
          binding: 2,
          visibility: GPUShaderStage.FRAGMENT,
          texture: {
            sampleType: "float",
            viewDimension: "3d"
          }
        },
        {
          binding: 3,
          visibility: GPUShaderStage.FRAGMENT,
          texture: {
            sampleType: "float",
            viewDimension: "3d"
          }
        },
        {
          binding: 4,
          visibility: GPUShaderStage.FRAGMENT,
          texture: {
            sampleType: "float",
            viewDimension: "3d"
          }
        }
      ]
    });

    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: { buffer: cameraBuffer }
        },
        {
          binding: 1,
          resource: volumeSampler
        },
        {
          binding: 2,
          resource: volumeView
        },
        {
          binding: 3,
          resource: sunTransmittanceView
        },
        {
          binding: 4,
          resource: finalMultiScatterView
        }
      ]
    });

    const pipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      vertex: {
        module: shader,
        entryPoint: "vsMain"
      },
      fragment: {
        module: shader,
        entryPoint: "fsMain",
        targets: [{ format }]
      },
      primitive: {
        topology: "triangle-list"
      }
    });

    return new WebGPURenderer(
      device,
      context,
      format,
      pipeline,
      bindGroup,
      cameraBuffer,
      fillPipeline,
      fillBindGroup,
      fillParamsBuffer,
      fillParamsData,
      sunTransmittancePipeline,
      sunTransmittanceBindGroup,
      multiScatterPipeline,
      multiScatterBindGroupA,
      multiScatterBindGroupB,
      sunPassParamsBuffer,
      sunPassParamsData,
      multiScatterParamsBuffer,
      multiScatterParamsData
    );
  }

  setRenderMode(mode: RenderMode): void {
    this.renderMode = mode;
  }

  setCloudDensity(value: number): void {
    if (!Number.isFinite(value)) {
      return;
    }
    this.sigma = Math.max(0.01, Math.min(value, 12.0));
    this.lightingDirty = true;
  }

  setDetailNoiseEnabled(enabled: boolean): void {
    const next = Boolean(enabled);
    if (this.detailNoiseEnabled === next) {
      return;
    }
    this.detailNoiseEnabled = next;
    this.volumeDirty = true;
  }

  setSunAngles(pitchDegrees: number, azimuthDegrees: number): void {
    if (!Number.isFinite(pitchDegrees) || !Number.isFinite(azimuthDegrees)) {
      return;
    }
    const pitch = Math.max(-89.0, Math.min(89.0, pitchDegrees)) * (Math.PI / 180.0);
    const azimuth = azimuthDegrees * (Math.PI / 180.0);
    const cosPitch = Math.cos(pitch);
    this.sunDirection = normalize3(
      cosPitch * Math.cos(azimuth),
      Math.sin(pitch),
      cosPitch * Math.sin(azimuth)
    );
    this.lightingDirty = true;
  }

  private updateVolumePrecompute(): void {
    this.fillParamsData[4] = this.detailNoiseEnabled ? 1 : 0;
    this.device.queue.writeBuffer(this.fillParamsBuffer, 0, this.fillParamsData);

    const fillEncoder = this.device.createCommandEncoder();
    const fillPass = fillEncoder.beginComputePass();
    fillPass.setPipeline(this.fillPipeline);
    fillPass.setBindGroup(0, this.fillBindGroup);
    fillPass.dispatchWorkgroups(VOLUME_SIZE / 4, VOLUME_SIZE / 4, VOLUME_SIZE / 4);
    fillPass.end();
    this.device.queue.submit([fillEncoder.finish()]);

    this.updateLightingPrecompute();
    this.volumeDirty = false;
  }

  private updateLightingPrecompute(): void {
    this.sunPassParamsData[0] = this.sunDirection[0];
    this.sunPassParamsData[1] = this.sunDirection[1];
    this.sunPassParamsData[2] = this.sunDirection[2];
    this.sunPassParamsData[3] = this.sunIntensity;
    this.sunPassParamsData[7] = this.sigma;
    this.sunPassParamsData[11] = LIGHT_MARCH_STEPS;
    this.device.queue.writeBuffer(this.sunPassParamsBuffer, 0, this.sunPassParamsData);

    this.multiScatterParamsData[0] = this.sunDirection[0];
    this.multiScatterParamsData[1] = this.sunDirection[1];
    this.multiScatterParamsData[2] = this.sunDirection[2];
    this.multiScatterParamsData[3] = this.sunIntensity;
    this.multiScatterParamsData[7] = this.sigma;
    this.multiScatterParamsData[11] = 0;
    this.device.queue.writeBuffer(this.multiScatterParamsBuffer, 0, this.multiScatterParamsData);

    const precomputeEncoder = this.device.createCommandEncoder();

    const sunPass = precomputeEncoder.beginComputePass();
    sunPass.setPipeline(this.sunTransmittancePipeline);
    sunPass.setBindGroup(0, this.sunTransmittanceBindGroup);
    sunPass.dispatchWorkgroups(VOLUME_SIZE / 4, VOLUME_SIZE / 4, VOLUME_SIZE / 4);
    sunPass.end();

    const msSeedPass = precomputeEncoder.beginComputePass();
    msSeedPass.setPipeline(this.multiScatterPipeline);
    msSeedPass.setBindGroup(0, this.multiScatterBindGroupA);
    msSeedPass.dispatchWorkgroups(VOLUME_SIZE / 4, VOLUME_SIZE / 4, VOLUME_SIZE / 4);
    msSeedPass.end();

    this.device.queue.submit([precomputeEncoder.finish()]);

    for (let i = 1; i < MULTISCATTER_ITERS; i++) {
      this.multiScatterParamsData[11] = i;
      this.device.queue.writeBuffer(this.multiScatterParamsBuffer, 0, this.multiScatterParamsData);
      const iterEncoder = this.device.createCommandEncoder();
      const iterPass = iterEncoder.beginComputePass();
      iterPass.setPipeline(this.multiScatterPipeline);
      iterPass.setBindGroup(0, (i & 1) === 0 ? this.multiScatterBindGroupA : this.multiScatterBindGroupB);
      iterPass.dispatchWorkgroups(VOLUME_SIZE / 4, VOLUME_SIZE / 4, VOLUME_SIZE / 4);
      iterPass.end();
      this.device.queue.submit([iterEncoder.finish()]);
    }

    this.lightingDirty = false;
  }

  render(camera: OrbitCamera): void {
    if (this.volumeDirty) {
      this.updateVolumePrecompute();
    } else if (this.lightingDirty) {
      this.updateLightingPrecompute();
    }

    const pos = camera.getPosition();
    const target = camera.getTarget();
    const up = camera.getUp();
    const [near, far] = camera.getNearFar();
    const modeFlag = this.renderMode === "cloudSky" ? 1 : 0;

    const uniformData = new Float32Array([
      pos[0],
      pos[1],
      pos[2],
      0,
      target[0],
      target[1],
      target[2],
      0,
      up[0],
      up[1],
      up[2],
      0,
      camera.getAspect(),
      camera.getTanHalfFov(),
      near,
      far,
      modeFlag,
      this.sigma,
      this.phaseG,
      0,
      this.sunDirection[0],
      this.sunDirection[1],
      this.sunDirection[2],
      this.sunIntensity
    ]);
    this.device.queue.writeBuffer(this.cameraBuffer, 0, uniformData);

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          loadOp: "clear",
          storeOp: "store",
          clearValue: { r: 0, g: 0, b: 0, a: 1 }
        }
      ]
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(6, 1, 0, 0);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}
