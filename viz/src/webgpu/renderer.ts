import shaderSource from "./shaders/raymarch.wgsl?raw";
import fillVolumeSource from "./shaders/fillVolume.wgsl?raw";
import type { OrbitCamera } from "./orbitCamera";

const CAMERA_UNIFORM_BYTES = 64;
const VOLUME_SIZE = 64;
const SHADER_MAX_INPUT_DIM = 27;
const SHADER_MAX_HIDDEN = 64;
const MLP_META_URL = "/mlp/residual_mlp_metadata.json";
const MLP_WEIGHTS_URL = "/mlp/residual_mlp_weights.bin";
const EXPECTED_ENCODING_ORDER = "input_xyz_then_per_level_sin_xyz_cos_xyz";

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

export class WebGPURenderer {
  private constructor(
    private readonly device: GPUDevice,
    private readonly context: GPUCanvasContext,
    private readonly format: GPUTextureFormat,
    private readonly pipeline: GPURenderPipeline,
    private readonly bindGroup: GPUBindGroup,
    private readonly cameraBuffer: GPUBuffer
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
        }
      ]
    });

    const fillEncoder = device.createCommandEncoder();
    const fillPass = fillEncoder.beginComputePass();
    fillPass.setPipeline(fillPipeline);
    fillPass.setBindGroup(0, fillBindGroup);
    fillPass.dispatchWorkgroups(VOLUME_SIZE / 4, VOLUME_SIZE / 4, VOLUME_SIZE / 4);
    fillPass.end();
    device.queue.submit([fillEncoder.finish()]);

    const volumeSampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
      addressModeW: "clamp-to-edge"
    });

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

    return new WebGPURenderer(device, context, format, pipeline, bindGroup, cameraBuffer);
  }

  render(camera: OrbitCamera): void {
    const pos = camera.getPosition();
    const target = camera.getTarget();
    const up = camera.getUp();
    const [near, far] = camera.getNearFar();

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
      far
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
