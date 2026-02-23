import shaderSource from "./shaders/raymarch.wgsl?raw";
import fillVolumeSource from "./shaders/fillVolume.wgsl?raw";
import type { OrbitCamera } from "./orbitCamera";

const CAMERA_UNIFORM_BYTES = 64;
const VOLUME_SIZE = 64;
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
    const device = await adapter.requestDevice();
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
