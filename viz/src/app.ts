import { OrbitCamera } from "./webgpu/orbitCamera";
import { WebGPURenderer } from "./webgpu/renderer";

export class App {
  private readonly canvas: HTMLCanvasElement;
  private renderer: WebGPURenderer | null = null;
  private readonly camera: OrbitCamera;

  constructor(container: HTMLElement) {
    this.canvas = document.createElement("canvas");
    this.canvas.className = "viewport";
    container.appendChild(this.canvas);
    this.camera = new OrbitCamera(this.canvas);
  }

  async start(): Promise<void> {
    this.renderer = await WebGPURenderer.create(this.canvas);
    window.addEventListener("resize", this.handleResize);
    this.handleResize();
    requestAnimationFrame(this.frame);
  }

  private readonly handleResize = (): void => {
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const width = Math.max(1, Math.floor(window.innerWidth * dpr));
    const height = Math.max(1, Math.floor(window.innerHeight * dpr));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    this.camera.setViewportSize(width, height);
  };

  private readonly frame = (_time: number): void => {
    if (!this.renderer) {
      return;
    }
    this.camera.update();
    this.renderer.render(this.camera);
    requestAnimationFrame(this.frame);
  };
}
