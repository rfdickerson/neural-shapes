import { OrbitCamera } from "./webgpu/orbitCamera";
import { WebGPURenderer, type RenderMode } from "./webgpu/renderer";

export class App {
  private readonly canvas: HTMLCanvasElement;
  private readonly modeSelect: HTMLSelectElement;
  private readonly densitySlider: HTMLInputElement;
  private readonly densityValue: HTMLSpanElement;
  private renderer: WebGPURenderer | null = null;
  private readonly camera: OrbitCamera;

  constructor(container: HTMLElement) {
    const controls = document.createElement("div");
    controls.className = "controls";

    const label = document.createElement("label");
    label.className = "controls__label";
    label.htmlFor = "render-mode";
    label.textContent = "Render Mode";

    this.modeSelect = document.createElement("select");
    this.modeSelect.id = "render-mode";
    this.modeSelect.className = "controls__select";
    this.modeSelect.innerHTML = `
      <option value="cloudSky">Cloud + Sky</option>
      <option value="fogOnly">Fog Only</option>
    `;

    const densityLabel = document.createElement("label");
    densityLabel.className = "controls__label";
    densityLabel.htmlFor = "density-slider";
    densityLabel.textContent = "Density";

    this.densitySlider = document.createElement("input");
    this.densitySlider.id = "density-slider";
    this.densitySlider.className = "controls__range";
    this.densitySlider.type = "range";
    this.densitySlider.min = "0.2";
    this.densitySlider.max = "8.0";
    this.densitySlider.step = "0.05";
    this.densitySlider.value = "4.0";

    this.densityValue = document.createElement("span");
    this.densityValue.className = "controls__value";
    this.densityValue.textContent = Number(this.densitySlider.value).toFixed(2);

    controls.append(label, this.modeSelect, densityLabel, this.densitySlider, this.densityValue);
    container.appendChild(controls);

    this.canvas = document.createElement("canvas");
    this.canvas.className = "viewport";
    container.appendChild(this.canvas);
    this.camera = new OrbitCamera(this.canvas);

    this.modeSelect.addEventListener("change", () => {
      const mode = this.modeSelect.value as RenderMode;
      this.renderer?.setRenderMode(mode);
    });

    this.densitySlider.addEventListener("input", () => {
      const value = Number(this.densitySlider.value);
      this.densityValue.textContent = value.toFixed(2);
      this.renderer?.setCloudDensity(value);
    });
  }

  async start(): Promise<void> {
    this.renderer = await WebGPURenderer.create(this.canvas);
    this.renderer.setRenderMode(this.modeSelect.value as RenderMode);
    this.renderer.setCloudDensity(Number(this.densitySlider.value));
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
