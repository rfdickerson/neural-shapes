import { OrbitCamera } from "./webgpu/orbitCamera";
import { WebGPURenderer, type RenderMode } from "./webgpu/renderer";

const DEFAULT_SUN_PITCH = 50;
const DEFAULT_SUN_AZIMUTH = 25;
const MAX_RENDER_DPR = 1.5;

export class App {
  private readonly canvas: HTMLCanvasElement;
  private readonly modeSelect: HTMLSelectElement;
  private readonly densitySlider: HTMLInputElement;
  private readonly densityValue: HTMLSpanElement;
  private readonly sunPitchSlider: HTMLInputElement;
  private readonly sunPitchValue: HTMLSpanElement;
  private readonly sunAzimuthSlider: HTMLInputElement;
  private readonly sunAzimuthValue: HTMLSpanElement;
  private renderer: WebGPURenderer | null = null;
  private readonly camera: OrbitCamera;
  private needsRender = true;

  constructor(container: HTMLElement) {
    const controlsStack = document.createElement("div");
    controlsStack.className = "controls-stack";

    const cloudPanel = document.createElement("section");
    cloudPanel.className = "controls-panel";

    const cloudTitle = document.createElement("h3");
    cloudTitle.className = "controls__title";
    cloudTitle.textContent = "Cloud Parameters";
    cloudPanel.appendChild(cloudTitle);

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
    const modeRow = document.createElement("div");
    modeRow.className = "controls__row";
    modeRow.append(label, this.modeSelect);
    cloudPanel.appendChild(modeRow);

    const densityLabel = document.createElement("label");
    densityLabel.className = "controls__label";
    densityLabel.htmlFor = "density-slider";
    densityLabel.textContent = "Density";

    this.densitySlider = document.createElement("input");
    this.densitySlider.id = "density-slider";
    this.densitySlider.className = "controls__range";
    this.densitySlider.type = "range";
    this.densitySlider.min = "0.2";
    this.densitySlider.max = "12.0";
    this.densitySlider.step = "0.05";
    this.densitySlider.value = "5.85";

    this.densityValue = document.createElement("span");
    this.densityValue.className = "controls__value";
    this.densityValue.textContent = Number(this.densitySlider.value).toFixed(2);
    const densityField = document.createElement("div");
    densityField.className = "controls__field";
    densityField.append(this.densitySlider, this.densityValue);
    const densityRow = document.createElement("div");
    densityRow.className = "controls__row";
    densityRow.append(densityLabel, densityField);
    cloudPanel.appendChild(densityRow);

    const sunPanel = document.createElement("section");
    sunPanel.className = "controls-panel";

    const sunTitle = document.createElement("h3");
    sunTitle.className = "controls__title";
    sunTitle.textContent = "Sun Controls";
    sunPanel.appendChild(sunTitle);

    const pitchLabel = document.createElement("label");
    pitchLabel.className = "controls__label";
    pitchLabel.htmlFor = "sun-pitch-slider";
    pitchLabel.textContent = "Sun Pitch";

    this.sunPitchSlider = document.createElement("input");
    this.sunPitchSlider.id = "sun-pitch-slider";
    this.sunPitchSlider.className = "controls__range";
    this.sunPitchSlider.type = "range";
    this.sunPitchSlider.min = "-89";
    this.sunPitchSlider.max = "89";
    this.sunPitchSlider.step = "1";
    this.sunPitchSlider.value = String(DEFAULT_SUN_PITCH);

    this.sunPitchValue = document.createElement("span");
    this.sunPitchValue.className = "controls__value";
    this.sunPitchValue.textContent = `${Math.round(Number(this.sunPitchSlider.value))}deg`;
    const pitchField = document.createElement("div");
    pitchField.className = "controls__field";
    pitchField.append(this.sunPitchSlider, this.sunPitchValue);
    const pitchRow = document.createElement("div");
    pitchRow.className = "controls__row";
    pitchRow.append(pitchLabel, pitchField);
    sunPanel.appendChild(pitchRow);

    const azimuthLabel = document.createElement("label");
    azimuthLabel.className = "controls__label";
    azimuthLabel.htmlFor = "sun-azimuth-slider";
    azimuthLabel.textContent = "Sun Azimuth";

    this.sunAzimuthSlider = document.createElement("input");
    this.sunAzimuthSlider.id = "sun-azimuth-slider";
    this.sunAzimuthSlider.className = "controls__range";
    this.sunAzimuthSlider.type = "range";
    this.sunAzimuthSlider.min = "-180";
    this.sunAzimuthSlider.max = "180";
    this.sunAzimuthSlider.step = "1";
    this.sunAzimuthSlider.value = String(DEFAULT_SUN_AZIMUTH);

    this.sunAzimuthValue = document.createElement("span");
    this.sunAzimuthValue.className = "controls__value";
    this.sunAzimuthValue.textContent = `${Math.round(Number(this.sunAzimuthSlider.value))}deg`;
    const azimuthField = document.createElement("div");
    azimuthField.className = "controls__field";
    azimuthField.append(this.sunAzimuthSlider, this.sunAzimuthValue);
    const azimuthRow = document.createElement("div");
    azimuthRow.className = "controls__row";
    azimuthRow.append(azimuthLabel, azimuthField);
    sunPanel.appendChild(azimuthRow);

    controlsStack.append(cloudPanel, sunPanel);
    container.appendChild(controlsStack);

    this.canvas = document.createElement("canvas");
    this.canvas.className = "viewport";
    container.appendChild(this.canvas);
    this.camera = new OrbitCamera(this.canvas);

    this.modeSelect.addEventListener("change", () => {
      const mode = this.modeSelect.value as RenderMode;
      this.renderer?.setRenderMode(mode);
      this.needsRender = true;
    });

    this.densitySlider.addEventListener("input", () => {
      const value = Number(this.densitySlider.value);
      this.densityValue.textContent = value.toFixed(2);
      this.renderer?.setCloudDensity(value);
      this.needsRender = true;
    });

    this.sunPitchSlider.addEventListener("input", () => {
      const pitch = Number(this.sunPitchSlider.value);
      this.sunPitchValue.textContent = `${Math.round(pitch)}deg`;
      this.renderer?.setSunAngles(pitch, Number(this.sunAzimuthSlider.value));
      this.needsRender = true;
    });

    this.sunAzimuthSlider.addEventListener("input", () => {
      const azimuth = Number(this.sunAzimuthSlider.value);
      this.sunAzimuthValue.textContent = `${Math.round(azimuth)}deg`;
      this.renderer?.setSunAngles(Number(this.sunPitchSlider.value), azimuth);
      this.needsRender = true;
    });
  }

  async start(): Promise<void> {
    this.renderer = await WebGPURenderer.create(this.canvas);
    this.renderer.setDensitySource("neural");
    this.renderer.setRenderMode(this.modeSelect.value as RenderMode);
    this.renderer.setCloudDensity(Number(this.densitySlider.value));
    this.renderer.setSunAngles(Number(this.sunPitchSlider.value), Number(this.sunAzimuthSlider.value));
    this.needsRender = true;
    window.addEventListener("resize", this.handleResize);
    this.handleResize();
    requestAnimationFrame(this.frame);
  }

  private readonly handleResize = (): void => {
    const dpr = Math.min(MAX_RENDER_DPR, Math.max(1, window.devicePixelRatio || 1));
    const width = Math.max(1, Math.floor(window.innerWidth * dpr));
    const height = Math.max(1, Math.floor(window.innerHeight * dpr));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.needsRender = true;
    }
    this.camera.setViewportSize(width, height);
  };

  private readonly frame = (_time: number): void => {
    if (!this.renderer) {
      return;
    }
    const cameraChanged = this.camera.update();
    if (cameraChanged) {
      this.needsRender = true;
    }
    if (this.needsRender || this.renderer.hasPendingRenderWork()) {
      this.renderer.render(this.camera);
      this.needsRender = false;
    }
    requestAnimationFrame(this.frame);
  };
}
