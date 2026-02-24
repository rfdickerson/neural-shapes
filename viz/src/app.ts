import { OrbitCamera } from "./webgpu/orbitCamera";
import { WebGPURenderer, type RenderMode } from "./webgpu/renderer";

const DEFAULT_SUN_PITCH = 51;
const DEFAULT_SUN_AZIMUTH = 36;
const DEFAULT_DETAIL_SCALE = 4.6;
const DEFAULT_EROSION_STRENGTH = 0.48;

export class App {
  private readonly canvas: HTMLCanvasElement;
  private readonly modeSelect: HTMLSelectElement;
  private readonly detailNoiseToggle: HTMLInputElement;
  private readonly detailScaleSlider: HTMLInputElement;
  private readonly detailScaleValue: HTMLSpanElement;
  private readonly erosionStrengthSlider: HTMLInputElement;
  private readonly erosionStrengthValue: HTMLSpanElement;
  private readonly densitySlider: HTMLInputElement;
  private readonly densityValue: HTMLSpanElement;
  private readonly sunPitchSlider: HTMLInputElement;
  private readonly sunPitchValue: HTMLSpanElement;
  private readonly sunAzimuthSlider: HTMLInputElement;
  private readonly sunAzimuthValue: HTMLSpanElement;
  private renderer: WebGPURenderer | null = null;
  private readonly camera: OrbitCamera;

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

    const detailLabel = document.createElement("label");
    detailLabel.className = "controls__label";
    detailLabel.htmlFor = "detail-noise-toggle";
    detailLabel.textContent = "Detail Noise";

    this.detailNoiseToggle = document.createElement("input");
    this.detailNoiseToggle.id = "detail-noise-toggle";
    this.detailNoiseToggle.type = "checkbox";
    this.detailNoiseToggle.className = "controls__checkbox";
    this.detailNoiseToggle.checked = true;
    const detailRow = document.createElement("div");
    detailRow.className = "controls__row controls__row--toggle";
    detailRow.append(detailLabel, this.detailNoiseToggle);
    cloudPanel.appendChild(detailRow);

    const detailScaleLabel = document.createElement("label");
    detailScaleLabel.className = "controls__label";
    detailScaleLabel.htmlFor = "detail-scale-slider";
    detailScaleLabel.textContent = "Detail Scale";

    this.detailScaleSlider = document.createElement("input");
    this.detailScaleSlider.id = "detail-scale-slider";
    this.detailScaleSlider.className = "controls__range";
    this.detailScaleSlider.type = "range";
    this.detailScaleSlider.min = "0.5";
    this.detailScaleSlider.max = "10.0";
    this.detailScaleSlider.step = "0.05";
    this.detailScaleSlider.value = DEFAULT_DETAIL_SCALE.toFixed(2);

    this.detailScaleValue = document.createElement("span");
    this.detailScaleValue.className = "controls__value";
    this.detailScaleValue.textContent = Number(this.detailScaleSlider.value).toFixed(2);
    const detailScaleField = document.createElement("div");
    detailScaleField.className = "controls__field";
    detailScaleField.append(this.detailScaleSlider, this.detailScaleValue);
    const detailScaleRow = document.createElement("div");
    detailScaleRow.className = "controls__row";
    detailScaleRow.append(detailScaleLabel, detailScaleField);
    cloudPanel.appendChild(detailScaleRow);

    const erosionStrengthLabel = document.createElement("label");
    erosionStrengthLabel.className = "controls__label";
    erosionStrengthLabel.htmlFor = "erosion-strength-slider";
    erosionStrengthLabel.textContent = "Erosion Strength";

    this.erosionStrengthSlider = document.createElement("input");
    this.erosionStrengthSlider.id = "erosion-strength-slider";
    this.erosionStrengthSlider.className = "controls__range";
    this.erosionStrengthSlider.type = "range";
    this.erosionStrengthSlider.min = "0.0";
    this.erosionStrengthSlider.max = "1.5";
    this.erosionStrengthSlider.step = "0.02";
    this.erosionStrengthSlider.value = DEFAULT_EROSION_STRENGTH.toFixed(2);

    this.erosionStrengthValue = document.createElement("span");
    this.erosionStrengthValue.className = "controls__value";
    this.erosionStrengthValue.textContent = Number(this.erosionStrengthSlider.value).toFixed(2);
    const erosionStrengthField = document.createElement("div");
    erosionStrengthField.className = "controls__field";
    erosionStrengthField.append(this.erosionStrengthSlider, this.erosionStrengthValue);
    const erosionStrengthRow = document.createElement("div");
    erosionStrengthRow.className = "controls__row";
    erosionStrengthRow.append(erosionStrengthLabel, erosionStrengthField);
    cloudPanel.appendChild(erosionStrengthRow);

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
    this.densitySlider.value = "4.0";

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
    });

    this.detailNoiseToggle.addEventListener("change", () => {
      this.renderer?.setDetailNoiseEnabled(this.detailNoiseToggle.checked);
    });

    this.detailScaleSlider.addEventListener("input", () => {
      const value = Number(this.detailScaleSlider.value);
      this.detailScaleValue.textContent = value.toFixed(2);
      this.renderer?.setDetailScale(value);
    });

    this.erosionStrengthSlider.addEventListener("input", () => {
      const value = Number(this.erosionStrengthSlider.value);
      this.erosionStrengthValue.textContent = value.toFixed(2);
      this.renderer?.setErosionStrength(value);
    });

    this.densitySlider.addEventListener("input", () => {
      const value = Number(this.densitySlider.value);
      this.densityValue.textContent = value.toFixed(2);
      this.renderer?.setCloudDensity(value);
    });

    this.sunPitchSlider.addEventListener("input", () => {
      const pitch = Number(this.sunPitchSlider.value);
      this.sunPitchValue.textContent = `${Math.round(pitch)}deg`;
      this.renderer?.setSunAngles(pitch, Number(this.sunAzimuthSlider.value));
    });

    this.sunAzimuthSlider.addEventListener("input", () => {
      const azimuth = Number(this.sunAzimuthSlider.value);
      this.sunAzimuthValue.textContent = `${Math.round(azimuth)}deg`;
      this.renderer?.setSunAngles(Number(this.sunPitchSlider.value), azimuth);
    });
  }

  async start(): Promise<void> {
    this.renderer = await WebGPURenderer.create(this.canvas);
    this.renderer.setRenderMode(this.modeSelect.value as RenderMode);
    this.renderer.setDetailNoiseEnabled(this.detailNoiseToggle.checked);
    this.renderer.setDetailScale(Number(this.detailScaleSlider.value));
    this.renderer.setErosionStrength(Number(this.erosionStrengthSlider.value));
    this.renderer.setCloudDensity(Number(this.densitySlider.value));
    this.renderer.setSunAngles(Number(this.sunPitchSlider.value), Number(this.sunAzimuthSlider.value));
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
