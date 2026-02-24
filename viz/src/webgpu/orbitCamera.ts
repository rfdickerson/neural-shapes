import { normalize, vec3, type Vec3 } from "./math";

export class OrbitCamera {
  private yaw = 0.9;
  private pitch = 0.35;
  private radius = 2.0;
  private viewportWidth = 1;
  private viewportHeight = 1;
  private isDragging = false;
  private lastX = 0;
  private lastY = 0;
  private readonly target: Vec3 = [0, 0, 0];
  private readonly up: Vec3 = [0, 1, 0];
  private readonly minRadius = 0.8;
  private readonly maxRadius = 8.0;
  private readonly minPitch = -1.45;
  private readonly maxPitch = 1.45;
  private dirty = true;

  constructor(canvas: HTMLCanvasElement) {
    canvas.addEventListener("pointerdown", (ev) => {
      this.isDragging = true;
      this.lastX = ev.clientX;
      this.lastY = ev.clientY;
      canvas.setPointerCapture(ev.pointerId);
    });

    canvas.addEventListener("pointerup", (ev) => {
      this.isDragging = false;
      canvas.releasePointerCapture(ev.pointerId);
    });

    canvas.addEventListener("pointermove", (ev) => {
      if (!this.isDragging) {
        return;
      }
      const dx = ev.clientX - this.lastX;
      const dy = ev.clientY - this.lastY;
      this.lastX = ev.clientX;
      this.lastY = ev.clientY;

      this.yaw += dx * 0.005;
      this.pitch -= dy * 0.005;
      this.pitch = Math.max(this.minPitch, Math.min(this.maxPitch, this.pitch));
      this.dirty = true;
    });

    canvas.addEventListener(
      "wheel",
      (ev) => {
        ev.preventDefault();
        const delta = ev.deltaY * 0.0015;
        this.radius = Math.max(this.minRadius, Math.min(this.maxRadius, this.radius * (1 + delta)));
        this.dirty = true;
      },
      { passive: false }
    );
  }

  setViewportSize(width: number, height: number): void {
    this.viewportWidth = Math.max(1, width);
    this.viewportHeight = Math.max(1, height);
    this.dirty = true;
  }

  update(): boolean {
    const changed = this.dirty;
    this.dirty = false;
    return changed;
  }

  getPosition(): Vec3 {
    const cp = Math.cos(this.pitch);
    const sp = Math.sin(this.pitch);
    const cy = Math.cos(this.yaw);
    const sy = Math.sin(this.yaw);
    return vec3(this.radius * cp * cy, this.radius * sp, this.radius * cp * sy);
  }

  getTarget(): Vec3 {
    return this.target;
  }

  getUp(): Vec3 {
    return normalize(this.up);
  }

  getAspect(): number {
    return this.viewportWidth / this.viewportHeight;
  }

  getTanHalfFov(): number {
    const fovY = (55 * Math.PI) / 180;
    return Math.tan(0.5 * fovY);
  }

  getNearFar(): [number, number] {
    return [0.05, 8.0];
  }
}
