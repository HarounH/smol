"""Interactive blank 2D canvas with simple camera controls."""

import contextlib
import math
import os
import tempfile
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import logging
import time

import imageio
import numpy as np
from PIL import ImageGrab

try:
    import mss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    mss = None

setup_done: bool = False

def setup_logger() -> None:
    global setup_done
    if setup_done:
        return
    setup_done = True
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

setup_logger()
logger = logging.getLogger(__name__)

Color = Tuple[int, int, int]


@dataclass
class Camera:
    """Track camera position and zoom."""

    x: float = 0.0
    y: float = 0.0
    zoom: float = 1.0

    def pan(self, dx: float, dy: float) -> None:
        self.x += dx
        self.y += dy

    def set_zoom(self, factor: float) -> None:
        self.zoom = max(0.1, factor)

    def reset(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.zoom = 1.0


class BaseCanvas:
    """Minimal interactive canvas ready for future drawing.
    Intended usage: inherit, override init, draw.
    """

    def __init__(
        self,
        title: str = "Smol Art - Blank Canvas",
        width: int = 640,
        height: int = 480,
        background: Color = (240, 240, 240),
        # Animation related things
        draw_time_ms: int = 16,
        automated_draw_time_estimation: bool = False,
        verbose: bool = False,
        save_dir: str | None = "./outputs",
        temp_dir: Optional[str] = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Canvas dimensions must be positive")
        self.width = width
        self.height = height
        self.background = background
        self.title = title
        self.draw_time_ms = draw_time_ms
        self.automated_draw_time_estimation = automated_draw_time_estimation
        if self.automated_draw_time_estimation:
            raise NotImplementedError("Automated draw time estimation is not implemented yet.")
        self.camera = Camera()
        self.verbose = verbose
        self.time_s = 0.0
        self.running = True
        self.recording = False
        self.recording_failed = False
        self.recorded_frames: List[np.ndarray] = []
        self.temp_dir = self._resolve_temp_dir(temp_dir)

        self.root = tk.Tk()
        self.root.title(title)
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, highlightthickness=0)
        self.canvas.pack()

        self._bind_keys()
        self.draw()
        self._tick()

    def draw(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, self.width, self.height, fill=self._color_to_hex(self.background), outline="")
        self._draw_origin_marker()
        self._draw_animation()
        self._draw_hud()
        self._capture_frame_if_recording()

    def _bind_keys(self) -> None:
        self.root.bind("<Escape>", lambda _e: self.root.destroy())
        self.root.bind("q", lambda _e: self.root.destroy())
        self.root.bind("<Left>", lambda _e: self._pan(20, 0))
        self.root.bind("<Right>", lambda _e: self._pan(-20, 0))
        self.root.bind("<Up>", lambda _e: self._pan(0, 20))
        self.root.bind("<Down>", lambda _e: self._pan(0, -20))
        self.root.bind("+", lambda _e: self._zoom(1.1))
        self.root.bind("=", lambda _e: self._zoom(1.1))
        self.root.bind("-", lambda _e: self._zoom(0.9))
        self.root.bind("c", lambda _e: self._reset_camera())
        self.root.bind("<space>", lambda _e: self._toggle_run())
        self.root.bind("r", lambda _e: self._toggle_recording())

    def _pan(self, dx: float, dy: float) -> None:
        self.camera.pan(dx / self.camera.zoom, dy / self.camera.zoom)
        self.draw()

    def _zoom(self, factor: float) -> None:
        self.camera.set_zoom(self.camera.zoom * factor)
        self.draw()

    def _reset_camera(self) -> None:
        self.camera.reset()
        self.draw()

    def _toggle_run(self) -> None:
        self.running = not self.running
        if self.running:
            self._tick()
        else:
            self.draw()

    def _tick(self) -> None:
        if not self.running:
            return
        self.time_s += self.draw_time_ms / 1000.0
        tic = time.perf_counter()
        self.draw()
        toc = time.perf_counter()
        elapsed_ms = (toc - tic) * 1000.0
        if self.verbose:
            logger.debug(f"Draw time: {elapsed_ms:.2f} ms")
        self.root.after(self.draw_time_ms, self._tick)

    def _world_to_screen(self, wx: float, wy: float) -> Tuple[float, float]:
        sx = (wx - self.camera.x) * self.camera.zoom + self.width / 2
        sy = (wy - self.camera.y) * self.camera.zoom + self.height / 2
        return sx, sy

    def _draw_origin_marker(self) -> None:
        center_x, center_y = self._world_to_screen(0, 0)
        size = 10 * self.camera.zoom
        self.canvas.create_line(center_x - size, center_y, center_x + size, center_y, fill="#888")
        self.canvas.create_line(center_x, center_y - size, center_x, center_y + size, fill="#888")

    def _draw_animation(self) -> None:
        """Simple evolving marker to demonstrate animation."""
        orbit_radius = 80
        wx = math.sin(self.time_s * 2.0) * orbit_radius
        wy = math.cos(self.time_s * 1.5) * orbit_radius
        sx, sy = self._world_to_screen(wx, wy)
        size = 10 * (1.0 + 0.2 * math.sin(self.time_s * 3.5)) * self.camera.zoom
        self.canvas.create_oval(sx - size, sy - size, sx + size, sy + size, fill="#ff6666", outline="")

    def _draw_hud(self) -> None:
        hud_text = [
            "Camera controls:",
            "Arrows: pan | +/-: zoom | c: reset | space: pause/resume | r: record | q/esc: quit",
            f"Pos: ({self.camera.x:.1f}, {self.camera.y:.1f})  Zoom: {self.camera.zoom:.2f}",
            f"Animation: {'running' if self.running else 'paused'}",
            f"Recording: {'ON' if self.recording else 'off'}",
        ]
        box_height = 8 + len(hud_text) * 12 + 8
        self.canvas.create_rectangle(8, 8, 380, box_height, fill="#000000", stipple="gray50", outline="")
        for idx, line in enumerate(hud_text):
            self.canvas.create_text(16, 16 + idx * 12, anchor="w", fill="#ffffff", font=("Helvetica", 9), text=line)
        if self.recording:
            # Small red indicator in the HUD while recording
            self.canvas.create_oval(350, 16, 364, 30, fill="#ff3333", outline="")

    def _toggle_recording(self) -> None:
        if self.recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        self.recording = True
        self.recording_failed = False
        self.recorded_frames = []
        logger.info("Recording started.")

    def _stop_recording(self) -> None:
        self.recording = False
        saved_path = self._write_recording()
        self.recorded_frames = []
        self.recording_failed = False
        if saved_path:
            logger.info("Recording saved to %s", saved_path)
        else:
            logger.warning("Recording stopped without output (no frames or write failed).")

    def _capture_frame_if_recording(self) -> None:
        if not self.recording or self.recording_failed:
            return
        try:
            self.root.update_idletasks()
            width = self.canvas.winfo_width()
            height = self.canvas.winfo_height()
            if width <= 1 or height <= 1 or not self.root.winfo_ismapped():
                return
            x0 = self.canvas.winfo_rootx()
            y0 = self.canvas.winfo_rooty()
            x1 = x0 + width
            y1 = y0 + height
            frame = self._grab_frame(x0, y0, x1, y1)
            if frame is not None:
                self.recorded_frames.append(np.array(frame))
            else:
                self.recording_failed = True
                logger.error("Failed to capture frame: no data returned from any backend.")
        except Exception as exc:
            self.recording_failed = True
            logger.error("Failed to capture frame for recording: %s", exc)

    def _grab_frame(self, x0: int, y0: int, x1: int, y1: int):
        """Capture the region using mss if available, else fallback to PIL ImageGrab."""
        # Preferred: mss (works better in WSL/X without temp files)
        if mss is not None:
            try:
                with mss.mss() as sct:
                    monitor = {"left": x0, "top": y0, "width": x1 - x0, "height": y1 - y0}
                    shot = sct.grab(monitor)
                    return ImageGrab.Image.frombytes("RGB", shot.size, shot.rgb)
            except Exception as exc:
                logger.warning("mss grab failed, falling back to PIL ImageGrab: %s", exc)

        # Fallback: PIL ImageGrab; ensure temp files go to our controlled dir
        with self._tempdir_override():
            grabbed = ImageGrab.grab(bbox=(x0, y0, x1, y1))
            return grabbed.convert("RGB") if grabbed else None

    def _resolve_temp_dir(self, temp_dir: Optional[str]) -> Path:
        """Choose where temp files (used by PIL on X11) are written."""
        base = Path(temp_dir).expanduser() if temp_dir else Path.cwd() / ".recording_tmp"
        base.mkdir(parents=True, exist_ok=True)
        return base.resolve()

    @contextlib.contextmanager
    def _tempdir_override(self):
        """Temporarily force PIL's temp usage into our chosen directory."""
        previous_tempdir = tempfile.tempdir
        previous_env_tmpdir = os.environ.get("TMPDIR")
        tempfile.tempdir = str(self.temp_dir)
        os.environ["TMPDIR"] = str(self.temp_dir)
        try:
            yield
        finally:
            tempfile.tempdir = previous_tempdir
            if previous_env_tmpdir is None:
                os.environ.pop("TMPDIR", None)
            else:
                os.environ["TMPDIR"] = previous_env_tmpdir

    def _write_recording(self) -> Optional[str]:
        if not self.recorded_frames:
            return None
        fps = max(1, int(round(1000.0 / self.draw_time_ms)))
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        base_name = f"recording_{timestamp}"
        mp4_path = f"{base_name}.mp4"
        try:
            with imageio.get_writer(mp4_path, fps=fps) as writer:
                for frame in self.recorded_frames:
                    writer.append_data(frame)
            return mp4_path
        except Exception as exc:
            logger.warning("MP4 export failed, falling back to GIF: %s", exc)

        gif_path = f"{base_name}.gif"
        try:
            imageio.mimsave(gif_path, self.recorded_frames, duration=1.0 / fps)
            return gif_path
        except Exception as exc:
            logger.error("GIF export failed: %s", exc)
            return None

    @staticmethod
    def _color_to_hex(color: Color) -> str:
        r, g, b = color
        return f"#{r:02x}{g:02x}{b:02x}"

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    app = BaseCanvas()
    app.run()
