"""Interactive blank 2D canvas with simple camera controls."""

import math
import tkinter as tk
from dataclasses import dataclass
from typing import Tuple
import logging
import time

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
        self.time_s = 0.0
        self.running = True

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
        self.root.bind("r", lambda _e: self._reset_camera())
        self.root.bind("<space>", lambda _e: self._toggle_run())

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
        if verbose:
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
            "Arrows: pan | +/-: zoom | r: reset | space: pause/resume | q/esc: quit",
            f"Pos: ({self.camera.x:.1f}, {self.camera.y:.1f})  Zoom: {self.camera.zoom:.2f}",
            f"Animation: {'running' if self.running else 'paused'}",
        ]
        box_height = 8 + len(hud_text) * 12 + 8
        self.canvas.create_rectangle(8, 8, 360, box_height, fill="#000000", stipple="gray50", outline="")
        for idx, line in enumerate(hud_text):
            self.canvas.create_text(16, 16 + idx * 12, anchor="w", fill="#ffffff", font=("Helvetica", 9), text=line)

    @staticmethod
    def _color_to_hex(color: Color) -> str:
        r, g, b = color
        return f"#{r:02x}{g:02x}{b:02x}"

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    app = BaseCanvas()
    app.run()
