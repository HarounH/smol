from smol.art.base import Camera, BaseCanvas
import math
import dataclasses


@dataclasses.dataclass
class CrossHatchArgs:
    enabled: bool = False

    def artist(self) -> "BaseCrossHatchArtist" | None:
        if self.enabled:
            return DefaultCrossHatchArtist(self)
        else:
            return None

class BaseCrossHatchArtist:
    def __init__(self, cross_hatch_args: CrossHatchArgs) -> None:
        self.cross_hatch_args = cross_hatch_args

class DefaultCrossHatchArtist(BaseCrossHatchArtist):
    def draw():
        pass


class CrossHatchCanvas(BaseCanvas):
    def __init__(self, *args, cross_hatch_args: CrossHatchArgs | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cross_hatch_args = cross_hatch_args or CrossHatchArgs()
        self.cross_hatch_artist = cross_hatch_args.artist()

    def draw(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, self.width, self.height, fill=self._color_to_hex(self.background), outline="")
        self._draw_origin_marker()
        self._draw_animation()
        self._draw_hud()
        self._capture_frame_if_recording()

    def _draw_animation(self) -> None:
        orbit_radius = 80
        wx = math.sin(self.time_s * 2.0) * orbit_radius
        wy = math.cos(self.time_s * 1.5) * orbit_radius
        sx, sy = self._world_to_screen(wx, wy)
        size = 10 * (1.0 + 0.2 * math.sin(self.time_s * 3.5)) * self.camera.zoom
        self.canvas.create_oval(sx - size, sy - size, sx + size, sy + size, dash=(50, 20), outline="black")
        if self.cross_hatch_artist is not None:
            self.cross_hatch_artist.draw(self)

if __name__ == "__main__":
    app = CrossHatchCanvas()
    app.run()
