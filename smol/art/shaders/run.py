#!/usr/bin/env python3
"""
run.py

A minimal Shadertoy-like runner for WSL2/Linux using real GLSL via ModernGL.

Features:
- Loads a fragment shader file containing either:
  (A) Shadertoy-style:  void mainImage(out vec4 fragColor, in vec2 fragCoord)
      (optionally also: void main() { ... }  -> ignored; we provide our own main)
  OR
  (B) Full fragment shader with: out vec4 fragColor; void main() { ... }
      (we will compile as-is)
- Uniforms:
    iResolution : vec3 (pixels, pixels, 1.0)
    iTime       : float (seconds since start)
    iTimeDelta  : float (seconds since last frame)
    iFrame      : int
    iMouse      : vec4 (xy current pos in pixels, zw click pos; negative if not pressed)
- Optional texture channels iChannel0..iChannel3 (2D textures)

Hot-reload:
- Automatically recompiles when the shader file changes on disk.
- Press R to force reload, Esc to quit.

Install:
  pip install moderngl moderngl-window pillow numpy

Run:
  python run.py path/to/shader.glsl --size 1280 720 --fps 60
  python run.py shader.glsl --channel0 some.png

Notes for WSL2:
- For GPU acceleration via WSLg, confirm you are not using llvmpipe:
    glxinfo -B | grep -i "OpenGL renderer"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image

import moderngl
import moderngl_window as mglw


VERT_SRC = r"""
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = (in_pos + 1.0) * 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

# A wrapper that enables Shadertoy-style mainImage shaders.
# We provide fragColor output and call mainImage with fragCoord in pixels.
FRAG_WRAPPER_PREFIX = r"""
#version 330

uniform vec3  iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform int   iFrame;
uniform vec4  iMouse;

uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform sampler2D iChannel2;
uniform sampler2D iChannel3;

in vec2 v_uv;
out vec4 fragColor;

"""

FRAG_WRAPPER_SUFFIX = r"""
void main() {
    vec2 fragCoord = v_uv * iResolution.xy;
    vec4 col = vec4(0.0);
    mainImage(col, fragCoord);
    fragColor = col;
}
"""


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _looks_like_shadertoy(src: str) -> bool:
    # If it contains mainImage signature, we treat it as Shadertoy style.
    return re.search(r"\bvoid\s+mainImage\s*\(\s*out\s+vec4\s+\w+\s*,\s*in\s+vec2\s+\w+\s*\)", src) is not None


def _has_fragment_main(src: str) -> bool:
    return re.search(r"\bvoid\s+main\s*\(\s*\)", src) is not None and re.search(r"\bout\s+vec4\s+\w+\s*;", src) is not None


def build_fragment_shader(user_src: str) -> str:
    """
    If user provides Shadertoy mainImage, wrap it with uniforms + main().
    If user provides a full fragment shader with out vec4 + main(), compile as-is.
    Otherwise, attempt to wrap (best-effort).
    """
    if _looks_like_shadertoy(user_src):
        # Strip any leading #version to avoid duplicates.
        user_src_wo_version = re.sub(r"^\s*#version\s+\d+\s*\n", "", user_src, flags=re.MULTILINE)
        return FRAG_WRAPPER_PREFIX + "\n" + user_src_wo_version + "\n" + FRAG_WRAPPER_SUFFIX

    if _has_fragment_main(user_src):
        return user_src

    # Fallback: assume they wrote shader code that sets fragColor in mainImage-like style.
    user_src_wo_version = re.sub(r"^\s*#version\s+\d+\s*\n", "", user_src, flags=re.MULTILINE)
    return FRAG_WRAPPER_PREFIX + "\n" + user_src_wo_version + "\n" + FRAG_WRAPPER_SUFFIX


def load_texture(ctx: moderngl.Context, path: Optional[str]) -> moderngl.Texture:
    """
    Loads a 2D texture from image file. If path is None, returns a 1x1 black texture.
    """
    if not path:
        tex = ctx.texture((1, 1), 4, data=b"\x00\x00\x00\xff")
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = True
        tex.repeat_y = True
        return tex

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Texture not found: {p}")

    img = Image.open(p).convert("RGBA")
    img = img.transpose(Image.FLIP_TOP_BOTTOM)  # OpenGL texture coordinate convention
    data = img.tobytes()
    tex = ctx.texture(img.size, 4, data=data)
    tex.build_mipmaps()
    tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    tex.repeat_x = True
    tex.repeat_y = True
    return tex


@dataclass
class MouseState:
    x: float = 0.0
    y: float = 0.0
    down: bool = False
    click_x: float = -1.0
    click_y: float = -1.0

    def iMouse_vec4(self, h: int) -> Tuple[float, float, float, float]:
        # Shadertoy uses origin bottom-left for iMouse in pixels.
        # moderngl_window gives mouse with origin top-left, so flip y.
        y_flipped = float(h) - float(self.y)
        if self.down:
            return (float(self.x), y_flipped, float(self.click_x), float(h) - float(self.click_y))
        # When not pressed, zw are negative.
        return (float(self.x), y_flipped, -1.0, -1.0)


class ShaderToyRunner(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "ShaderToy Runner (Python)"
    window_size = (1280, 720)
    aspect_ratio = None
    resizable = True
    vsync = True

    # Set by main()
    shader_path: Path = Path("shader.glsl")
    target_fps: Optional[int] = None
    channels: List[Optional[str]] = [None, None, None, None]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._start_time = time.perf_counter()
        self._last_frame_time = self._start_time
        self._frame = 0

        self._mouse = MouseState()

        # Fullscreen quad (two triangles)
        vbo = self.ctx.buffer(np.array([
            -1, -1,  1, -1, -1,  1,
            -1,  1,  1, -1,  1,  1,
        ], dtype="f4").tobytes())

        self._vbo = vbo
        self._vao = None
        self._prog = None

        # Textures
        self._tex = [None, None, None, None]
        for i in range(4):
            self._tex[i] = load_texture(self.ctx, self.channels[i])

        self._shader_mtime = 0.0
        self._compile_shader(first_time=True)

    def _compile_shader(self, first_time: bool = False):
        try:
            user_src = _read_text(self.shader_path)
            frag_src = build_fragment_shader(user_src)
            prog = self.ctx.program(vertex_shader=VERT_SRC, fragment_shader=frag_src)
            vao = self.ctx.simple_vertex_array(prog, self._vbo, "in_pos")

            # Bind textures to units 0..3 and set sampler uniforms if present.
            for i in range(4):
                if self._tex[i] is not None:
                    self._tex[i].use(location=i)
                name = f"iChannel{i}"
                if name in prog:
                    prog[name].value = i

            self._prog = prog
            self._vao = vao

            self._shader_mtime = self.shader_path.stat().st_mtime
            if not first_time:
                print(f"[reloaded] {self.shader_path}")
        except Exception as e:
            print(f"[shader compile error] {e}", file=sys.stderr)

    def _maybe_reload_shader(self):
        try:
            mtime = self.shader_path.stat().st_mtime
        except FileNotFoundError:
            return
        if mtime > self._shader_mtime:
            self._compile_shader()

    def key_event(self, key, action, modifiers):
        keys = self.wnd.keys
        if action == keys.ACTION_PRESS:
            if key == keys.ESCAPE:
                self.wnd.close()
            elif key == keys.R:
                self._compile_shader()
            elif key == keys.V:
                self.vsync = not self.vsync
                self.wnd.vsync = self.vsync
                print(f"[vsync] {self.vsync}")

    def mouse_position_event(self, x, y, dx, dy):
        self._mouse.x = x
        self._mouse.y = y

    def mouse_press_event(self, x, y, button):
        self._mouse.down = True
        self._mouse.click_x = x
        self._mouse.click_y = y

    def mouse_release_event(self, x, y, button):
        self._mouse.down = False

    def on_render(self, time: float, frame_time: float):
        # moderngl-window compatibility (some versions call on_render)
        return self.render(time, frame_time)

    def render(self, time_now: float, frame_time: float):
        # Optional FPS cap
        if self.target_fps is not None and self.target_fps > 0:
            min_dt = 1.0 / float(self.target_fps)
            now = time.perf_counter()
            dt = now - self._last_frame_time
            if dt < min_dt:
                time.sleep(min_dt - dt)
            now2 = time.perf_counter()
            frame_dt = now2 - self._last_frame_time
        else:
            now2 = time.perf_counter()
            frame_dt = now2 - self._last_frame_time

        self._last_frame_time = now2

        self._maybe_reload_shader()
        if self._prog is None or self._vao is None:
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            return

        w, h = self.wnd.size
        iTime = now2 - self._start_time

        if "iResolution" in self._prog:
            self._prog["iResolution"].value = (float(w), float(h), 1.0)
        if "iTime" in self._prog:
            self._prog["iTime"].value = float(iTime)
        if "iTimeDelta" in self._prog:
            self._prog["iTimeDelta"].value = float(frame_dt)
        if "iFrame" in self._prog:
            self._prog["iFrame"].value = int(self._frame)
        if "iMouse" in self._prog:
            self._prog["iMouse"].value = self._mouse.iMouse_vec4(h)

        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self._vao.render()
        self._frame += 1


def main():
    ap = argparse.ArgumentParser(description="Run Shadertoy-like GLSL fragment shaders in Python (ModernGL).")
    ap.add_argument("shader", nargs="?", help="Path to shader")
    ap.add_argument("--shader", dest="shader_flag", help="Path to shader")
    ap.add_argument("--size", type=int, nargs=2, default=[1280, 720], metavar=("W", "H"))
    ap.add_argument("--title", type=str, default="ShaderToy Runner (Python)")
    ap.add_argument("--fps", type=int, default=0, help="Cap FPS (0 = uncapped)")
    ap.add_argument("--no-vsync", action="store_true", help="Disable vsync")
    ap.add_argument("--channel0", type=str, default=None, help="Texture path for iChannel0")
    ap.add_argument("--channel1", type=str, default=None, help="Texture path for iChannel1")
    ap.add_argument("--channel2", type=str, default=None, help="Texture path for iChannel2")
    ap.add_argument("--channel3", type=str, default=None, help="Texture path for iChannel3")
    args = ap.parse_args()

    ShaderToyRunner.shader_path = Path(args.shader or args.shader_flag).expanduser().resolve()
    ShaderToyRunner.window_size = (int(args.size[0]), int(args.size[1]))
    ShaderToyRunner.title = args.title
    ShaderToyRunner.vsync = not args.no_vsync
    ShaderToyRunner.target_fps = args.fps if args.fps and args.fps > 0 else None
    ShaderToyRunner.channels = [args.channel0, args.channel1, args.channel2, args.channel3]

    if not ShaderToyRunner.shader_path.exists():
        print(f"Shader not found: {ShaderToyRunner.shader_path}", file=sys.stderr)
        sys.exit(1)

    sys.argv = [sys.argv[0]]
    mglw.run_window_config(ShaderToyRunner)


if __name__ == "__main__":
    main()
