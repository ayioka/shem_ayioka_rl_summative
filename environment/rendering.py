"""Top-down 2D Pygame renderer for ShambaGuard.

render_mode="human"     -> live window (D toggles hidden pests, Esc/close quits)
render_mode="rgb_array" -> returns an (H, W, 3) uint8 frame (works headless)

The spec's 3D PyOpenGL scene can replace this later; it uses the same draw(env) API.
"""
from __future__ import annotations

import math
import os

import numpy as np

from environment.custom_env import (ACTION_NAMES, FERTILIZE, HARVEST, IRRIGATE, RECHARGE,
                                    SCAN, SPRAY, WEATHER_NAMES)

BG = (24, 28, 34)
PANEL = (34, 39, 48)
TEXT = (232, 234, 238)
MUTED = (150, 158, 170)
SOIL_DRY = (205, 172, 125)
SOIL_WET = (88, 60, 38)
GRID_LINE = (60, 45, 30)
HEALTHY = (52, 168, 72)
STRESSED = (222, 200, 64)
DYING = (128, 84, 40)
MATURE_RING = (245, 196, 40)
PEST = (226, 46, 46)
HIDDEN_PEST = (210, 90, 230)
WATER = (64, 150, 235)
NUTRIENT = (110, 200, 90)
BATTERY_OK = (80, 200, 110)
BATTERY_LOW = (230, 70, 60)


def _lerp(a, b, t):
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def _health_colour(h):
    return _lerp(DYING, STRESSED, h / 0.5) if h < 0.5 else _lerp(STRESSED, HEALTHY, (h - 0.5) / 0.5)


class FarmRenderer:
    def __init__(self, env, mode: str = "human", show_true_state: bool = False,
                 cell: int = 96, fps: int | None = None):
        if mode == "rgb_array":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless (Colab, CI)
        import pygame  # imported here so the env works without pygame installed

        self.pg = pygame
        pygame.init()
        self.mode = mode
        self.show_true_state = show_true_state
        self.window_closed = False
        self.n = env.n
        self.cell = cell
        self.margin = max(cell // 6, 10)
        self.hud_w = max(280, int(cell * 3.2))
        grid_px = self.n * cell
        self.width = 2 * self.margin + grid_px + self.hud_w
        self.height = max(2 * self.margin + grid_px, 600 if cell >= 80 else 440)
        self.fps = fps or env.metadata.get("render_fps", 10)
        scale = cell / 96
        self.font = pygame.font.SysFont("segoeui,dejavusans,arial", max(int(17 * scale), 11))
        self.font_big = pygame.font.SysFont("segoeui,dejavusans,arial", max(int(24 * scale), 14),
                                            bold=True)
        self.font_small = pygame.font.SysFont("segoeui,dejavusans,arial", max(int(13 * scale), 9))
        if mode == "human":
            pygame.display.set_caption("ShambaGuard")
            self.surface = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
        else:
            self.surface = pygame.Surface((self.width, self.height))
            self.clock = None
        self.overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        rng = np.random.default_rng(42)
        self._speck_offsets = rng.uniform(0.18, 0.82, size=(self.n * self.n, 10, 2))

    # ----------------------------------------------------------------- public
    def draw(self, env):
        pg = self.pg
        if self.mode == "human":
            for event in pg.event.get():
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                    self.window_closed = True
                elif event.type == pg.KEYDOWN and event.key == pg.K_d:
                    self.show_true_state = not self.show_true_state
            if self.window_closed:
                return None

        self.surface.fill(BG)
        self.overlay.fill((0, 0, 0, 0))
        known = env.known_pest()
        for r in range(self.n):
            for c in range(self.n):
                self._draw_cell(env, r, c, known[r, c])
        self._draw_effect(env)
        self.surface.blit(self.overlay, (0, 0))
        self._draw_drone(env)
        self._draw_hud(env)

        if self.mode == "human":
            pg.display.flip()
            self.clock.tick(self.fps)
            return None
        return np.transpose(pg.surfarray.array3d(self.surface), (1, 0, 2)).copy()

    def close(self):
        if self.mode == "human":
            self.pg.display.quit()
        self.pg.quit()

    # ------------------------------------------------------------------ cells
    def _cell_rect(self, r, c):
        return self.pg.Rect(self.margin + c * self.cell, self.margin + r * self.cell,
                            self.cell, self.cell)

    def _draw_cell(self, env, r, c, known_p):
        pg, cell = self.pg, self.cell
        rect = self._cell_rect(r, c)
        if (r, c) == (0, 0):
            self._draw_base(rect)
            return
        m = env.moisture[r, c]
        pg.draw.rect(self.surface, _lerp(SOIL_DRY, SOIL_WET, m), rect)
        pg.draw.rect(self.surface, GRID_LINE, rect, 1)
        cx, cy = rect.centerx, rect.top + int(cell * 0.44)
        status = env.status[r, c]

        if env.crop[r, c]:
            g, h = env.growth[r, c], env.health[r, c]
            col = _health_colour(h)
            radius = int(cell * (0.08 + 0.17 * g))
            # a small cluster of leaves, growing with g
            for k, (dx, dy) in enumerate(((0, 0), (-0.55, 0.3), (0.55, 0.3), (0, -0.6))):
                rr = radius if k == 0 else int(radius * 0.7)
                pg.draw.circle(self.surface, col, (cx + int(dx * radius), cy + int(dy * radius)), rr)
            pg.draw.circle(self.surface, _lerp(col, (0, 0, 0), 0.35), (cx, cy), radius, 1)
            if g >= 1.0:
                pg.draw.circle(self.surface, MATURE_RING, (cx, cy), int(cell * 0.40), max(cell // 24, 3))
            # known pests: red specks; hidden pests (debug): purple rings
            true_p = env.pest[r, c]
            offs = self._speck_offsets[r * self.n + c]
            if known_p > 0:
                for ox, oy in offs[:max(1, math.ceil(known_p * 10))]:
                    pg.draw.circle(self.surface, PEST,
                                   (rect.left + int(ox * cell), rect.top + int(oy * cell * 0.8)),
                                   max(2, cell // 28))
            elif self.show_true_state and true_p > 0:
                for ox, oy in offs[:max(1, math.ceil(true_p * 10))]:
                    pg.draw.circle(self.surface, HIDDEN_PEST,
                                   (rect.left + int(ox * cell), rect.top + int(oy * cell * 0.8)),
                                   max(2, cell // 28), 1)
        elif status == 1:  # harvested: stubble rows
            for k in range(4):
                y = rect.top + int(cell * (0.25 + 0.15 * k))
                pg.draw.line(self.surface, (225, 205, 150), (rect.left + cell // 5, y),
                             (rect.right - cell // 5, y), 2)
        elif status == 2:  # dead crop
            d = cell // 4
            pg.draw.line(self.surface, (50, 30, 20), (cx - d, cy - d), (cx + d, cy + d), 4)
            pg.draw.line(self.surface, (50, 30, 20), (cx - d, cy + d), (cx + d, cy - d), 4)

        # moisture (blue) and nutrient (green) gauges, with ideal-band ticks
        bw, bh = cell - 12, max(cell // 16, 3)
        x0 = rect.left + 6
        for i, (val, colr, band) in enumerate(((m, WATER, env.cfg.moisture_band),
                                               (env.nutrient[r, c], NUTRIENT, env.cfg.nutrient_band))):
            y0 = rect.bottom - 6 - (2 - i) * (bh + 2)
            pg.draw.rect(self.surface, (30, 30, 30), (x0, y0, bw, bh))
            pg.draw.rect(self.surface, colr, (x0, y0, int(bw * np.clip(val, 0, 1)), bh))
            for b in band:
                bx = x0 + int(bw * b)
                pg.draw.line(self.surface, (245, 245, 245), (bx, y0 - 1), (bx, y0 + bh), 1)

    def _draw_base(self, rect):
        pg, cell = self.pg, self.cell
        pg.draw.rect(self.surface, (70, 78, 90), rect)
        pg.draw.rect(self.surface, (40, 44, 52), rect, 2)
        pad = rect.inflate(-cell * 0.3, -cell * 0.3)
        pg.draw.ellipse(self.surface, (110, 118, 130), pad)
        pg.draw.ellipse(self.surface, (230, 200, 60), pad, 2)
        h = self.font_small.render("H", True, (230, 200, 60))
        self.surface.blit(h, h.get_rect(center=pad.center))
        # solar panel and water tank
        panel = pg.Rect(rect.left + 4, rect.top + 4, cell // 3, cell // 5)
        pg.draw.rect(self.surface, (40, 80, 160), panel)
        for k in range(1, 3):
            x = panel.left + k * panel.width // 3
            pg.draw.line(self.surface, (150, 180, 230), (x, panel.top), (x, panel.bottom), 1)
        pg.draw.circle(self.surface, WATER, (rect.right - cell // 7, rect.top + cell // 7), cell // 10)

    # ------------------------------------------------------------ drone + fx
    def _drone_centre(self, env):
        r, c = env.pos
        rect = self._cell_rect(r, c)
        return rect.centerx, rect.top + int(self.cell * 0.42)

    def _draw_drone(self, env):
        pg, cell = self.pg, self.cell
        x, y = self._drone_centre(env)
        arm = int(cell * 0.2)
        rotor = max(int(cell * 0.09), 4)
        shadow = pg.Surface((cell, cell), pg.SRCALPHA)
        pg.draw.ellipse(shadow, (0, 0, 0, 70), (cell * 0.3, cell * 0.62, cell * 0.4, cell * 0.12))
        self.surface.blit(shadow, (x - cell // 2, y - int(cell * 0.42)))
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            end = (x + dx * arm, y + dy * arm)
            pg.draw.line(self.surface, (40, 40, 46), (x, y), end, 3)
            pg.draw.circle(self.surface, (200, 205, 215), end, rotor)
            pg.draw.circle(self.surface, (60, 60, 70), end, rotor, 1)
        body = pg.Rect(0, 0, int(cell * 0.18), int(cell * 0.18))
        body.center = (x, y)
        pg.draw.rect(self.surface, (30, 32, 40), body, border_radius=4)
        frac = env.battery / env.cfg.battery_capacity
        pg.draw.circle(self.surface, _lerp(BATTERY_LOW, BATTERY_OK, frac), (x, y), max(cell // 30, 2))

    def _draw_effect(self, env):
        a = env.last_action
        if a is None:
            return
        pg, cell, ov = self.pg, self.cell, self.overlay
        x, y = self._drone_centre(env)
        r, c = env.pos
        rng = np.random.default_rng(env.t)
        if a == SCAN:
            k = env.cfg.scan_radius
            r0, c0 = max(r - k, 0), max(c - k, 0)
            r1, c1 = min(r + k, self.n - 1), min(c + k, self.n - 1)
            area = self._cell_rect(r0, c0).union(self._cell_rect(r1, c1))
            pg.draw.rect(ov, (90, 220, 255, 60), area)
            pg.draw.rect(ov, (90, 220, 255, 200), area, 3)
        elif a in (IRRIGATE, FERTILIZE, SPRAY):
            colour = {IRRIGATE: (70, 160, 255, 150), FERTILIZE: (120, 230, 90, 170),
                      SPRAY: (255, 150, 40, 160)}[a]
            for _ in range(26):
                ox, oy = rng.normal(0, cell * 0.14), rng.uniform(0.05, 0.45) * cell
                pg.draw.circle(ov, colour, (int(x + ox), int(y + oy)), max(cell // 30, 2))
        elif a == HARVEST:
            for ang in range(0, 360, 45):
                dx, dy = math.cos(math.radians(ang)), math.sin(math.radians(ang))
                pg.draw.line(ov, (255, 220, 60, 220), (x + dx * cell * 0.22, y + dy * cell * 0.22),
                             (x + dx * cell * 0.4, y + dy * cell * 0.4), 3)
        elif a == RECHARGE:
            pts = [(x + 4, y - 22), (x - 8, y + 2), (x, y + 2), (x - 4, y + 22), (x + 10, y - 4), (x + 2, y - 4)]
            pg.draw.polygon(ov, (255, 225, 60, 230), [(int(px), int(py)) for px, py in pts])

    # -------------------------------------------------------------------- HUD
    def _draw_hud(self, env):
        pg = self.pg
        x0 = 2 * self.margin + self.n * self.cell
        panel = pg.Rect(x0 - self.margin // 2, 0, self.width - x0 + self.margin // 2, self.height)
        pg.draw.rect(self.surface, PANEL, panel)
        x = x0 + 6
        y = self.margin
        lh = self.font.get_linesize()

        def text(s, colour=TEXT, font=None, dy=None):
            nonlocal y
            f = font or self.font
            self.surface.blit(f.render(s, True, colour), (x, y))
            y += dy if dy is not None else f.get_linesize()

        cfg = env.cfg
        text("ShambaGuard", font=self.font_big)
        text(f"Day {env.day}/{cfg.season_days}   step {env.step_in_day}/{cfg.steps_per_day}")
        self._weather_icon(x + 14, y + 14, env.weather)
        self.surface.blit(self.font.render(
            WEATHER_NAMES[env.weather] + ("  - rained" if env.rained_today else ""), True, TEXT),
            (x + 36, y + 4))
        y += 34

        bar_w = panel.right - x - 14
        for label, val, cap, colour in (
                ("Battery", env.battery, cfg.battery_capacity,
                 _lerp(BATTERY_LOW, BATTERY_OK, env.battery / cfg.battery_capacity)),
                ("Water", env.water, cfg.water_capacity, WATER),
                ("Fertilizer", env.fertilizer, cfg.fertilizer_budget, NUTRIENT),
                ("Pesticide", env.pesticide, cfg.pesticide_budget, (255, 150, 40))):
            text(f"{label}: {val}/{cap}", MUTED, self.font_small)
            pg.draw.rect(self.surface, (20, 22, 28), (x, y, bar_w, 9))
            pg.draw.rect(self.surface, colour, (x, y, int(bar_w * val / max(cap, 1)), 9))
            y += 15

        y += 6
        last = ACTION_NAMES[env.last_action] if env.last_action is not None else "-"
        text(f"Action: {last}")
        text(f"Step reward: {env.last_reward:+.2f}")
        s = env.stats
        text(f"Return: {s['episode_return']:.1f}  (harvest {s['harvest_return']:.1f})")
        text(f"Harvested {s['plots_harvested']}   Lost {s['plots_lost']}   "
             f"Growing {int(env.crop.sum())}")
        y += 8
        text("Legend", MUTED, self.font_small)
        for colour, label in ((PEST, "known pests (red specks)"),
                              (MATURE_RING, "mature: ready to harvest"),
                              (WATER, "moisture gauge, ticks = ideal band"),
                              (NUTRIENT, "nutrient gauge")):
            sh = self.font_small.get_linesize()
            pg.draw.circle(self.surface, colour, (x + 6, y + sh // 2), 5)
            self.surface.blit(self.font_small.render(label, True, MUTED), (x + 18, y))
            y += sh
        hidden = "on" if self.show_true_state else "off"
        text(f"Hidden pests (purple): {hidden}" + ("   [D]" if self.mode == "human" else ""),
             HIDDEN_PEST if self.show_true_state else MUTED, self.font_small)

    def _weather_icon(self, cx, cy, weather):
        pg = self.pg
        if weather in (0, 1):
            sun = (cx - 4, cy - 2) if weather == 1 else (cx, cy)
            pg.draw.circle(self.surface, (250, 205, 50), sun, 8)
            if weather == 0:
                for ang in range(0, 360, 45):
                    dx, dy = math.cos(math.radians(ang)), math.sin(math.radians(ang))
                    pg.draw.line(self.surface, (250, 205, 50), (cx + dx * 11, cy + dy * 11),
                                 (cx + dx * 15, cy + dy * 15), 2)
        if weather in (1, 2):
            for ox, oy, rad in ((-5, 3, 7), (3, 0, 9), (10, 4, 6)):
                pg.draw.circle(self.surface, (200, 205, 215), (cx + ox, cy + oy), rad)
        if weather == 2:
            for ox in (-6, 1, 8):
                pg.draw.line(self.surface, WATER, (cx + ox, cy + 12), (cx + ox - 3, cy + 19), 2)
