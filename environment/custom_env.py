"""ShambaGuard: a farm-drone RL environment (Gymnasium).

A drone tends an N x N smallholder maize farm for one growing season. The base
station sits at cell (0, 0); every other cell is a maize plot. See the
"ShambaGuard - RL Environment Spec" doc for the full design.

Every constant lives in ``EnvConfig`` so difficulty presets and generalization
variants are just config overrides (see ``PRESETS`` / ``make_config``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# --------------------------------------------------------------------------- #
# Actions, weather, layout
# --------------------------------------------------------------------------- #
(MOVE_N, MOVE_S, MOVE_E, MOVE_W, IRRIGATE, FERTILIZE,
 SCAN, SPRAY, HARVEST, RECHARGE, WAIT) = range(11)
ACTION_NAMES = ("MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W", "IRRIGATE", "FERTILIZE",
                "SCAN", "SPRAY", "HARVEST", "RECHARGE", "WAIT")
N_ACTIONS = len(ACTION_NAMES)

DRY, NORMAL, WET = 0, 1, 2
WEATHER_NAMES = ("dry", "normal", "wet")

# Map channels, in observation order.
CH_MOISTURE, CH_NUTRIENT, CH_GROWTH, CH_HEALTH, CH_KNOWN_PEST, CH_CROP = range(6)
N_CHANNELS = 6
N_GLOBAL = 10  # battery, water, fert, pest, day, step-in-day, weather x3, distance
# Navigation "sensor" features (cfg.nav_features): for each target kind, whether one
# exists, the direction (row, col) and distance to the nearest one, and how many there
# are; then the battery margin over the trip home and a must-return flag.
NAV_TARGETS = ("dry", "infested", "low_nutrient", "harvestable", "mature")
N_NAV = 5 * len(NAV_TARGETS) + 2   # (exists, d_row, d_col, distance, count) per kind + 2

_MOVES = {MOVE_N: (-1, 0), MOVE_S: (1, 0), MOVE_E: (0, 1), MOVE_W: (0, -1)}
_NEIGHBOUR_SHIFTS = ((-1, 0), (1, 0), (0, 1), (0, -1))


@dataclass(frozen=True)
class EnvConfig:
    # World and time
    grid_size: int = 6
    season_days: int = 45
    steps_per_day: int = 10

    # Resources
    battery_capacity: int = 100
    water_capacity: int = 20
    fertilizer_budget: int = 20    # spec v1: 10
    pesticide_budget: int = 15     # spec v1: 10
    battery_cost: int = 1          # most actions
    battery_cost_heavy: int = 2    # SCAN and SPRAY

    # Weather (3-state Markov chain: dry, normal, wet)
    weather_stay_prob: float = 0.7
    rain_prob: tuple = (0.05, 0.25, 0.6)
    evaporation: tuple = (0.06, 0.04, 0.02)
    rain_amount: float = 0.12      # spec v1: 0.30 (moisture drifted up -> waterlogging)

    # Action effects
    irrigate_amount: float = 0.40
    fertilize_amount: float = 0.30
    scan_radius: int = 1           # 1 -> 3x3 area

    # Crop
    nutrient_uptake: float = 0.010  # n -= uptake * (0.5 + g) per day; spec v1: 0.015
    growth_rate: float = 0.04       # ~25 days to mature with perfect care; spec v1: 0.032
    moisture_band: tuple = (0.3, 0.7)
    nutrient_band: tuple = (0.4, 0.8)
    immature_harvest_factor: float = 0.3
    # HARVEST is only possible from this growth on (fix B). PPO first learned to strip the
    # farm at growth ~0.25 for 0.3*g*h each; when such harvests paid 0 it still used them
    # to remove plots before they died (dodging r_crop_death). Spec v1: 0.0.
    harvest_min_growth: float = 0.8

    # Pests
    outbreak_prob: float = 0.004   # spec v1: 0.01
    outbreak_level: float = 0.1
    pest_growth: float = 0.15
    spread_threshold: float = 0.4
    spread_factor: float = 0.25
    start_infested: int = 2
    start_pest_range: tuple = (0.1, 0.3)
    visible_threshold: float = 0.3  # spec v1: 0.5
    scan_memory_days: int = 3      # visible on the scan day and the next 2 days

    # Health damage per day
    drought_threshold: float = 0.2
    drought_damage: float = 0.08
    waterlog_threshold: float = 0.9
    waterlog_damage: float = 0.04
    pest_damage: float = 0.10      # x pest level
    burn_threshold: float = 0.9
    burn_damage: float = 0.05
    overripe_damage: float = 0.05

    # Start state
    start_moisture: tuple = (0.3, 0.6)
    start_nutrient: tuple = (0.4, 0.6)

    # Rewards
    r_harvest: float = 10.0        # x yield
    r_crop_death: float = -3.0
    r_spray_infested: float = 0.5
    spray_infested_threshold: float = 0.2
    r_spray_clean: float = -0.5
    spray_clean_threshold: float = 0.05
    r_irrigate_dry: float = 0.1
    irrigate_dry_threshold: float = 0.3
    r_irrigate_wet: float = -0.2
    irrigate_wet_threshold: float = 0.7
    cost_water: float = 0.02
    cost_fertilizer: float = 0.10
    cost_pesticide: float = 0.20
    r_invalid: float = -0.1
    r_step: float = -0.01
    r_stranded: float = -20.0

    # Observation: append navigation features (NAV_TARGETS, battery margin). Computed only
    # from what the agent can already observe. Spec v1: False (234-d observation).
    nav_features: bool = True
    return_margin: int = 3         # battery steps kept in reserve for the must-return flag
    # Fix C, first version: reward per unit of growth gained per plot. Too weak to guide
    # exploration (+0.01 for a well-timed irrigation); replaced by potential-based shaping.
    r_growth: float = 0.0
    # Fix C: potential-based shaping (Ng et al. 1999). F = shaping_gamma * phi(s') - phi(s),
    # phi = shaping_k * sum over growing plots of growth * health, phi = 0 at episode end.
    # With shaping_gamma equal to the agent's gamma the optimal policy is unchanged.
    # Trainers set shaping_gamma to their gamma. Spec v1: shaping_k = 0.
    # Growth alone only moves phi at day ticks, mixed with every plot's random changes, so an
    # irrigation was invisible. shaping_care adds the plot's current growing conditions:
    # phi = k * sum h * (g + care * band_m * band_n * (1 - pest)), so care pays on the step.
    shaping_k: float = 3.0
    shaping_care: float = 0.5
    shaping_gamma: float = 0.995

    def to_dict(self) -> dict:
        return asdict(self)


# Standard (the EnvConfig defaults) was rebalanced after scripts/dynamics_check.py
# showed that under the original spec numbers a greedy heuristic harvested 0 plots
# and lost ~24/35. "spec_v1" keeps the original numbers for the report.
PRESETS: dict[str, dict] = {
    "easy": dict(grid_size=4, season_days=30, outbreak_prob=0.002,
                 spread_factor=0.15, start_infested=1),
    "standard": {},
    # Drought: rarer rain AND faster drying (evaporation x1.5). Rarer rain alone made the
    # season easier (rain can only overshoot the moisture band; the agent irrigates freely),
    # greedy+fert +177 -> +195. With x1.5 evaporation it drops to +69 (13 plots lost);
    # x2 collapses even greedy (-13). (30 unseen seeds, 25 Nov 2026.)
    "drought": dict(rain_prob=(0.02, 0.10, 0.3), evaporation=(0.09, 0.06, 0.03)),
    "high_pest": dict(outbreak_prob=0.012, spread_factor=0.35, start_infested=4),  # outbreak x3
    "spec_v1": dict(fertilizer_budget=10, pesticide_budget=10, rain_amount=0.30,
                    nutrient_uptake=0.015, growth_rate=0.032, outbreak_prob=0.01,
                    visible_threshold=0.5, harvest_min_growth=0.0, shaping_k=0.0,
                    nav_features=False),
}


def make_config(preset: str = "standard", **overrides) -> EnvConfig:
    """Build a config from a named preset plus any field overrides."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {list(PRESETS)}")
    return replace(EnvConfig(), **{**PRESETS[preset], **overrides})


def band_factor(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """1 inside [lo, hi], falling linearly to 0 at x=0 and x=1."""
    below = np.clip(x / lo, 0.0, 1.0)
    above = np.clip((1.0 - x) / (1.0 - hi), 0.0, 1.0)
    return np.where(x < lo, below, np.where(x > hi, above, 1.0))


def _shift(a: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """out[r, c] = a[r - dr, c - dc], zero-filled at the edges."""
    n_r, n_c = a.shape
    out = np.zeros_like(a)
    out[max(dr, 0):n_r + min(dr, 0), max(dc, 0):n_c + min(dc, 0)] = \
        a[max(-dr, 0):n_r + min(-dr, 0), max(-dc, 0):n_c + min(-dc, 0)]
    return out


def obs_dim(grid_size: int, nav_features: bool = True) -> int:
    return N_CHANNELS * grid_size * grid_size + 2 + N_CHANNELS + N_GLOBAL + (N_NAV if nav_features else 0)


def decode_obs(obs: np.ndarray, grid_size: int) -> dict:
    """Split a flat observation back into named parts (for heuristics/tests)."""
    n, cells = grid_size, grid_size * grid_size
    maps = obs[:N_CHANNELS * cells].reshape(N_CHANNELS, n, n)
    i = N_CHANNELS * cells
    pos = np.rint(obs[i:i + 2] * (n - 1)).astype(int)
    i += 2
    current = obs[i:i + N_CHANNELS]
    i += N_CHANNELS
    g = obs[i:i + N_GLOBAL]
    i += N_GLOBAL
    nav = None
    if len(obs) >= i + N_NAV:  # nav_features on
        v = obs[i:i + N_NAV]
        nav = {name: dict(exists=bool(v[5 * k] > 0.5),
                          d_row=float((v[5 * k + 1] - 0.5) * 2 * (n - 1)),
                          d_col=float((v[5 * k + 2] - 0.5) * 2 * (n - 1)),
                          distance=float(v[5 * k + 3]), count=float(v[5 * k + 4]))
               for k, name in enumerate(NAV_TARGETS)}
        nav["battery_margin"], nav["must_return"] = float(v[-2]), bool(v[-1] > 0.5)
    return dict(
        nav=nav,
        moisture=maps[CH_MOISTURE], nutrient=maps[CH_NUTRIENT],
        growth=maps[CH_GROWTH], health=maps[CH_HEALTH],
        known_pest=maps[CH_KNOWN_PEST], crop=maps[CH_CROP] > 0.5,
        pos=(int(pos[0]), int(pos[1])), current=current,
        battery=g[0], water=g[1], fertilizer=g[2], pesticide=g[3],
        day=g[4], step_in_day=g[5], weather=int(np.argmax(g[6:9])),
        distance=g[9],
    )


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
class ShambaGuardEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 10}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None,
                 render_kwargs: dict | None = None):
        self.cfg = config if config is not None else EnvConfig()
        self.render_kwargs = render_kwargs or {}
        n = self.cfg.grid_size
        if n < 2:
            raise ValueError("grid_size must be at least 2")
        self.n = n
        self.observation_space = spaces.Box(0.0, 1.0, (obs_dim(n, self.cfg.nav_features),),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode
        self._renderer = None
        self._plot_mask = np.ones((n, n), dtype=bool)
        self._plot_mask[0, 0] = False  # base station
        self._max_dist = 2 * (n - 1)

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        cfg, n, rng, mask = self.cfg, self.n, self.np_random, self._plot_mask

        self.moisture = np.where(mask, rng.uniform(*cfg.start_moisture, (n, n)), 0.0)
        self.nutrient = np.where(mask, rng.uniform(*cfg.start_nutrient, (n, n)), 0.0)
        self.growth = np.zeros((n, n))
        self.health = mask.astype(np.float64)
        self.crop = mask.copy()
        # Plot status for rendering/analysis: 0 growing, 1 harvested, 2 dead, -1 base.
        self.status = np.where(mask, 0, -1).astype(np.int8)
        self.pest = np.zeros((n, n))
        plots = np.flatnonzero(mask)
        k = min(cfg.start_infested, plots.size)
        if k:
            idx = rng.choice(plots, size=k, replace=False)
            self.pest.flat[idx] = rng.uniform(*cfg.start_pest_range, size=k)
        self.last_scan = np.full((n, n), -10_000, dtype=np.int64)

        self.pos = (0, 0)
        self.battery = cfg.battery_capacity
        self.water = cfg.water_capacity
        self.fertilizer = cfg.fertilizer_budget
        self.pesticide = cfg.pesticide_budget
        self.weather = int(rng.integers(3))
        self.rained_today = False
        self.day = 0
        self.step_in_day = 0
        self.t = 0

        self.stats = dict(total_yield=0.0, plots_harvested=0, plots_lost=0,
                          water_used=0, fertilizer_used=0, pesticide_used=0,
                          recharges=0, invalid_actions=0, stranded=False,
                          episode_return=0.0, harvest_return=0.0, growth_return=0.0,
                          shaping_return=0.0, season_end=False)
        self.last_action: int | None = None
        self.last_reward = 0.0

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._info()

    # ------------------------------------------------------------------- step
    def step(self, action):
        a = int(action)
        if not 0 <= a < N_ACTIONS:
            raise ValueError(f"invalid action {action!r}")
        cfg = self.cfg

        reward = cfg.r_step
        phi_before = self.potential()
        act_reward, harvest_reward = self._apply_action(a)
        reward += act_reward

        terminated = False
        if self.battery <= 0 and not self._at_base():
            reward += cfg.r_stranded
            self.stats["stranded"] = True
            terminated = True

        self.t += 1
        self.step_in_day += 1
        if not terminated and self.step_in_day >= cfg.steps_per_day:
            self.step_in_day = 0
            reward += self._day_tick()
            self.day += 1

        if not terminated and not self.crop.any():
            terminated = True  # every plot harvested or dead
        # The season end is a true terminal state, not a time-limit truncation: the day is
        # part of the observation and nothing follows it, so learners must not bootstrap a
        # value past it. (Spec v1 reported it as truncated; bootstrapping there credited
        # harvests that can never happen and, with shaping, biased against live crops.)
        if not terminated and self.day >= cfg.season_days:
            terminated = True
            self.stats["season_end"] = True
        truncated = False

        if cfg.shaping_k:
            phi_after = 0.0 if (terminated or truncated) else self.potential()
            shaping = cfg.shaping_gamma * phi_after - phi_before
            reward += shaping
            self.stats["shaping_return"] += shaping

        self.last_action = a
        self.last_reward = reward
        self.stats["episode_return"] += reward
        self.stats["harvest_return"] += harvest_reward

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, self._info()

    # --------------------------------------------------------------- actions
    def _at_base(self) -> bool:
        return self.pos == (0, 0)

    def _apply_action(self, a: int) -> tuple[float, float]:
        """Apply one agent action. Returns (shaping + harvest reward, harvest reward)."""
        cfg = self.cfg
        r, c = self.pos
        on_crop = bool(self.crop[r, c])
        cost = cfg.battery_cost
        reward = 0.0
        harvest_reward = 0.0
        valid = True

        if a in _MOVES:
            dr, dc = _MOVES[a]
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.n and 0 <= nc < self.n:
                self.pos = (nr, nc)
            else:
                valid = False

        elif a == IRRIGATE:
            if on_crop and self.water > 0:
                m = self.moisture[r, c]
                if m < cfg.irrigate_dry_threshold:
                    reward += cfg.r_irrigate_dry
                elif m > cfg.irrigate_wet_threshold:
                    reward += cfg.r_irrigate_wet
                self.moisture[r, c] = min(1.0, m + cfg.irrigate_amount)
                self.water -= 1
                self.stats["water_used"] += 1
                reward -= cfg.cost_water
            else:
                valid = False

        elif a == FERTILIZE:
            if on_crop and self.fertilizer > 0:
                self.nutrient[r, c] = min(1.0, self.nutrient[r, c] + cfg.fertilize_amount)
                self.fertilizer -= 1
                self.stats["fertilizer_used"] += 1
                reward -= cfg.cost_fertilizer
            else:
                valid = False

        elif a == SCAN:
            cost = cfg.battery_cost_heavy
            k = cfg.scan_radius
            self.last_scan[max(r - k, 0):r + k + 1, max(c - k, 0):c + k + 1] = self.day

        elif a == SPRAY:
            cost = cfg.battery_cost_heavy
            if on_crop and self.pesticide > 0:
                p = self.pest[r, c]
                if p > cfg.spray_infested_threshold:
                    reward += cfg.r_spray_infested
                elif p < cfg.spray_clean_threshold:
                    reward += cfg.r_spray_clean
                self.pest[r, c] = 0.0
                self.pesticide -= 1
                self.stats["pesticide_used"] += 1
                reward -= cfg.cost_pesticide
            else:
                valid = False

        elif a == HARVEST:
            if on_crop and self.growth[r, c] >= cfg.harvest_min_growth:
                g, h = self.growth[r, c], self.health[r, c]
                y = h if g >= 1.0 else cfg.immature_harvest_factor * g * h
                harvest_reward = cfg.r_harvest * y
                reward += harvest_reward
                self.stats["total_yield"] += y
                self.stats["plots_harvested"] += 1
                self._clear_plot(r, c, status=1)
            else:
                valid = False

        elif a == RECHARGE:
            cost = 0
            if self._at_base():
                if self.battery < cfg.battery_capacity or self.water < cfg.water_capacity:
                    self.stats["recharges"] += 1
                self.battery = cfg.battery_capacity
                self.water = cfg.water_capacity
            else:
                valid = False

        else:  # WAIT
            cost = 0

        if valid:
            self.battery = max(0, self.battery - cost)
        else:
            reward += cfg.r_invalid
            self.stats["invalid_actions"] += 1
        return reward, harvest_reward

    def _clear_plot(self, r: int, c: int, status: int) -> None:
        self.status[r, c] = status
        self.crop[r, c] = False
        self.growth[r, c] = 0.0
        self.health[r, c] = 0.0
        self.pest[r, c] = 0.0

    # -------------------------------------------------------------- dynamics
    def _day_tick(self) -> float:
        """Advance the farm one day. Returns the growth reward plus crop-death penalties."""
        cfg, rng, n, mask = self.cfg, self.np_random, self.n, self._plot_mask
        crop = self.crop

        # 1. Weather
        if rng.random() >= cfg.weather_stay_prob:
            others = [w for w in range(3) if w != self.weather]
            self.weather = int(others[rng.integers(2)])
        self.rained_today = bool(rng.random() < cfg.rain_prob[self.weather])

        # 2. Moisture
        m = self.moisture - cfg.evaporation[self.weather]
        if self.rained_today:
            m = m + cfg.rain_amount
        self.moisture = np.where(mask, np.clip(m, 0.0, 1.0), 0.0)

        # 3. Nutrients (crop uptake)
        uptake = cfg.nutrient_uptake * (0.5 + self.growth)
        self.nutrient = np.where(crop, np.clip(self.nutrient - uptake, 0.0, 1.0), self.nutrient)

        # 4. Growth
        ripe_before = crop & (self.growth >= 1.0)
        growing = crop & (self.growth < 1.0)
        dg = (cfg.growth_rate * self.health
              * band_factor(self.moisture, *cfg.moisture_band)
              * band_factor(self.nutrient, *cfg.nutrient_band))
        new_growth = np.where(growing, np.minimum(1.0, self.growth + dg), self.growth)
        growth_reward = cfg.r_growth * float((new_growth - self.growth).sum())
        self.stats["growth_return"] += growth_reward
        self.growth = new_growth

        # 5. Pests: outbreak, growth, spread
        p = self.pest
        outbreak = crop & (p <= 0) & (rng.random((n, n)) < cfg.outbreak_prob)
        p = np.where(outbreak, cfg.outbreak_level, p)
        p = np.where(crop & (p > 0), np.minimum(1.0, p * (1.0 + cfg.pest_growth)), p)
        source = np.where(crop & (p > cfg.spread_threshold), cfg.spread_factor * p, 0.0)
        infected = np.zeros((n, n), dtype=bool)
        for dr, dc in _NEIGHBOUR_SHIFTS:
            infected |= rng.random((n, n)) < _shift(source, dr, dc)
        p = np.where(crop & (p <= 0) & infected, cfg.outbreak_level, p)
        self.pest = np.where(crop, p, 0.0)

        # 6. Health damage
        dmg = (cfg.drought_damage * (self.moisture < cfg.drought_threshold)
               + cfg.waterlog_damage * (self.moisture > cfg.waterlog_threshold)
               + cfg.pest_damage * self.pest
               + cfg.burn_damage * (self.nutrient > cfg.burn_threshold)
               + cfg.overripe_damage * ripe_before)
        self.health = np.where(crop, np.clip(self.health - dmg, 0.0, 1.0), 0.0)

        died = crop & (self.health <= 0.0)
        n_died = int(died.sum())
        if n_died:
            for r, c in zip(*np.nonzero(died)):
                self._clear_plot(r, c, status=2)
            self.stats["plots_lost"] += n_died
        return growth_reward + cfg.r_crop_death * n_died

    # ----------------------------------------------------------- observation
    def known_pest(self) -> np.ndarray:
        """Pest level the agent can see: visible symptoms or a recent scan."""
        cfg = self.cfg
        recent = (self.day - self.last_scan) < cfg.scan_memory_days
        visible = (self.pest >= cfg.visible_threshold) | recent
        return np.where(visible & self.crop, self.pest, 0.0)

    def _get_obs(self) -> np.ndarray:
        cfg, n = self.cfg, self.n
        maps = np.stack([self.moisture, self.nutrient, self.growth, self.health,
                         self.known_pest(), self.crop.astype(np.float64)])
        maps[:, ~self._plot_mask] = 0.0
        r, c = self.pos
        weather = np.zeros(3)
        weather[self.weather] = 1.0
        glob = np.array([
            self.battery / cfg.battery_capacity,
            self.water / max(cfg.water_capacity, 1),
            self.fertilizer / max(cfg.fertilizer_budget, 1),
            self.pesticide / max(cfg.pesticide_budget, 1),
            min(self.day / cfg.season_days, 1.0),
            self.step_in_day / cfg.steps_per_day,
            *weather,
            (r + c) / self._max_dist,
        ])
        parts = [maps.ravel(), [r / (n - 1), c / (n - 1)], maps[:, r, c], glob]
        if cfg.nav_features:
            parts.append(self._nav_features(maps))
        obs = np.concatenate(parts)
        return np.clip(obs, 0.0, 1.0).astype(np.float32)

    def _nav_features(self, maps: np.ndarray) -> np.ndarray:
        """Nearest-target features from observable maps; all values in [0, 1].

        Per target kind: exists, row offset and column offset to the nearest one (mapped
        from [-1, 1] to [0, 1], 0.5 = same row/col), Manhattan distance, count / plots."""
        cfg, n = self.cfg, self.n
        r, c = self.pos
        crop = maps[CH_CROP] > 0.5
        kinds = dict(
            dry=crop & (maps[CH_MOISTURE] < cfg.irrigate_dry_threshold),
            infested=crop & (maps[CH_KNOWN_PEST] > 0),
            low_nutrient=crop & (maps[CH_NUTRIENT] < cfg.nutrient_band[0]),
            harvestable=crop & (maps[CH_GROWTH] >= cfg.harvest_min_growth),
            mature=crop & (maps[CH_GROWTH] >= 1.0),
        )
        rows, cols = np.indices((n, n))
        dist = np.abs(rows - r) + np.abs(cols - c)
        n_plots = max(int(self._plot_mask.sum()), 1)
        out = []
        for name in NAV_TARGETS:
            m = kinds[name]
            if m.any():
                d = np.where(m, dist, 10 ** 6)
                i = np.unravel_index(int(np.argmin(d)), d.shape)
                out += [1.0, 0.5 + 0.5 * (i[0] - r) / (n - 1), 0.5 + 0.5 * (i[1] - c) / (n - 1),
                        d[i] / self._max_dist, m.sum() / n_plots]
            else:
                out += [0.0, 0.5, 0.5, 1.0, 0.0]
        home = r + c
        out += [max(self.battery - home, 0) / cfg.battery_capacity,
                float(self.battery <= home + cfg.return_margin)]
        return np.asarray(out, dtype=np.float64)

    def potential(self) -> float:
        """Shaping potential over growing plots:
        shaping_k * sum health * (growth + shaping_care * band_m * band_n * (1 - pest))."""
        cfg = self.cfg
        if not cfg.shaping_k:
            return 0.0
        value = self.growth
        if cfg.shaping_care:
            conditions = (band_factor(self.moisture, *cfg.moisture_band)
                          * band_factor(self.nutrient, *cfg.nutrient_band) * (1.0 - self.pest))
            value = value + cfg.shaping_care * conditions
        return cfg.shaping_k * float((self.health * value)[self.crop].sum())

    def action_mask(self) -> np.ndarray:
        """Boolean mask of valid actions (fix A). Masked actions are exactly the ones
        _apply_action would reject with r_invalid; harmful-but-valid actions stay open."""
        r, c = self.pos
        on_crop = bool(self.crop[r, c])
        m = np.ones(N_ACTIONS, dtype=bool)
        m[MOVE_N], m[MOVE_S] = r > 0, r < self.n - 1
        m[MOVE_E], m[MOVE_W] = c < self.n - 1, c > 0
        m[IRRIGATE] = on_crop and self.water > 0
        m[FERTILIZE] = on_crop and self.fertilizer > 0
        m[SPRAY] = on_crop and self.pesticide > 0
        m[HARVEST] = on_crop and self.growth[r, c] >= self.cfg.harvest_min_growth
        m[RECHARGE] = self._at_base()
        return m

    def _info(self) -> dict:
        # task_return: the return without potential-based shaping, for fair comparisons.
        return dict(self.stats, action_mask=self.action_mask(),
                    task_return=self.stats["episode_return"] - self.stats["shaping_return"],
                    day=self.day, step_in_day=self.step_in_day,
                    weather=WEATHER_NAMES[self.weather], rained=self.rained_today,
                    battery=self.battery, water=self.water,
                    fertilizer=self.fertilizer, pesticide=self.pesticide,
                    pos=self.pos)

    # ------------------------------------------------------------- rendering
    def render(self):
        if self.render_mode == "ansi":
            return self._render_text()
        if self.render_mode in ("human", "rgb_array"):
            if self._renderer is None:
                try:
                    from environment.rendering import FarmRenderer
                except ImportError as e:  # renderer not written yet
                    raise NotImplementedError(
                        "environment/rendering.py (FarmRenderer) is not available yet; "
                        "use render_mode='ansi' for now") from e
                self._renderer = FarmRenderer(self, mode=self.render_mode, **self.render_kwargs)
            return self._renderer.draw(self)
        return None

    def verbose_line(self) -> str:
        a = ACTION_NAMES[self.last_action] if self.last_action is not None else "-"
        return (f"Day {self.day:2d} | step {self.step_in_day} | {a:<9} -> {self.pos} | "
                f"r={self.last_reward:+.2f} | batt {self.battery:3d} | water {self.water:2d} | "
                f"return {self.stats['episode_return']:.1f}")

    def _render_text(self) -> str:
        """Each cell: growth digit (0-9, M = mature), ! = known pest, ~ = dry, x = empty."""
        known = self.known_pest()
        rows = []
        for r in range(self.n):
            cells = []
            for c in range(self.n):
                if (r, c) == (0, 0):
                    s = "BS"
                elif not self.crop[r, c]:
                    s = "x "
                else:
                    g = self.growth[r, c]
                    s = ("M" if g >= 1 else str(min(int(g * 10), 9)))
                    s += "!" if known[r, c] > 0 else ("~" if self.moisture[r, c] < 0.3 else " ")
                cells.append(f"[{s}]" if (r, c) == self.pos else f" {s} ")
            rows.append("".join(cells))
        header = (f"Day {self.day}/{self.cfg.season_days}  {WEATHER_NAMES[self.weather]}  "
                  f"batt {self.battery}  water {self.water}  fert {self.fertilizer}  "
                  f"pest {self.pesticide}  return {self.stats['episode_return']:.1f}")
        return header + "\n" + "\n".join(rows)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
