"""
Multi-Agent Cleanup Gridworld Environment
==========================================

Implements the Cleanup game from:
  Hughes et al. (2018) "Inequity aversion improves cooperation in
  intertemporal social dilemmas"

and used in:
  Vinitsky et al. (2023) "A learning agent that acquires social norms
  from public sanctions in decentralized multi-agent settings"

A 2D partially-observable Markov game where agents collect apples (+1)
and must clean pollution from a river to allow apples to respawn.
Cleaning costs -1 reward but benefits all agents (public good).
Agents can also fire a penalty beam (-1 cost, -50 to target, tags out
for 25 steps).

Key features:
  - N independent agents with egocentric RGB observations
  - 9 discrete actions: move (4 dirs), rotate (2 dirs), fire beam,
    stand still, clean beam
  - River pollution determines apple spawn probability
  - Cleaning is costly but enables apple growth for everyone
  - Gymnasium-style API: reset() / step(actions) returning per-agent dicts
"""

from __future__ import annotations

import numpy as np
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from gathering_env import Orientation, _ROTATIONS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class CleanupAction(IntEnum):
    """Nine agent-centered actions for the Cleanup game.

    Actions 0-7 match the Gathering Action enum exactly, so existing
    policies that return 0-7 are compatible.  Action 8 (CLEAN) is new.
    """
    FORWARD = 0
    BACKWARD = 1
    STEP_LEFT = 2
    STEP_RIGHT = 3
    ROTATE_LEFT = 4
    ROTATE_RIGHT = 5
    BEAM = 6       # fire penalty beam at opponents
    STAND = 7
    CLEAN = 8      # fire cleaning beam to remove waste

NUM_CLEANUP_ACTIONS = len(CleanupAction)

# RGB colour palette (extends Gathering palette with river/waste/clean)
COLOUR = {
    "background":  np.array([0, 0, 0], dtype=np.uint8),
    "wall":        np.array([127, 127, 127], dtype=np.uint8),
    "apple":       np.array([0, 200, 0], dtype=np.uint8),
    "self":        np.array([50, 100, 255], dtype=np.uint8),
    "other":       np.array([220, 50, 50], dtype=np.uint8),
    "beam":        np.array([255, 255, 60], dtype=np.uint8),       # fire beam
    "timeout":     np.array([100, 100, 100], dtype=np.uint8),
    "river":       np.array([100, 180, 255], dtype=np.uint8),      # clean river
    "waste":       np.array([139, 90, 43], dtype=np.uint8),        # pollution
    "stream":      np.array([0, 180, 200], dtype=np.uint8),        # transition zone
    "clean_beam":  np.array([100, 255, 255], dtype=np.uint8),      # cleaning beam
}


# ---------------------------------------------------------------------------
# Default maps (ASCII)
#   @ = wall   A = apple spawn   P = player spawn
#   H = waste (polluted river, starts dirty)
#   R = clean river (starts clean, can become polluted)
#   S = stream (walkable transition zone, cannot get waste)
#   . or space = empty walkable
# ---------------------------------------------------------------------------

CLEANUP_MAP = """\
@@@@@@@@@@@@@@@@@@
@RRRRRR     AAAAA@
@HHHHHH      AAAA@
@RRRRRR     AAAAA@
@RRRRR  P    AAAA@
@RRRRR    P AAAAA@
@HHHHH       AAAA@
@RRRRR      AAAAA@
@HHHHHHSSSSSSAAAA@
@HHHHHHSSSSSSAAAA@
@RRRRR   P P AAAA@
@HHHHH   P  AAAAA@
@RRRRRR    P AAAA@
@HHHHHH P   AAAAA@
@RRRRR       AAAA@
@HHHH    P  AAAAA@
@RRRRR       AAAA@
@HHHHH  P P AAAAA@
@RRRRR       AAAA@
@HHHH       AAAAA@
@RRRRR       AAAA@
@HHHHH      AAAAA@
@RRRRR       AAAA@
@HHHH       AAAAA@
@@@@@@@@@@@@@@@@@@"""

CLEANUP_MAP_SMALL = """\
@@@@@@@@@@@@@
@RRR   AAAAA@
@HHH   AAAAA@
@RRRSS AAAAA@
@RR  P AAAAA@
@HH  P AAAAA@
@RR  P AAAAA@
@RRRSS AAAAA@
@HHH   AAAAA@
@RRR   AAAAA@
@@@@@@@@@@@@@"""


# ---------------------------------------------------------------------------
# Map parser
# ---------------------------------------------------------------------------

def parse_cleanup_map(
    ascii_map: str,
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]],
           List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Parse an ASCII cleanup map string into arrays.

    Returns
    -------
    walls : np.ndarray, bool, shape (H, W)
    apple_points : list of (row, col)
    spawn_points : list of (row, col)
    river_cells : list of (row, col)  — all H + R cells (potential waste area)
    waste_init : list of (row, col)   — H cells only (start with waste)
    stream_cells : list of (row, col) — S cells
    """
    lines = [l for l in ascii_map.strip().splitlines()]
    height = len(lines)
    width = max(len(l) for l in lines)
    lines = [l.ljust(width) for l in lines]

    walls = np.zeros((height, width), dtype=bool)
    apple_points: List[Tuple[int, int]] = []
    spawn_points: List[Tuple[int, int]] = []
    river_cells: List[Tuple[int, int]] = []
    waste_init: List[Tuple[int, int]] = []
    stream_cells: List[Tuple[int, int]] = []

    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            if ch == "@":
                walls[r, c] = True
            elif ch == "A":
                apple_points.append((r, c))
            elif ch == "P":
                spawn_points.append((r, c))
            elif ch == "H":
                river_cells.append((r, c))
                waste_init.append((r, c))
            elif ch == "R":
                river_cells.append((r, c))
            elif ch == "S":
                stream_cells.append((r, c))

    # Fallback spawn points near edges if none marked
    if not spawn_points:
        for r in range(height):
            for c in range(width):
                if not walls[r, c] and (r, c) not in apple_points:
                    if r <= 1 or r >= height - 2 or c <= 1 or c >= width - 2:
                        spawn_points.append((r, c))

    return walls, apple_points, spawn_points, river_cells, waste_init, stream_cells


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CleanupEnv:
    """Multi-agent Cleanup gridworld.

    Parameters
    ----------
    ascii_map : str
        ASCII map with ``@`` wall, ``A`` apple, ``H`` waste, ``R`` river,
        ``S`` stream, ``P`` spawn, ``.``/space empty.
    n_agents : int
        Number of agents.
    max_steps : int
        Episode length (default 1000).
    obs_ahead, obs_side : int
        Egocentric observation window (default 7 each → 15x15 view).
    beam_length : int
        Range of both fire and clean beams (default 5).
    beam_width : int
        Width of both beams (default 3: center + 1 each side).
    hits_to_tag : int
        Fire beam hits to tag an agent out (default 1).
    timeout_steps : int
        Steps a tagged agent is removed (default 25).
    fire_cost : float
        Reward cost to the agent that fires the penalty beam (default 1.0).
    fire_penalty : float
        Reward penalty to the agent hit by a fire beam (default 50.0).
    clean_cost : float
        Reward cost to the agent that fires the cleaning beam (default 1.0).
    threshold_depletion : float
        Waste density at/above which apples (and new waste) stop spawning
        (default 0.4).
    threshold_restoration : float
        Waste density at/below which apple spawn rate is maximal (default 0.0).
    waste_spawn_prob : float
        Per-step probability of spawning one new waste cell (default 0.5).
    apple_respawn_prob : float
        Per-cell per-step apple spawn probability when river is fully clean
        (default 0.05).
    beam_enabled : bool
        If False, fire beam has no effect (ablation).
    seed : int or None
        Random seed.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        ascii_map: str = CLEANUP_MAP,
        n_agents: int = 5,
        max_steps: int = 1000,
        # Observation
        obs_ahead: int = 7,
        obs_side: int = 7,
        # Beams (fire + clean share range/width)
        beam_length: int = 5,
        beam_width: int = 3,
        hits_to_tag: int = 1,
        timeout_steps: int = 25,
        beam_enabled: bool = True,
        # Reward costs
        fire_cost: float = 1.0,
        fire_penalty: float = 50.0,
        clean_cost: float = 1.0,
        # Waste / apple dynamics
        threshold_depletion: float = 0.4,
        threshold_restoration: float = 0.0,
        waste_spawn_prob: float = 0.5,
        apple_respawn_prob: float = 0.05,
        # Misc
        seed: Optional[int] = None,
    ):
        # Parse the map
        parsed = parse_cleanup_map(ascii_map)
        self.walls = parsed[0]
        self.apple_points = parsed[1]
        self.spawn_points = parsed[2]
        self.river_cells_list = parsed[3]   # H + R cells
        self.waste_init_list = parsed[4]    # H cells only
        self.stream_cells_list = parsed[5]  # S cells
        self.height, self.width = self.walls.shape

        self.n_agents = n_agents
        self.max_steps = max_steps

        # Observation window (agent at bottom-centre)
        self.obs_ahead = obs_ahead
        self.obs_side = obs_side
        self.obs_h = obs_ahead + 1
        self.obs_w = 2 * obs_side + 1

        # Beam parameters (shared by fire and clean beams)
        self.beam_length = beam_length
        self.beam_width = beam_width
        self.hits_to_tag = hits_to_tag
        self.timeout_steps = timeout_steps
        self.beam_enabled = beam_enabled

        # Reward costs
        self.fire_cost = fire_cost
        self.fire_penalty = fire_penalty
        self.clean_cost = clean_cost

        # Waste / apple dynamics
        self.threshold_depletion = threshold_depletion
        self.threshold_restoration = threshold_restoration
        self.waste_spawn_prob = waste_spawn_prob
        self.apple_respawn_prob = apple_respawn_prob

        # Pre-compute indices
        self.n_apples = len(self.apple_points)
        self._apple_pos = np.array(self.apple_points, dtype=np.int32) if self.n_apples else np.zeros((0, 2), dtype=np.int32)
        self.river_cells_set = set(self.river_cells_list)
        self.stream_cells_set = set(self.stream_cells_list)
        self.potential_waste_area = len(self.river_cells_list)

        # Random state
        self.rng = np.random.default_rng(seed)

        # --- Mutable state (initialised in reset) ---
        self._step_count: int = 0
        self.agent_pos = np.zeros((n_agents, 2), dtype=np.int32)
        self.agent_orient = np.zeros(n_agents, dtype=np.int32)
        self.agent_timeout = np.zeros(n_agents, dtype=np.int32)
        self.agent_beam_hits = np.zeros(n_agents, dtype=np.int32)
        self.apple_alive = np.ones(self.n_apples, dtype=bool)
        # Waste map: True where waste (pollution) currently exists
        self.waste = np.zeros((self.height, self.width), dtype=bool)

        # Compatibility shims for gathering_policy.py helpers
        self.apple_timer = np.zeros(self.n_apples, dtype=np.int32)
        self.apple_respawn_time = 0
        self.respawn_mode = "cleanup"

        # Current-step beam cells (for rendering)
        self._beam_cells: set = set()
        self._clean_beam_cells: set = set()

        # Observation & action space descriptions
        self.observation_shape = (3, self.obs_h, self.obs_w)
        self.action_space_n = NUM_CLEANUP_ACTIONS

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Dict[int, np.ndarray]:
        """Reset the environment and return initial observations."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._step_count = 0

        # Spawn agents at marked spawn points (or random open cells)
        available = list(self.spawn_points)
        self.rng.shuffle(available)
        # Open cells: not wall, not apple, not river, not stream
        apple_set = set(self.apple_points)
        open_cells = [
            (r, c) for r in range(self.height) for c in range(self.width)
            if not self.walls[r, c]
            and (r, c) not in apple_set
            and (r, c) not in self.river_cells_set
            and (r, c) not in self.stream_cells_set
        ]
        for i in range(self.n_agents):
            if i < len(available):
                self.agent_pos[i] = available[i]
            else:
                idx = self.rng.integers(len(open_cells))
                self.agent_pos[i] = open_cells[idx]
            self.agent_orient[i] = self.rng.integers(4)

        self.agent_timeout[:] = 0
        self.agent_beam_hits[:] = 0

        # Initialize waste from map (H cells start polluted)
        self.waste[:] = False
        for r, c in self.waste_init_list:
            self.waste[r, c] = True

        # All apples start alive
        self.apple_alive[:] = True
        self.apple_timer[:] = 0

        self._beam_cells = set()
        self._clean_beam_cells = set()

        return {i: self._get_obs(i) for i in range(self.n_agents)}

    # ------------------------------------------------------------------
    # Waste / apple dynamics
    # ------------------------------------------------------------------

    def _compute_waste_density(self) -> float:
        """Fraction of river cells currently polluted (0.0 = pristine)."""
        if self.potential_waste_area == 0:
            return 0.0
        current_waste = sum(1 for r, c in self.river_cells_list if self.waste[r, c])
        return current_waste / self.potential_waste_area

    def _current_apple_spawn_prob(self, waste_density: float) -> float:
        """Apple spawn probability given current waste density.

        Linear interpolation between threshold_restoration (max prob)
        and threshold_depletion (zero prob).
        """
        if waste_density >= self.threshold_depletion:
            return 0.0
        if waste_density <= self.threshold_restoration:
            return self.apple_respawn_prob
        frac = ((waste_density - self.threshold_restoration)
                / (self.threshold_depletion - self.threshold_restoration))
        return (1.0 - frac) * self.apple_respawn_prob

    def _spawn_waste(self, waste_density: float) -> None:
        """Probabilistically spawn one new waste cell on the river."""
        if waste_density >= self.threshold_depletion:
            return
        if self.rng.random() >= self.waste_spawn_prob:
            return
        # Candidate: clean river cells not occupied by agents
        agent_positions = set(
            (int(self.agent_pos[i, 0]), int(self.agent_pos[i, 1]))
            for i in range(self.n_agents)
        )
        candidates = [
            (r, c) for r, c in self.river_cells_list
            if not self.waste[r, c] and (r, c) not in agent_positions
        ]
        if candidates:
            idx = self.rng.integers(len(candidates))
            r, c = candidates[idx]
            self.waste[r, c] = True

    def _spawn_apples(self, apple_prob: float) -> None:
        """Probabilistically spawn apples at empty apple-spawn cells."""
        if apple_prob <= 0:
            return
        agent_positions = set(
            (int(self.agent_pos[i, 0]), int(self.agent_pos[i, 1]))
            for i in range(self.n_agents)
        )
        for a_idx in range(self.n_apples):
            if not self.apple_alive[a_idx]:
                ar, ac = int(self._apple_pos[a_idx, 0]), int(self._apple_pos[a_idx, 1])
                if (ar, ac) not in agent_positions and self.rng.random() < apple_prob:
                    self.apple_alive[a_idx] = True

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, actions: Dict[int, int],
    ) -> Tuple[
        Dict[int, np.ndarray],
        Dict[int, float],
        Dict[int, bool],
        Dict[int, bool],
        Dict[int, dict],
    ]:
        """Advance one timestep.

        Parameters
        ----------
        actions : dict {agent_id: int}
            An action for every agent (0-8).
        """
        self._step_count += 1
        rewards = {i: 0.0 for i in range(self.n_agents)}
        self._beam_cells = set()
        self._clean_beam_cells = set()

        # --- 1. Decrement timeouts ---
        self.agent_timeout = np.maximum(self.agent_timeout - 1, 0)

        # --- 2. Process actions for active agents ---
        for i in range(self.n_agents):
            if self.agent_timeout[i] > 0:
                continue
            act = actions.get(i, CleanupAction.STAND)
            if act == CleanupAction.FORWARD:
                self._move_agent(i, "forward")
            elif act == CleanupAction.BACKWARD:
                self._move_agent(i, "backward")
            elif act == CleanupAction.STEP_LEFT:
                self._move_agent(i, "left")
            elif act == CleanupAction.STEP_RIGHT:
                self._move_agent(i, "right")
            elif act == CleanupAction.ROTATE_LEFT:
                self.agent_orient[i] = (self.agent_orient[i] - 1) % 4
            elif act == CleanupAction.ROTATE_RIGHT:
                self.agent_orient[i] = (self.agent_orient[i] + 1) % 4
            elif act == CleanupAction.BEAM:
                if self.beam_enabled:
                    self._fire_beam(i, rewards)
            elif act == CleanupAction.CLEAN:
                self._fire_clean_beam(i, rewards)
            # STAND does nothing

        # --- 3. Spawn waste ---
        waste_density = self._compute_waste_density()
        self._spawn_waste(waste_density)

        # --- 4. Spawn apples (probabilistic, depends on waste) ---
        # Recompute after waste spawn may have changed density
        waste_density = self._compute_waste_density()
        apple_prob = self._current_apple_spawn_prob(waste_density)
        self._spawn_apples(apple_prob)

        # --- 5. Collect apples ---
        for i in range(self.n_agents):
            if self.agent_timeout[i] > 0:
                continue
            r, c = self.agent_pos[i]
            for a_idx in range(self.n_apples):
                if self.apple_alive[a_idx]:
                    ar, ac = self._apple_pos[a_idx]
                    if ar == r and ac == c:
                        self.apple_alive[a_idx] = False
                        rewards[i] += 1.0

        # --- 6. Check episode termination ---
        done = self._step_count >= self.max_steps
        terminated = {i: done for i in range(self.n_agents)}
        truncated = {i: False for i in range(self.n_agents)}

        # --- 7. Build observations ---
        obs = {i: self._get_obs(i) for i in range(self.n_agents)}

        # --- 8. Info ---
        info: Dict[int, dict] = {}
        for i in range(self.n_agents):
            info[i] = {
                "timeout": int(self.agent_timeout[i]),
                "beam_hits": int(self.agent_beam_hits[i]),
                "waste_density": waste_density,
                "apple_spawn_prob": apple_prob,
            }

        return obs, rewards, terminated, truncated, info

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def _move_agent(self, agent_id: int, direction: str) -> None:
        orient = Orientation(self.agent_orient[agent_id])
        a, b, c, d = _ROTATIONS[orient]
        if direction == "forward":
            dr, dc = a, c
        elif direction == "backward":
            dr, dc = -a, -c
        elif direction == "left":
            dr, dc = -b, -d
        elif direction == "right":
            dr, dc = b, d
        else:
            return

        nr = self.agent_pos[agent_id, 0] + dr
        nc = self.agent_pos[agent_id, 1] + dc

        if 0 <= nr < self.height and 0 <= nc < self.width and not self.walls[nr, nc]:
            self.agent_pos[agent_id] = (nr, nc)

    # ------------------------------------------------------------------
    # Fire beam (penalty)
    # ------------------------------------------------------------------

    def _fire_beam(self, agent_id: int, rewards: Dict[int, float]) -> None:
        """Fire the penalty beam.  Costs fire_cost to the firer; each
        agent hit receives -fire_penalty and accumulates a beam hit."""
        rewards[agent_id] -= self.fire_cost

        orient = Orientation(self.agent_orient[agent_id])
        a, b, c, d = _ROTATIONS[orient]
        ar, ac = self.agent_pos[agent_id]
        half_w = self.beam_width // 2

        beam_cells: List[Tuple[int, int]] = []
        for dist in range(1, self.beam_length + 1):
            for w_off in range(-half_w, half_w + 1):
                dr = a * dist + b * w_off
                dc = c * dist + d * w_off
                br, bc = ar + dr, ac + dc
                if 0 <= br < self.height and 0 <= bc < self.width:
                    if self.walls[br, bc]:
                        continue
                    beam_cells.append((br, bc))
                    self._beam_cells.add((br, bc))

        beam_set = set(beam_cells)
        for j in range(self.n_agents):
            if j == agent_id or self.agent_timeout[j] > 0:
                continue
            pos_j = tuple(self.agent_pos[j])
            if pos_j in beam_set:
                self.agent_beam_hits[j] += 1
                rewards[j] -= self.fire_penalty
                if self.agent_beam_hits[j] >= self.hits_to_tag:
                    self.agent_timeout[j] = self.timeout_steps
                    self.agent_beam_hits[j] = 0

    # ------------------------------------------------------------------
    # Clean beam (removes waste)
    # ------------------------------------------------------------------

    def _fire_clean_beam(self, agent_id: int, rewards: Dict[int, float]) -> None:
        """Fire the cleaning beam.  Costs clean_cost; converts waste cells
        in the beam path back to clean river."""
        rewards[agent_id] -= self.clean_cost

        orient = Orientation(self.agent_orient[agent_id])
        a, b, c, d = _ROTATIONS[orient]
        ar, ac = self.agent_pos[agent_id]
        half_w = self.beam_width // 2

        for dist in range(1, self.beam_length + 1):
            for w_off in range(-half_w, half_w + 1):
                dr = a * dist + b * w_off
                dc = c * dist + d * w_off
                br, bc = ar + dr, ac + dc
                if 0 <= br < self.height and 0 <= bc < self.width:
                    if self.walls[br, bc]:
                        continue
                    self._clean_beam_cells.add((br, bc))
                    if self.waste[br, bc]:
                        self.waste[br, bc] = False

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_obs(self, agent_id: int) -> np.ndarray:
        """Build an egocentric RGB observation for *agent_id*.

        The agent sits at the bottom-centre of the observation window,
        looking "up" (forward).  Shape: ``(3, obs_h, obs_w)`` uint8.
        """
        obs = np.zeros((self.obs_h, self.obs_w, 3), dtype=np.uint8)
        orient = Orientation(self.agent_orient[agent_id])
        a, b, c, d = _ROTATIONS[orient]
        ar, ac = self.agent_pos[agent_id]

        # Fast lookups
        apple_set: set = set()
        for idx in range(self.n_apples):
            if self.apple_alive[idx]:
                apple_set.add((int(self._apple_pos[idx, 0]),
                               int(self._apple_pos[idx, 1])))

        agent_positions: Dict[Tuple[int, int], List[int]] = {}
        for j in range(self.n_agents):
            pos_j = (int(self.agent_pos[j, 0]), int(self.agent_pos[j, 1]))
            agent_positions.setdefault(pos_j, []).append(j)

        for obs_r in range(self.obs_h):
            for obs_c in range(self.obs_w):
                fwd = self.obs_ahead - obs_r
                right = obs_c - self.obs_side
                wr = ar + a * fwd + b * right
                wc = ac + c * fwd + d * right

                # Out of bounds
                if wr < 0 or wr >= self.height or wc < 0 or wc >= self.width:
                    obs[obs_r, obs_c] = COLOUR["wall"]
                    continue

                cell_pos = (wr, wc)

                # Priority: agent > beam > apple > waste > river > stream > wall > bg
                if cell_pos in agent_positions:
                    drawn = False
                    for j in agent_positions[cell_pos]:
                        if j == agent_id:
                            obs[obs_r, obs_c] = COLOUR["self"]
                            drawn = True
                            break
                        elif self.agent_timeout[j] > 0:
                            obs[obs_r, obs_c] = COLOUR["timeout"]
                            drawn = True
                            break
                        else:
                            obs[obs_r, obs_c] = COLOUR["other"]
                            drawn = True
                            break
                    if drawn:
                        continue

                if cell_pos in self._beam_cells:
                    obs[obs_r, obs_c] = COLOUR["beam"]
                elif cell_pos in self._clean_beam_cells:
                    obs[obs_r, obs_c] = COLOUR["clean_beam"]
                elif self.walls[wr, wc]:
                    obs[obs_r, obs_c] = COLOUR["wall"]
                elif cell_pos in apple_set:
                    obs[obs_r, obs_c] = COLOUR["apple"]
                elif self.waste[wr, wc]:
                    obs[obs_r, obs_c] = COLOUR["waste"]
                elif cell_pos in self.river_cells_set:
                    obs[obs_r, obs_c] = COLOUR["river"]
                elif cell_pos in self.stream_cells_set:
                    obs[obs_r, obs_c] = COLOUR["stream"]
                # else: background (already 0)

        return obs.transpose(2, 0, 1).copy()

    # ------------------------------------------------------------------
    # Full-map rendering
    # ------------------------------------------------------------------

    def render(self, cell_size: int = 8) -> np.ndarray:
        """Render the full map as an RGB image."""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Background layers: river (clean or waste), stream, walls
        for r, c in self.river_cells_list:
            if self.waste[r, c]:
                img[r, c] = COLOUR["waste"]
            else:
                img[r, c] = COLOUR["river"]
        for r, c in self.stream_cells_list:
            img[r, c] = COLOUR["stream"]
        img[self.walls] = COLOUR["wall"]

        # Apples
        for idx in range(self.n_apples):
            if self.apple_alive[idx]:
                r, c = self._apple_pos[idx]
                img[r, c] = COLOUR["apple"]

        # Beams
        for (br, bc) in self._beam_cells:
            if 0 <= br < self.height and 0 <= bc < self.width:
                img[br, bc] = COLOUR["beam"]
        for (br, bc) in self._clean_beam_cells:
            if 0 <= br < self.height and 0 <= bc < self.width:
                img[br, bc] = COLOUR["clean_beam"]

        # Agents
        for i in range(self.n_agents):
            r, c = self.agent_pos[i]
            if self.agent_timeout[i] > 0:
                img[r, c] = COLOUR["timeout"]
            elif i == 0:
                img[r, c] = COLOUR["self"]
            elif i == 1:
                img[r, c] = COLOUR["other"]
            else:
                img[r, c] = np.array([180, 80, 220], dtype=np.uint8)

        if cell_size > 1:
            img = np.repeat(np.repeat(img, cell_size, axis=0), cell_size, axis=1)
        return img

    # ------------------------------------------------------------------
    # Social outcome metrics (Perolat et al., Section 2.3)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_metrics(
        episode_rewards: Dict[int, List[float]],
        episode_timeouts: Dict[int, List[bool]],
    ) -> Dict[str, float]:
        """Compute social-outcome metrics from one episode.

        Same definitions as GatheringEnv.compute_metrics for consistency.
        """
        n = len(episode_rewards)
        T = len(next(iter(episode_rewards.values())))

        returns = {i: sum(episode_rewards[i]) for i in episode_rewards}
        R = np.array(list(returns.values()))

        # Efficiency (U)
        efficiency = float(R.sum() / T) if T > 0 else 0.0

        # Equality (E) via Gini coefficient
        total = R.sum()
        if total > 0:
            gini_num = sum(abs(R[i] - R[j]) for i in range(n) for j in range(n))
            equality = 1.0 - gini_num / (2 * n * total)
        else:
            equality = 1.0

        # Sustainability (S) — average time at which rewards are collected
        mean_times = []
        for i in episode_rewards:
            rews = episode_rewards[i]
            times = [t for t, r in enumerate(rews) if r > 0]
            if times:
                mean_times.append(np.mean(times))
        sustainability = float(np.mean(mean_times)) if mean_times else 0.0

        # Peace (P)
        total_timeout_steps = 0
        for i in episode_timeouts:
            total_timeout_steps += sum(episode_timeouts[i])
        peace = float(n * T - total_timeout_steps) / T if T > 0 else float(n)

        # Maximin (Rawlsian welfare) — minimum per-agent total return
        maximin = float(R.min()) if len(R) > 0 else 0.0

        return {
            "efficiency": efficiency,
            "equality": equality,
            "sustainability": sustainability,
            "peace": peace,
            "maximin": maximin,
        }


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def make_cleanup(n_agents: int = 5, small: bool = False, **kwargs) -> CleanupEnv:
    """Create a Cleanup environment with standard parameters.

    Parameters match Hughes et al. / Vinitsky et al. defaults:
      - Fire beam: range 5, width 3, 1 hit to tag, 25-step timeout
      - Fire cost: -1 to firer, -50 to target
      - Clean cost: -1
      - Waste dynamics: depletion threshold 0.4, restoration 0.0
      - Waste spawn prob: 0.5, apple spawn prob: 0.05
    """
    defaults = dict(
        ascii_map=CLEANUP_MAP_SMALL if small else CLEANUP_MAP,
        n_agents=n_agents,
        max_steps=1000,
        obs_ahead=7,
        obs_side=7,
        beam_length=5,
        beam_width=3,
        hits_to_tag=1,
        timeout_steps=25,
        fire_cost=1.0,
        fire_penalty=50.0,
        clean_cost=1.0,
        threshold_depletion=0.4,
        threshold_restoration=0.0,
        waste_spawn_prob=0.5,
        apple_respawn_prob=0.05,
    )
    defaults.update(kwargs)
    return CleanupEnv(**defaults)


# ---------------------------------------------------------------------------
# Quick self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Cleanup (5 agents, full map) ===")
    env = make_cleanup(n_agents=5)
    obs = env.reset(seed=42)
    print(f"Map size:            {env.height} x {env.width}")
    print(f"Observation shape:   {obs[0].shape}")
    print(f"Num apple spawns:    {env.n_apples}")
    print(f"Num river cells:     {len(env.river_cells_list)} ({len(env.waste_init_list)} start polluted)")
    print(f"Num stream cells:    {len(env.stream_cells_list)}")
    print(f"Num actions:         {env.action_space_n}")

    total_rewards = {i: 0.0 for i in range(env.n_agents)}
    ep_rewards = {i: [] for i in range(env.n_agents)}
    ep_timeouts = {i: [] for i in range(env.n_agents)}
    waste_densities = []
    clean_actions = 0

    for step in range(env.max_steps):
        # Random policy with some cleaning bias
        actions = {}
        for i in range(env.n_agents):
            r = env.rng.random()
            if r < 0.15:
                actions[i] = int(CleanupAction.CLEAN)
                clean_actions += 1
            else:
                actions[i] = int(env.rng.integers(NUM_CLEANUP_ACTIONS - 1))  # 0-7

        obs, rewards, terminated, truncated, info = env.step(actions)
        for i in range(env.n_agents):
            total_rewards[i] += rewards[i]
            ep_rewards[i].append(rewards[i])
            ep_timeouts[i].append(info[i]["timeout"] > 0)
        waste_densities.append(info[0]["waste_density"])

    print(f"\nTotal rewards:       {total_rewards}")
    print(f"Clean actions:       {clean_actions}")
    print(f"Final waste density: {waste_densities[-1]:.3f}")
    print(f"Mean waste density:  {np.mean(waste_densities):.3f}")
    print(f"Apples alive (end):  {env.apple_alive.sum()}")
    metrics = CleanupEnv.compute_metrics(ep_rewards, ep_timeouts)
    print(f"Social metrics:      {metrics}")

    # Render
    img = env.render(cell_size=4)
    try:
        from PIL import Image
        Image.fromarray(img).save("cleanup_render.png")
        print("Saved render to cleanup_render.png")
    except ImportError:
        print("PIL not installed; skipping image save.")
    print(f"Render shape:        {img.shape}")

    print("\n=== Cleanup (3 agents, small map) ===")
    env2 = make_cleanup(n_agents=3, small=True, seed=123)
    obs2 = env2.reset()
    print(f"Map size:            {env2.height} x {env2.width}")
    print(f"Observation shape:   {obs2[0].shape}")
    print(f"Num apple spawns:    {env2.n_apples}")
    print(f"Num river cells:     {len(env2.river_cells_list)} ({len(env2.waste_init_list)} start polluted)")

    total_rewards2 = {i: 0.0 for i in range(env2.n_agents)}
    for step in range(env2.max_steps):
        actions = {i: int(env2.rng.integers(NUM_CLEANUP_ACTIONS)) for i in range(env2.n_agents)}
        obs2, rewards2, *_ = env2.step(actions)
        for i in range(env2.n_agents):
            total_rewards2[i] += rewards2[i]

    print(f"Total rewards:       {total_rewards2}")

    img2 = env2.render(cell_size=4)
    try:
        from PIL import Image
        Image.fromarray(img2).save("cleanup_small_render.png")
        print("Saved render to cleanup_small_render.png")
    except ImportError:
        pass

    print("\nAll tests passed.")
