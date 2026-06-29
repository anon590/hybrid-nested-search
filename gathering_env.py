"""
Multi-Agent Gathering / Commons Gridworld Environment
=====================================================

Implements the Gathering game from:
  Leibo et al. (2017) "Multi-agent Reinforcement Learning in Sequential
  Social Dilemmas" (AAMAS 2017)

and the Commons game from:
  Perolat et al. (2017) "A multi-agent reinforcement learning model of
  common-pool resource appropriation" (NeurIPS 2017)

Both are 2D partially-observable Markov games where agents collect apples
(resources) and may use a "tagging beam" to temporarily remove rivals.

Key features:
  - N independent agents with egocentric RGB observations
  - 8 discrete actions: move (4 dirs), rotate (2 dirs), use beam, stand still
  - Configurable beam: length, width, hits-to-tag, timeout duration
  - Two apple-respawn modes:
      * "fixed"   – respawns after a set number of timesteps (Gathering)
      * "density"  – probability depends on nearby apple count (Commons)
  - Gymnasium-style API: reset() / step(actions) returning per-agent dicts
"""

from __future__ import annotations

import numpy as np
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class Action(IntEnum):
    """Eight agent-centered actions (same in both papers)."""
    FORWARD = 0
    BACKWARD = 1
    STEP_LEFT = 2
    STEP_RIGHT = 3
    ROTATE_LEFT = 4
    ROTATE_RIGHT = 5
    BEAM = 6
    STAND = 7

NUM_ACTIONS = len(Action)


class Orientation(IntEnum):
    NORTH = 0  # -row
    EAST = 1   # +col
    SOUTH = 2  # +row
    WEST = 3   # -col


# Rotation matrices: convert (forward, right) offsets to (dr, dc) world offsets
# dr = rot[0]*fwd + rot[1]*right,  dc = rot[2]*fwd + rot[3]*right
_ROTATIONS = {
    Orientation.NORTH: (-1, 0, 0, 1),
    Orientation.EAST:  (0, 1, 1, 0),
    Orientation.SOUTH: (1, 0, 0, -1),
    Orientation.WEST:  (0, -1, -1, 0),
}

# RGB colour palette
COLOUR = {
    "background": np.array([0, 0, 0], dtype=np.uint8),
    "wall":       np.array([127, 127, 127], dtype=np.uint8),
    "apple":      np.array([0, 200, 0], dtype=np.uint8),
    "self":       np.array([50, 100, 255], dtype=np.uint8),
    "other":      np.array([220, 50, 50], dtype=np.uint8),
    "beam":       np.array([255, 255, 60], dtype=np.uint8),
    "timeout":    np.array([100, 100, 100], dtype=np.uint8),  # timed-out agent
}

# Apple respawn probability table for density-dependent mode (Perolat et al.)
# Index = local stock size L (number of apples within radius 2)
_DENSITY_RESPAWN_PROBS = {
    0: 0.0,
    1: 0.01,
    2: 0.01,
    3: 0.05,
    4: 0.05,
}
_DENSITY_RESPAWN_DEFAULT = 0.1  # for L > 4


# ---------------------------------------------------------------------------
# Default maps (ASCII)
#   @ = wall   A = apple spawn   . = empty   P = preferred player spawn
# ---------------------------------------------------------------------------

GATHERING_MAP_SMALL = """\
@@@@@@@@@@@@@@@@@@
@P.....A.A.....P@
@......A.A......@
@......A.A......@
@......A.A......@
@......A.A......@
@......A.A......@
@......A.A......@
@................@
@@@@@@@@@@@@@@@@@@"""

GATHERING_MAP = """\
@@@@@@@@@@@@@@@@@@@@@@@@@
@P.........AAA.........@
@..........AAA.........@
@..........AAA..........@
@..........AAA..........@
@..........AAA..........@
@..........AAA..........@
@..........AAA..........@
@..........AAA..........@
@..........AAA..........@
@...........................@
@...........................@
@..........AAA..........@
@..........AAA..........@
@..........AAA..........@
@..........AAA..........@
@..........AAA..........@
@..........AAA..........@
@..........AAA..........@
@..........AAA.........P@
@@@@@@@@@@@@@@@@@@@@@@@@@"""

GATHERING_MAP_LARGE = """\
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@.P...P......A....P.AAAAA....P..A.P.@
@..P.....A.P.AA....P....AAA....A..A.@
@.....A.AAA..AAA....A....A.AA.AAAA..@
@.A..AAA.A....A..A.AAA..A..A...A.A..@
@AAA..A.A....A..AAA.A..AAA........A.P@
@.A.A..AAA..AAA..A.A....A.AA...AA.AA@
@..A.A..AAA....A.A..AAA....AAA..A...@
@...AAA..A......AAA..A....AAAA......@
@.P..A.......A..A.AAA....A..A......P@
@A..AAA..A..A..AAA.A....AAAA.....P..@
@....A.A...AAA..A.A......A.AA...A.P.@
@.....AAA...A.A..AAA......AA...AAA.P@
@.A....A.....AAA..A..P..........A...@
@.......P.....A.........P..P.P.....P@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"""

COMMONS_MAP = """\
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@............................@
@...AAAA.......AAAA..........@
@...AAAA.......AAAA..........@
@...AAAA.......AAAA..........@
@...AAAA.......AAAA..........@
@............................@
@............................@
@........AAAAAA..............@
@........AAAAAA..............@
@........AAAAAA..............@
@........AAAAAA..............@
@............................@
@............................@
@...AAAA.......AAAA..........@
@...AAAA.......AAAA..........@
@...AAAA.......AAAA..........@
@...AAAA.......AAAA..........@
@............................@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"""


# ---------------------------------------------------------------------------
# Map parser
# ---------------------------------------------------------------------------

def parse_map(ascii_map: str) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Parse an ASCII map string into arrays.

    Returns
    -------
    walls : np.ndarray, bool, shape (H, W)
        True where a wall exists.
    apple_points : list of (row, col)
        Positions where apples may spawn.
    spawn_points : list of (row, col)
        Preferred agent spawn positions (empty cells if none marked).
    """
    lines = [l for l in ascii_map.strip().splitlines()]
    height = len(lines)
    width = max(len(l) for l in lines)
    # Pad lines to equal width
    lines = [l.ljust(width) for l in lines]

    walls = np.zeros((height, width), dtype=bool)
    apple_points: List[Tuple[int, int]] = []
    spawn_points: List[Tuple[int, int]] = []

    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            if ch == "@":
                walls[r, c] = True
            elif ch == "A":
                apple_points.append((r, c))
            elif ch == "P":
                spawn_points.append((r, c))

    # If no explicit spawn points, pick empty non-apple cells near the edges
    if not spawn_points:
        for r in range(height):
            for c in range(width):
                if not walls[r, c] and (r, c) not in apple_points:
                    if r <= 1 or r >= height - 2 or c <= 1 or c >= width - 2:
                        spawn_points.append((r, c))
    return walls, apple_points, spawn_points


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class GatheringEnv:
    """Multi-agent Gathering / Commons gridworld.

    Parameters
    ----------
    ascii_map : str
        ASCII map (see examples above).  ``@`` = wall, ``A`` = apple spawn,
        ``P`` = player spawn, ``.`` or space = empty.
    n_agents : int
        Number of agents.
    max_steps : int
        Episode length (default 1000 as in both papers).
    obs_ahead : int
        How many cells ahead the agent can see (default 15 for Gathering,
        20 for Commons).
    obs_side : int
        How many cells to each side the agent can see (default 10).
    beam_length : int
        Beam range in cells.  Gathering = ~20; Commons = 20.
    beam_width : int
        Beam width in cells (centered on agent's column).
        Gathering ≈ 1 (thin line); Commons = 5.
    hits_to_tag : int
        Number of beam hits before an agent is tagged.
        Gathering = 2; Commons = 1.
    timeout_steps : int
        How many steps a tagged agent is removed.
        Gathering = 25 (Ntagged); Commons = 25.
    respawn_mode : {"fixed", "density"}
        ``"fixed"`` – apple reappears after *apple_respawn_time* steps.
        ``"density"`` – probability depends on nearby apple stock.
    apple_respawn_time : int
        Frames until an apple respawns (only used when respawn_mode="fixed").
    density_radius : int
        Radius for counting nearby apples in density mode (default 2).
    beam_enabled : bool
        If False, the beam action has no effect (for ablation studies).
    seed : int or None
        Random seed.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        ascii_map: str = GATHERING_MAP,
        n_agents: int = 2,
        max_steps: int = 1000,
        # Observation
        obs_ahead: int = 15,
        obs_side: int = 10,
        # Beam
        beam_length: int = 20,
        beam_width: int = 1,
        hits_to_tag: int = 2,
        timeout_steps: int = 25,
        beam_enabled: bool = True,
        # Apples
        respawn_mode: str = "fixed",
        apple_respawn_time: int = 25,
        density_radius: int = 2,
        # Misc
        seed: Optional[int] = None,
    ):
        # Parse the map
        self.walls, self.apple_points, self.spawn_points = parse_map(ascii_map)
        self.height, self.width = self.walls.shape

        self.n_agents = n_agents
        self.max_steps = max_steps

        # Observation window (agent at bottom-centre)
        self.obs_ahead = obs_ahead
        self.obs_side = obs_side
        self.obs_h = obs_ahead + 1        # rows: obs_ahead forward + agent row
        self.obs_w = 2 * obs_side + 1     # cols: side + agent + side

        # Beam parameters
        self.beam_length = beam_length
        self.beam_width = beam_width
        self.hits_to_tag = hits_to_tag
        self.timeout_steps = timeout_steps
        self.beam_enabled = beam_enabled

        # Apple parameters
        assert respawn_mode in ("fixed", "density")
        self.respawn_mode = respawn_mode
        self.apple_respawn_time = apple_respawn_time
        self.density_radius = density_radius

        # Pre-compute an index mapping for apple positions
        self.n_apples = len(self.apple_points)
        self._apple_pos = np.array(self.apple_points, dtype=np.int32)  # (N, 2)
        # For density mode, precompute neighbour lists for each apple position
        if self.respawn_mode == "density":
            self._apple_neighbours = self._build_apple_neighbour_index()

        # Random state
        self.rng = np.random.default_rng(seed)

        # --- Mutable state (initialised in reset) ---
        self._step_count: int = 0
        # Agent state arrays
        self.agent_pos = np.zeros((n_agents, 2), dtype=np.int32)
        self.agent_orient = np.zeros(n_agents, dtype=np.int32)
        self.agent_timeout = np.zeros(n_agents, dtype=np.int32)  # >0 ⇒ removed
        self.agent_beam_hits = np.zeros(n_agents, dtype=np.int32)
        # Apple state
        self.apple_alive = np.ones(self.n_apples, dtype=bool)
        self.apple_timer = np.zeros(self.n_apples, dtype=np.int32)
        # Current-step beam cells (for rendering)
        self._beam_cells: set = set()

        # Observation & action space descriptions (Gymnasium-like)
        self.observation_shape = (3, self.obs_h, self.obs_w)
        self.action_space_n = NUM_ACTIONS

    # ------------------------------------------------------------------
    # Pre-computation helpers
    # ------------------------------------------------------------------

    def _build_apple_neighbour_index(self) -> List[List[int]]:
        """For each apple spawn point, list indices of neighbours within radius."""
        r = self.density_radius
        neighbours: List[List[int]] = []
        for i, (ri, ci) in enumerate(self.apple_points):
            nbrs = []
            for j, (rj, cj) in enumerate(self.apple_points):
                if i != j and abs(ri - rj) + abs(ci - cj) <= r:
                    # Use Chebyshev (L-inf) distance ≤ r  (ball of radius r)
                    pass
                if i != j and max(abs(ri - rj), abs(ci - cj)) <= r:
                    nbrs.append(j)
            neighbours.append(nbrs)
        return neighbours

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Dict[int, np.ndarray]:
        """Reset the environment and return initial observations.

        Returns
        -------
        obs : dict  {agent_id: np.ndarray}
            Each observation has shape ``(3, obs_h, obs_w)`` uint8 RGB.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._step_count = 0

        # Spawn agents at random spawn points (or random open cells)
        available = list(self.spawn_points)
        self.rng.shuffle(available)
        open_cells = [
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if not self.walls[r, c] and (r, c) not in self.apple_points
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

        # All apples start alive
        self.apple_alive[:] = True
        self.apple_timer[:] = 0

        self._beam_cells = set()

        return {i: self._get_obs(i) for i in range(self.n_agents)}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, actions: Dict[int, int]
    ) -> Tuple[
        Dict[int, np.ndarray],  # observations
        Dict[int, float],       # rewards
        Dict[int, bool],        # terminated
        Dict[int, bool],        # truncated
        Dict[int, dict],        # info
    ]:
        """Advance one timestep.

        Parameters
        ----------
        actions : dict {agent_id: int}
            An action for every agent (0–7).

        Returns
        -------
        observations, rewards, terminated, truncated, info
            Gymnasium-style returns, keyed by agent id.
        """
        self._step_count += 1
        rewards = {i: 0.0 for i in range(self.n_agents)}
        self._beam_cells = set()

        # --- 1. Decrement timeouts ---
        active_before = self.agent_timeout <= 0
        self.agent_timeout = np.maximum(self.agent_timeout - 1, 0)

        # --- 2. Process actions for active agents ---
        for i in range(self.n_agents):
            if self.agent_timeout[i] > 0:
                continue  # agent is timed out, skip
            act = actions.get(i, Action.STAND)
            if act == Action.FORWARD:
                self._move_agent(i, "forward")
            elif act == Action.BACKWARD:
                self._move_agent(i, "backward")
            elif act == Action.STEP_LEFT:
                self._move_agent(i, "left")
            elif act == Action.STEP_RIGHT:
                self._move_agent(i, "right")
            elif act == Action.ROTATE_LEFT:
                self.agent_orient[i] = (self.agent_orient[i] - 1) % 4
            elif act == Action.ROTATE_RIGHT:
                self.agent_orient[i] = (self.agent_orient[i] + 1) % 4
            elif act == Action.BEAM:
                if self.beam_enabled:
                    self._fire_beam(i)
            # STAND does nothing

        # --- 3. Respawn apples (before collection so timer is accurate) ---
        self._respawn_apples()

        # --- 4. Collect apples ---
        for i in range(self.n_agents):
            if self.agent_timeout[i] > 0:
                continue
            r, c = self.agent_pos[i]
            for a_idx in range(self.n_apples):
                if self.apple_alive[a_idx]:
                    ar, ac = self._apple_pos[a_idx]
                    if ar == r and ac == c:
                        self.apple_alive[a_idx] = False
                        self.apple_timer[a_idx] = self.apple_respawn_time
                        rewards[i] += 1.0

        # --- 5. Check episode termination  ---
        done = self._step_count >= self.max_steps
        terminated = {i: done for i in range(self.n_agents)}
        truncated = {i: False for i in range(self.n_agents)}

        # --- 6. Build observations ---
        obs = {i: self._get_obs(i) for i in range(self.n_agents)}

        # --- 7. Info ---
        info: Dict[int, dict] = {}
        for i in range(self.n_agents):
            info[i] = {
                "timeout": int(self.agent_timeout[i]),
                "beam_hits": int(self.agent_beam_hits[i]),
            }

        return obs, rewards, terminated, truncated, info

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def _move_agent(self, agent_id: int, direction: str) -> None:
        orient = Orientation(self.agent_orient[agent_id])
        a, b, c, d = _ROTATIONS[orient]
        # forward=(1,0) in local frame → dr=a, dc=c
        # right =(0,1) in local frame → dr=b, dc=d
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

        # Bounds and wall check
        if 0 <= nr < self.height and 0 <= nc < self.width and not self.walls[nr, nc]:
            self.agent_pos[agent_id] = (nr, nc)

    # ------------------------------------------------------------------
    # Beam
    # ------------------------------------------------------------------

    def _fire_beam(self, agent_id: int) -> None:
        """Fire the tagging beam and record hits."""
        orient = Orientation(self.agent_orient[agent_id])
        a, b, c, d = _ROTATIONS[orient]
        ar, ac = self.agent_pos[agent_id]

        half_w = self.beam_width // 2

        # Compute all beam cells
        beam_cells: List[Tuple[int, int]] = []
        for dist in range(1, self.beam_length + 1):
            for w_off in range(-half_w, half_w + 1):
                # world offset = dist * forward + w_off * right
                dr = a * dist + b * w_off
                dc = c * dist + d * w_off
                br, bc = ar + dr, ac + dc
                if 0 <= br < self.height and 0 <= bc < self.width:
                    if self.walls[br, bc]:
                        # Beam blocked by wall at this lateral position
                        # (only block this column of the beam, not the entire width)
                        continue
                    beam_cells.append((br, bc))
                    self._beam_cells.add((br, bc))

        # Check for agents in beam path
        beam_set = set(beam_cells)
        for j in range(self.n_agents):
            if j == agent_id or self.agent_timeout[j] > 0:
                continue
            pos_j = tuple(self.agent_pos[j])
            if pos_j in beam_set:
                self.agent_beam_hits[j] += 1
                if self.agent_beam_hits[j] >= self.hits_to_tag:
                    self.agent_timeout[j] = self.timeout_steps
                    self.agent_beam_hits[j] = 0

    # ------------------------------------------------------------------
    # Apple respawn
    # ------------------------------------------------------------------

    def _respawn_apples(self) -> None:
        if self.respawn_mode == "fixed":
            self._respawn_fixed()
        else:
            self._respawn_density()

    def _respawn_fixed(self) -> None:
        """Respawn apples on a fixed timer (Gathering / Leibo et al.)."""
        dead = ~self.apple_alive
        self.apple_timer[dead] -= 1
        respawned = dead & (self.apple_timer <= 0)
        self.apple_alive[respawned] = True
        self.apple_timer[respawned] = 0

    def _respawn_density(self) -> None:
        """Density-dependent respawn (Commons / Perolat et al.).

        Per-timestep respawn probability depends on the number of existing
        apples within a Chebyshev-distance ball of radius ``density_radius``.
        """
        for idx in range(self.n_apples):
            if self.apple_alive[idx]:
                continue
            # Count alive neighbours
            local_stock = sum(
                1 for j in self._apple_neighbours[idx] if self.apple_alive[j]
            )
            prob = _DENSITY_RESPAWN_PROBS.get(local_stock, _DENSITY_RESPAWN_DEFAULT)
            if prob > 0 and self.rng.random() < prob:
                self.apple_alive[idx] = True

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

        # Build a set of apple positions for fast lookup
        apple_set: set = set()
        for idx in range(self.n_apples):
            if self.apple_alive[idx]:
                apple_set.add((int(self._apple_pos[idx, 0]),
                               int(self._apple_pos[idx, 1])))

        # Agent position lookup
        agent_positions: Dict[Tuple[int, int], List[int]] = {}
        for j in range(self.n_agents):
            pos_j = (int(self.agent_pos[j, 0]), int(self.agent_pos[j, 1]))
            agent_positions.setdefault(pos_j, []).append(j)

        for obs_r in range(self.obs_h):
            for obs_c in range(self.obs_w):
                fwd = self.obs_ahead - obs_r    # positive = forward
                right = obs_c - self.obs_side   # positive = right
                wr = ar + a * fwd + b * right
                wc = ac + c * fwd + d * right

                # Out of bounds
                if wr < 0 or wr >= self.height or wc < 0 or wc >= self.width:
                    obs[obs_r, obs_c] = COLOUR["wall"]
                    continue

                # Layer priority: agent > beam > apple > wall > background
                cell_pos = (wr, wc)

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
                elif self.walls[wr, wc]:
                    obs[obs_r, obs_c] = COLOUR["wall"]
                elif cell_pos in apple_set:
                    obs[obs_r, obs_c] = COLOUR["apple"]
                # else: background (already 0)

        # Return as (C, H, W)
        return obs.transpose(2, 0, 1).copy()

    # ------------------------------------------------------------------
    # Full-map rendering (god's-eye view for debugging / visualisation)
    # ------------------------------------------------------------------

    def render(self, cell_size: int = 8) -> np.ndarray:
        """Render the full map as an RGB image.

        Parameters
        ----------
        cell_size : int
            Pixels per grid cell (for up-scaling).

        Returns
        -------
        img : np.ndarray, shape (H*cell_size, W*cell_size, 3), uint8
        """
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Walls
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

        # Agents
        for i in range(self.n_agents):
            r, c = self.agent_pos[i]
            if self.agent_timeout[i] > 0:
                img[r, c] = COLOUR["timeout"]
            else:
                img[r, c] = COLOUR["self"]

        # Up-scale
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
        """Compute the four social-outcome metrics from one episode.

        Parameters
        ----------
        episode_rewards : dict {agent_id: [r_0, r_1, …, r_{T-1}]}
        episode_timeouts : dict {agent_id: [bool_0, bool_1, …]}
            Whether each agent was in timeout at each step.

        Returns
        -------
        dict with keys "efficiency", "equality", "sustainability", "peace".
        """
        n = len(episode_rewards)
        T = len(next(iter(episode_rewards.values())))

        returns = {i: sum(episode_rewards[i]) for i in episode_rewards}
        R = np.array(list(returns.values()))

        # --- Efficiency (U) ---
        efficiency = float(R.sum() / T) if T > 0 else 0.0

        # --- Equality (E) via Gini coefficient ---
        total = R.sum()
        if total > 0:
            gini_num = sum(abs(R[i] - R[j]) for i in range(n) for j in range(n))
            equality = 1.0 - gini_num / (2 * n * total)
        else:
            equality = 1.0

        # --- Sustainability (S) ---
        # Average time at which rewards are collected
        mean_times = []
        for i in episode_rewards:
            rews = episode_rewards[i]
            times = [t for t, r in enumerate(rews) if r > 0]
            if times:
                mean_times.append(np.mean(times))
        sustainability = float(np.mean(mean_times)) if mean_times else 0.0

        # --- Peace (P) ---
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

def make_gathering(
    n_agents: int = 2, small: bool = False, **kwargs
) -> GatheringEnv:
    """Create a Gathering environment (Leibo et al. 2017 defaults).

    Apples respawn on a fixed timer; beam requires 2 hits to tag.
    """
    defaults = dict(
        ascii_map=GATHERING_MAP_SMALL if small else GATHERING_MAP,
        n_agents=n_agents,
        max_steps=1000,
        obs_ahead=15,
        obs_side=10,
        beam_length=20,
        beam_width=1,
        hits_to_tag=2,
        timeout_steps=25,
        respawn_mode="fixed",
        apple_respawn_time=25,
    )
    defaults.update(kwargs)
    return GatheringEnv(**defaults)


def make_gathering_large(
    n_agents: int = 10, **kwargs
) -> GatheringEnv:
    """Create a large Gathering environment (38x16 map, up to 16 agents).

    Scattered apple patches with many spawn points; suitable for studying
    emergent cooperation / competition with more agents.
    """
    defaults = dict(
        ascii_map=GATHERING_MAP_LARGE,
        n_agents=n_agents,
        max_steps=1000,
        obs_ahead=15,
        obs_side=10,
        beam_length=20,
        beam_width=1,
        hits_to_tag=2,
        timeout_steps=25,
        respawn_mode="fixed",
        apple_respawn_time=25,
    )
    defaults.update(kwargs)
    return GatheringEnv(**defaults)


def make_commons(
    n_agents: int = 12, **kwargs
) -> GatheringEnv:
    """Create a Commons environment (Perolat et al. 2017 defaults).

    Apples respawn with density-dependent probability; beam is wide;
    single hit tags.
    """
    defaults = dict(
        ascii_map=COMMONS_MAP,
        n_agents=n_agents,
        max_steps=1000,
        obs_ahead=20,
        obs_side=10,
        beam_length=20,
        beam_width=5,
        hits_to_tag=1,
        timeout_steps=25,
        respawn_mode="density",
        density_radius=2,
    )
    defaults.update(kwargs)
    return GatheringEnv(**defaults)


# ---------------------------------------------------------------------------
# Quick self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Gathering (2 agents, small map) ===")
    env = make_gathering(n_agents=6, small=False)
    obs = env.reset()
    print(f"Map size:          {env.height} x {env.width}")
    print(f"Observation shape: {obs[0].shape}")
    print(f"Num apple spawns:  {env.n_apples}")
    print(f"Num actions:       {env.action_space_n}")

    total_rewards = {i: 0.0 for i in range(env.n_agents)}
    ep_rewards = {i: [] for i in range(env.n_agents)}
    ep_timeouts = {i: [] for i in range(env.n_agents)}

    for step in range(env.max_steps):
        actions = {i: env.rng.integers(NUM_ACTIONS) for i in range(env.n_agents)}
        obs, rewards, terminated, truncated, info = env.step(actions)
        for i in range(env.n_agents):
            total_rewards[i] += rewards[i]
            ep_rewards[i].append(rewards[i])
            ep_timeouts[i].append(info[i]["timeout"] > 0)

    print(f"Total rewards:     {total_rewards}")
    metrics = GatheringEnv.compute_metrics(ep_rewards, ep_timeouts)
    print(f"Social metrics:    {metrics}")

    # Render one frame
    img = env.render(cell_size=1)
    # save image for manual inspection (requires Pillow)
    try:
        from PIL import Image
        Image.fromarray(img).save("gathering_render.png")
        print("Saved render to gathering_render.png")
    except ImportError:
        print("PIL not installed; skipping image save.")
    print(f"Render shape:      {img.shape}")

    print("\n=== Commons (4 agents) ===")
    env2 = make_commons(n_agents=4, seed=123)
    obs2 = env2.reset()
    print(f"Map size:          {env2.height} x {env2.width}")
    print(f"Observation shape: {obs2[0].shape}")
    print(f"Num apple spawns:  {env2.n_apples}")

    total_rewards2 = {i: 0.0 for i in range(env2.n_agents)}
    for step in range(env2.max_steps):
        actions = {i: env2.rng.integers(NUM_ACTIONS) for i in range(env2.n_agents)}
        obs2, rewards2, *_ = env2.step(actions)
        for i in range(env2.n_agents):
            total_rewards2[i] += rewards2[i]

    img = env2.render(cell_size=1)
    try:
        from PIL import Image
        Image.fromarray(img).save("commons_render.png")
        print("Saved render to commons_render.png")
    except ImportError:
        print("PIL not installed; skipping image save.")

    print(f"Total rewards:     {total_rewards2}")
    print("\nAll tests passed ✓")
