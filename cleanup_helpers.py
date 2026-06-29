"""
Skill / role helper library for the LLM-driven Hybrid Nested Search on Cleanup.

The hybrid-search method note (paper.pdf, Sec. 6.1, Testbed A) describes the
Cleanup sketches the LLM proposes as a STRUCTURE that maps the global pollution
level to a *number of cleaners*, e.g.

  (S1) waste-adaptive threshold ladder mapping pollution -> cleaner count;
  (S2) proportional controller  n_c = round(g * waste_ratio);
  (S3) fixed per-agent roles.

This module supplies the *low-level skills* so the LLM only has to write the
high-level structure (the waste_ratio -> n_cleaners map + the role dispatch),
exactly the way the meta-optimizer hands the LLM a black-box grad oracle so it
only writes the update rule. The skills are:

  waste_ratio(env)                  -> float    global pollution fraction in [0,1]
  assign_roles(env, n_cleaners)     -> dict     deterministic team role split
  clean_action(env, agent_id)       -> int      navigate to waste & fire CLEAN
  harvest_action(env, agent_id)     -> int      navigate to apples & collect
  bfs_step(env, agent_id, targets)  -> (dr,dc)  first BFS step toward a cell set
  direction_to_action(dr, dc, ori)  -> int      world step -> CleanupAction int

All skills are deterministic functions of the environment state, so every agent
running the same policy reconstructs the *same* global role assignment from
common knowledge (the coordination-via-common-state mechanism the cooperative
Voronoi policy already uses in the SSD codebase). `assign_roles` caches its
result per timestep on the env object so the O(n_agents) recomputation that
would otherwise happen once per agent per step is done only once.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from gathering_env import Orientation, _ROTATIONS
from cleanup_env import CleanupAction

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# --------------------------------------------------------------------------- #
# Global state read-outs
# --------------------------------------------------------------------------- #
def waste_ratio(env) -> float:
    """Fraction of river cells currently polluted (0.0 = pristine, 1.0 = full).

    This is the single observable that the proposed structures threshold on.
    The *dynamics constants* that determine the optimal threshold
    (waste_spawn_prob, threshold_depletion, apple_respawn_prob) are deliberately
    NOT exposed to the policy -- that hidden coupling is what creates the
    parameter-tuning gap the inner CMA-ES de-aliases.
    """
    cells = env.river_cells_list
    if not cells:
        return 0.0
    w = env.waste
    polluted = 0
    for (r, c) in cells:
        if w[r, c]:
            polluted += 1
    return polluted / len(cells)


def _alive_apple_set(env) -> set:
    out = set()
    for idx in range(env.n_apples):
        if env.apple_alive[idx]:
            out.add((int(env._apple_pos[idx, 0]), int(env._apple_pos[idx, 1])))
    return out


def _waste_cells(env) -> list:
    return [(r, c) for (r, c) in env.river_cells_list if env.waste[r, c]]


# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #
def _bfs_first_step(env, start, target_set):
    """First step (dr, dc) on a shortest path from `start` to nearest cell in
    `target_set`; (0, 0) if already there; None if unreachable."""
    if not target_set:
        return None
    if start in target_set:
        return (0, 0)
    visited = {start}
    queue = deque()
    for dr, dc in _DIRS:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < env.height and 0 <= nc < env.width and not env.walls[nr, nc]:
            pos = (nr, nc)
            if pos not in visited:
                visited.add(pos)
                if pos in target_set:
                    return (dr, dc)
                queue.append((nr, nc, dr, dc))
    while queue:
        r, c, fdr, fdc = queue.popleft()
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < env.height and 0 <= nc < env.width and not env.walls[nr, nc]:
                pos = (nr, nc)
                if pos not in visited:
                    visited.add(pos)
                    if pos in target_set:
                        return (fdr, fdc)
                    queue.append((nr, nc, fdr, fdc))
    return None


def bfs_step(env, agent_id, target_set):
    """Public BFS wrapper: first step from agent `agent_id` toward `target_set`."""
    start = (int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1]))
    return _bfs_first_step(env, start, set(target_set))


def direction_to_action(dr, dc, orientation) -> int:
    """Convert a world-frame cardinal step (dr, dc) into a CleanupAction int,
    given the agent's orientation. Agents strafe in all 4 directions without
    rotating, so a cardinal step never needs a rotation first."""
    if dr == 0 and dc == 0:
        return int(CleanupAction.STAND)
    a, b, c, d = _ROTATIONS[Orientation(int(orientation))]
    if (dr, dc) == (a, c):
        return int(CleanupAction.FORWARD)
    if (dr, dc) == (-a, -c):
        return int(CleanupAction.BACKWARD)
    if (dr, dc) == (-b, -d):
        return int(CleanupAction.STEP_LEFT)
    if (dr, dc) == (b, d):
        return int(CleanupAction.STEP_RIGHT)
    return int(CleanupAction.FORWARD)


# --------------------------------------------------------------------------- #
# Role assignment (team-level, cached per step)
# --------------------------------------------------------------------------- #
def assign_roles(env, n_cleaners) -> dict:
    """Split the team into `n_cleaners` cleaners + the rest harvesters.

    The n_cleaners agents *closest* to the pollution (Manhattan distance to the
    nearest waste cell, or to the nearest river cell when the river is clean)
    are assigned to clean; ties broken by agent id for determinism. Returns
    {agent_id: 'clean' | 'harvest'}. Cached on the env per timestep.
    """
    n = env.n_agents
    try:
        n_cleaners = int(round(float(n_cleaners)))
    except (TypeError, ValueError):
        n_cleaners = 0
    n_cleaners = max(0, min(n, n_cleaners))

    step = int(getattr(env, "_step_count", 0))
    cache = getattr(env, "_role_cache", None)
    if cache is not None and cache[0] == step and cache[1] == n_cleaners:
        return cache[2]

    targets = _waste_cells(env)
    if not targets:
        targets = list(env.river_cells_list)

    dists = []
    for i in range(n):
        ar, ac = int(env.agent_pos[i, 0]), int(env.agent_pos[i, 1])
        if targets:
            d = min(abs(ar - r) + abs(ac - c) for (r, c) in targets)
        else:
            d = 0
        dists.append((d, i))
    dists.sort()
    cleaners = set(i for _, i in dists[:n_cleaners])
    assign = {i: ("clean" if i in cleaners else "harvest") for i in range(n)}

    env._role_cache = (step, n_cleaners, assign)
    return assign


# --------------------------------------------------------------------------- #
# Skills: turn a role into a concrete action
# --------------------------------------------------------------------------- #
def _beam_hits_waste(env, ar, ac, orient_val, waste_set) -> bool:
    """Would firing CLEAN from (ar, ac) facing `orient_val` remove >=1 waste?
    Mirrors CleanupEnv._fire_clean_beam geometry (walls are skipped, not
    blocking)."""
    a, b, c, d = _ROTATIONS[Orientation(int(orient_val))]
    half_w = env.beam_width // 2
    for dist in range(1, env.beam_length + 1):
        for w_off in range(-half_w, half_w + 1):
            br = ar + a * dist + b * w_off
            bc = ac + c * dist + d * w_off
            if 0 <= br < env.height and 0 <= bc < env.width:
                if env.walls[br, bc]:
                    continue
                if (br, bc) in waste_set:
                    return True
    return False


def clean_action(env, agent_id) -> int:
    """Cleaner skill: fire CLEAN when waste is in the beam cone; otherwise
    rotate to face waste, or navigate toward the nearest waste cell."""
    if int(env.agent_timeout[agent_id]) > 0:
        return int(CleanupAction.STAND)

    ar, ac = int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1])
    orient = int(env.agent_orient[agent_id])
    waste = _waste_cells(env)
    if not waste:
        return int(CleanupAction.STAND)
    waste_set = set(waste)

    # 1. Already facing waste in range -> fire.
    if _beam_hits_waste(env, ar, ac, orient, waste_set):
        return int(CleanupAction.CLEAN)

    # 2. One rotation away from facing waste -> rotate toward it.
    left = (orient - 1) % 4
    right = (orient + 1) % 4
    hit_left = _beam_hits_waste(env, ar, ac, left, waste_set)
    hit_right = _beam_hits_waste(env, ar, ac, right, waste_set)
    if hit_left and not hit_right:
        return int(CleanupAction.ROTATE_LEFT)
    if hit_right and not hit_left:
        return int(CleanupAction.ROTATE_RIGHT)
    if hit_left and hit_right:
        return int(CleanupAction.ROTATE_RIGHT)

    # 3. Navigate toward the nearest waste cell.
    step = _bfs_first_step(env, (ar, ac), waste_set)
    if step is None:
        return int(CleanupAction.STAND)
    if step == (0, 0):
        # Standing on a waste cell: turn to sweep an adjacent waste run.
        return int(CleanupAction.ROTATE_RIGHT)
    return direction_to_action(step[0], step[1], orient)


def harvest_action(env, agent_id) -> int:
    """Harvester skill: BFS to the nearest live apple and step onto it; if the
    orchard is momentarily empty, move toward apple spawns to be ready."""
    if int(env.agent_timeout[agent_id]) > 0:
        return int(CleanupAction.STAND)

    orient = int(env.agent_orient[agent_id])
    apples = _alive_apple_set(env)
    if apples:
        step = bfs_step(env, agent_id, apples)
        if step is not None:
            return direction_to_action(step[0], step[1], orient)
        return int(CleanupAction.STAND)

    # Orchard empty: position near apple spawns for the next respawn.
    spawns = set((int(env._apple_pos[i, 0]), int(env._apple_pos[i, 1]))
                 for i in range(env.n_apples))
    step = bfs_step(env, agent_id, spawns)
    if step is None:
        return int(CleanupAction.STAND)
    return direction_to_action(step[0], step[1], orient)


# --------------------------------------------------------------------------- #
# Namespace assembly for compiled LLM policies
# --------------------------------------------------------------------------- #
def policy_namespace() -> dict:
    """The skill/helper names made available to a compiled LLM policy."""
    import math
    return {
        "np": np,
        "numpy": np,
        "math": math,
        "deque": deque,
        "CleanupAction": CleanupAction,
        "Orientation": Orientation,
        "waste_ratio": waste_ratio,
        "assign_roles": assign_roles,
        "clean_action": clean_action,
        "harvest_action": harvest_action,
        "bfs_step": bfs_step,
        "direction_to_action": direction_to_action,
    }
