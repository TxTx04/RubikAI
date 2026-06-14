"""
Solveur OPTIMAL exact par recherche bidirectionnelle (BFS des deux côtés).

- On cherche en avant depuis l'état mélangé et en arrière depuis l'état résolu.
- Les deux fronts se rencontrent "au milieu" : la solution trouvée est de
  longueur MINIMALE garantie (optimale).
- Utilisable en pratique sur le 2x2 (espace d'états ~3,6 millions, profondeur <= 14).

Sert de vérité-terrain pour évaluer la qualité de l'IA entraînée.
"""
from __future__ import annotations
import numpy as np
from cube import Cube


def bidirectional_bfs(cube: Cube, start_colors: np.ndarray, max_depth: int = 16):
    perms = cube.perms
    moves = cube.moveset
    solved = cube.solved

    start_key = start_colors.tobytes()
    goal_key = solved.tobytes()
    if start_key == goal_key:
        return []

    # parent[key] = (clé_parent, coup_appliqué, colors)
    par_f = {start_key: (None, None, start_colors.copy())}
    par_b = {goal_key: (None, None, solved.copy())}
    front_f = [start_key]
    front_b = [goal_key]

    def expand(frontier, par, other):
        meet = None
        nxt = []
        for k in frontier:
            cols = par[k][2]
            for m in moves:
                nc = cols[perms[m]]
                nk = nc.tobytes()
                if nk not in par:
                    par[nk] = (k, m, nc)
                    nxt.append(nk)
                    if nk in other:
                        return nxt, nk
        return nxt, meet

    depth = 0
    while front_f and front_b and depth < max_depth:
        # on développe toujours le plus petit front (plus efficace)
        if len(front_f) <= len(front_b):
            front_f, meet = expand(front_f, par_f, par_b)
        else:
            front_b, meet = expand(front_b, par_b, par_f)
        depth += 1
        if meet:
            return _reconstruct(par_f, par_b, meet)
    return None  # pas trouvé dans la limite de profondeur


def _path(par, key):
    """Remonte les coups du départ jusqu'à key."""
    seq = []
    while par[key][1] is not None:
        pk, m, _ = par[key]
        seq.append(m)
        key = pk
    return list(reversed(seq))


def _reconstruct(par_f, par_b, meet):
    forward = _path(par_f, meet)                 # départ -> meet
    backward = _path(par_b, meet)                # résolu -> meet
    # meet -> résolu = inverses des coups backward, en ordre inverse
    tail = [Cube.inverse_move(m) for m in reversed(backward)]
    return forward + tail


def solve_optimal(cube: Cube, start_colors: np.ndarray, max_depth: int = 16):
    return bidirectional_bfs(cube, start_colors, max_depth)


if __name__ == "__main__":
    # Démo : on mélange un 2x2 et on le résout de façon optimale
    cube = Cube(2)
    rng = np.random.default_rng(7)
    for trial in range(5):
        cube.reset()
        seq = cube.scramble(8, rng=rng)
        scrambled = cube.colors.copy()
        sol = solve_optimal(cube, scrambled)
        # vérifie que la solution résout vraiment le cube
        check = scrambled.copy()
        for m in sol:
            check = check[cube.perms[m]]
        ok = cube.is_solved(check)
        print(f"Melange ({len(seq)} coups): {' '.join(seq)}")
        print(f"  -> Solution OPTIMALE ({len(sol)} coups): {' '.join(sol)}  [{'OK' if ok else 'ECHEC'}]")
