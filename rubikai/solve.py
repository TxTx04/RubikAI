"""
Solveurs guidés par le réseau de neurones.

- greedy_solve : descend gloutonnement vers la plus petite valeur estimée (rapide,
  utile pour évaluer le réseau pendant l'entraînement).
- astar_solve  : recherche A* avec heuristique = valeur du réseau (meilleure qualité).
  f = g (coups joués) + poids * h (estimation du réseau).
"""
from __future__ import annotations
import heapq
import numpy as np
import torch
from cube import Cube
from model import CubeNet, encode_batch


def _neighbors(cube, colors):
    return np.stack([colors[cube.perms[m]] for m in cube.moveset])  # (M, S)


def greedy_solve(cube, net, start_colors, max_steps=60):
    colors = start_colors.copy()
    solved = cube.solved
    moves = cube.moveset
    path, visited = [], {start_colors.tobytes()}
    for _ in range(max_steps):
        if np.array_equal(colors, solved):
            return path
        neigh = _neighbors(cube, colors)
        vals = net.value(encode_batch(neigh)).cpu().numpy()
        order = np.argsort(vals)
        chosen = None
        for idx in order:
            nk = neigh[idx].tobytes()
            if np.array_equal(neigh[idx], solved):
                chosen = idx
                break
            if nk not in visited:
                chosen = idx
                break
        if chosen is None:
            chosen = int(order[0])           # tout visité : on force (rare)
        visited.add(neigh[chosen].tobytes())
        colors = neigh[chosen]
        path.append(moves[chosen])
    return path if np.array_equal(colors, solved) else None


def astar_solve(cube, net, start_colors, weight=1.2, max_expansions=100000):
    solved = cube.solved
    moves = cube.moveset
    start_key = start_colors.tobytes()
    if np.array_equal(start_colors, solved):
        return []
    h0 = float(net.value(encode_batch(start_colors)).item())
    cnt = 0
    openh = [(weight * h0, cnt, 0, start_colors, [])]
    best_g = {start_key: 0}
    exp = 0
    while openh and exp < max_expansions:
        f, _, g, colors, path = heapq.heappop(openh)
        if np.array_equal(colors, solved):
            return path
        exp += 1
        neigh = _neighbors(cube, colors)
        hs = net.value(encode_batch(neigh)).cpu().numpy()
        for j, m in enumerate(moves):
            nc = neigh[j]
            ng = g + 1
            if np.array_equal(nc, solved):
                return path + [m]
            nk = nc.tobytes()
            if nk in best_g and best_g[nk] <= ng:
                continue
            best_g[nk] = ng
            cnt += 1
            heapq.heappush(openh, (ng + weight * float(hs[j]), cnt, ng, nc, path + [m]))
    return None


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cube = Cube(ckpt["N"])
    net = CubeNet(ckpt["in_dim"], len(cube.moveset))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return cube, net


if __name__ == "__main__":
    import sys
    from solver_bfs import solve_optimal
    path = sys.argv[1] if len(sys.argv) > 1 else "model_2x2.pt"
    cube, net = load_model(path)
    rng = np.random.default_rng(123)
    print(f"Demo solveur IA ({cube.N}x{cube.N})  —  comparaison avec l'optimal exact (BFS)\n")
    n_ok = 0
    gaps = []
    for i in range(10):
        cube.reset()
        cube.scramble(12, rng=rng)
        s = cube.colors.copy()
        sol = astar_solve(cube, net, s, weight=1.2)
        opt = solve_optimal(cube, s)
        check = s.copy()
        for m in (sol or []):
            check = check[cube.perms[m]]
        ok = sol is not None and cube.is_solved(check)
        n_ok += ok
        if ok:
            gaps.append(len(sol) - len(opt))
        ia = f"{len(sol)} coups" if sol else "ECHEC"
        print(f"#{i+1:2d}  IA: {ia:9s}  | optimal: {len(opt)} coups  | {'OK' if ok else 'X'}")
    print(f"\nResolus: {n_ok}/10   ecart moyen a l'optimal: "
          f"{(sum(gaps)/len(gaps)) if gaps else float('nan'):.2f} coups")
