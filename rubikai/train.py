"""
Entraînement par Autodidactic Iteration (méthode DeepCube).

Principe (itération de valeur approximée) :
  1. On génère des états en mélangeant le cube résolu de 1 à Kmax coups.
  2. Pour chaque état s, on regarde ses voisins (un par coup possible).
     La cible de valeur est :  v_cible(s) = min_coup [ 1 + v_reseau(voisin) ],
     avec v(état résolu) = 0.  -> la valeur 0 connue du cube résolu se propage.
  3. On entraîne le réseau à prédire cette cible (régression).
  4. Un "réseau cible" figé périodiquement stabilise l'apprentissage.

Le réseau n'apprend QUE par auto-jeu : il ne voit jamais de solution humaine.
"""
from __future__ import annotations
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from cube import Cube
from model import CubeNet, encode_batch
from solve import greedy_solve


def generate_scrambles(cube, B, Kmax, rng):
    """Génère B états mélangés de profondeur 1..Kmax (vectorisé)."""
    S = cube.n_stickers
    Pmat = np.stack([cube.perms[m] for m in cube.moveset])  # (M, S)
    M = Pmat.shape[0]
    identity = np.arange(S)
    depths = rng.integers(1, Kmax + 1, size=B)
    colors = np.tile(cube.solved, (B, 1)).astype(np.int8)
    for step in range(Kmax):
        m_idx = rng.integers(0, M, size=B)
        sel = Pmat[m_idx].copy()
        sel[step >= depths] = identity                      # couche inactive
        colors = np.take_along_axis(colors, sel, axis=1)
    return colors, depths


def compute_targets(cube, colors, target_net):
    """Calcule les cibles de valeur et de politique via le réseau cible."""
    S = cube.n_stickers
    B = colors.shape[0]
    moves = cube.moveset
    M = len(moves)
    solved = cube.solved
    neigh = np.stack([colors[:, cube.perms[m]] for m in moves], axis=1)  # (B,M,S)
    solved_mask = (neigh == solved).all(axis=2)             # voisin résolu ?
    with torch.no_grad():
        v = target_net.value(encode_batch(neigh.reshape(B * M, S))).cpu().numpy()
    v = v.reshape(B, M)
    v[solved_mask] = 0.0
    cost = 1.0 + v
    best_move = cost.argmin(axis=1).astype(np.int64)
    target_val = cost.min(axis=1).astype(np.float32)
    state_solved = (colors == solved).all(axis=1)
    target_val[state_solved] = 0.0
    return target_val, best_move, state_solved


def quick_eval(cube, net, rng, depth, n=50):
    ok, lens = 0, []
    for _ in range(n):
        cube.reset()
        cube.scramble(depth, rng=rng)
        sol = greedy_solve(cube, net, cube.colors.copy(), max_steps=depth * 4 + 10)
        if sol is not None:
            ok += 1
            lens.append(len(sol))
    return ok / n, (np.mean(lens) if lens else float("nan"))


def train(N=2, iters=4000, batch=1000, Kmax=14, lr=1e-3,
          target_update=50, pol_weight=0.3, seed=0, save=None, eval_every=500):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    cube = Cube(N)
    in_dim = cube.n_stickers * 6
    M = len(cube.moveset)
    net = CubeNet(in_dim, M)
    target = CubeNet(in_dim, M)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    save = save or f"model_{N}x{N}.pt"

    history = []
    t0 = time.time()
    print(f"Entrainement {N}x{N} | {iters} iters | batch {batch} | Kmax {Kmax} | "
          f"{sum(p.numel() for p in net.parameters()):,} parametres\n")
    for it in range(1, iters + 1):
        colors, depths = generate_scrambles(cube, batch, Kmax, rng)
        tval, bmove, state_solved = compute_targets(cube, colors, target)

        net.train()
        x = encode_batch(colors)
        v_pred, p_pred = net(x)
        w = torch.from_numpy((1.0 / depths).astype(np.float32))
        loss_v = (w * (v_pred - torch.from_numpy(tval)) ** 2).mean()
        keep = torch.from_numpy(~state_solved)
        if keep.any():
            loss_p = F.cross_entropy(p_pred[keep], torch.from_numpy(bmove)[keep])
        else:
            loss_p = torch.zeros(())
        loss = loss_v + pol_weight * loss_p

        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % target_update == 0:
            target.load_state_dict(net.state_dict())

        if it == 1 or it % eval_every == 0:
            sr, avglen = quick_eval(cube, net, rng, depth=min(7, Kmax), n=40)
            history.append({"iter": it, "loss": float(loss.item()),
                            "loss_v": float(loss_v.item()), "loss_p": float(loss_p.item()),
                            "solve_rate": float(sr),
                            "avg_len": float(avglen) if avglen == avglen else None})
            print(f"iter {it:5d} | loss {loss.item():6.3f} "
                  f"(v {loss_v.item():.3f} / p {loss_p.item():.3f}) | "
                  f"solve@{min(7,Kmax)}: {sr:5.0%}  len {avglen:4.1f} | "
                  f"{time.time()-t0:5.0f}s")

    torch.save({"state_dict": net.state_dict(), "N": N,
                "in_dim": in_dim, "moveset": cube.moveset}, save)
    hist_path = os.path.splitext(save)[0] + "_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nModele sauvegarde : {save}  ({time.time()-t0:.0f}s)")
    print(f"Historique sauvegarde : {hist_path}")
    return save


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=2)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--Kmax", type=int, default=14)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--eval_every", type=int, default=500)
    args = ap.parse_args()
    train(N=args.N, iters=args.iters, batch=args.batch, Kmax=args.Kmax,
          lr=args.lr, save=args.save, eval_every=args.eval_every)
