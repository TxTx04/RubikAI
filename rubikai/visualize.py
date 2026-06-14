"""
Génère les images d'analyse du projet RubikAI (sauvegardées dans results/).

Images produites :
  1. cube_filmstrip.png    : l'IA résout un cube, état par état (pédagogique + visuel)
  2. value_vs_distance.png : la valeur APPRISE vs la distance RÉELLE (le réseau a-t-il
                             vraiment appris une notion de "distance à la solution" ?)
  3. ia_vs_optimal.png     : longueur des solutions de l'IA vs l'optimum exact
  4. training_curve.png    : courbes d'apprentissage (loss + taux de résolution)
"""
from __future__ import annotations
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cube import Cube
from model import encode_batch
from solve import load_model, astar_solve
from solver_bfs import solve_optimal

# Couleurs standard d'un Rubik's Cube, indexées par id de face (0=R..5=B)
FACE_RGB = np.array([
    [0.85, 0.12, 0.12],   # 0 R rouge
    [0.95, 0.50, 0.10],   # 1 L orange
    [0.96, 0.96, 0.96],   # 2 U blanc
    [0.96, 0.85, 0.10],   # 3 D jaune
    [0.10, 0.70, 0.25],   # 4 F vert
    [0.10, 0.35, 0.80],   # 5 B bleu
])
BG = np.array([0.12, 0.12, 0.15])
# position (ligne, colonne) de chaque face dans le patron déplié
NET_POS = {2: (0, 1), 1: (1, 0), 4: (1, 1), 0: (1, 2), 5: (1, 3), 3: (2, 1)}


def _net_image(cube: Cube, colors: np.ndarray) -> np.ndarray:
    N = cube.N
    H, W = 3 * N, 4 * N
    img = np.tile(BG, (H, W, 1))
    for f, (r, c) in NET_POS.items():
        sub = colors[f * N * N:(f + 1) * N * N].reshape(N, N)
        img[r * N:(r + 1) * N, c * N:(c + 1) * N] = FACE_RGB[sub]
    return img


def _draw_net(ax, cube, colors, title=""):
    N = cube.N
    ax.imshow(_net_image(cube, colors), interpolation="nearest")
    # quadrillage
    for r, c in NET_POS.values():
        for i in range(N + 1):
            ax.plot([c * N - 0.5, c * N - 0.5 + N], [r * N - 0.5 + i, r * N - 0.5 + i],
                    color="black", lw=1.2)
            ax.plot([c * N - 0.5 + i, c * N - 0.5 + i], [r * N - 0.5, r * N - 0.5 + N],
                    color="black", lw=1.2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)


def cube_filmstrip(cube, net, out, depth=9, seed=2):
    rng = np.random.default_rng(seed)
    cube.reset()
    cube.scramble(depth, rng=rng)
    start = cube.colors.copy()
    sol = astar_solve(cube, net, start, weight=1.2) or []
    # reconstruit les états successifs
    states = [start.copy()]
    cur = start.copy()
    for m in sol:
        cur = cur[cube.perms[m]]
        states.append(cur.copy())
    n = len(states)
    fig, axes = plt.subplots(1, n, figsize=(2.1 * n, 2.6))
    if n == 1:
        axes = [axes]
    for i, (ax, st) in enumerate(zip(axes, states)):
        if i == 0:
            t = "MELANGE"
        elif i == n - 1:
            t = "RESOLU"
        else:
            t = f"coup {i}: {sol[i-1]}"
        _draw_net(ax, cube, st, t)
    fig.suptitle(f"RubikAI resout un cube {cube.N}x{cube.N} en {len(sol)} coups (recherche A* guidee par le reseau)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return len(sol)


def value_vs_distance(cube, net, out, n=400, seed=5):
    rng = np.random.default_rng(seed)
    true_d, learned_v = [], []
    for _ in range(n):
        cube.reset()
        d = int(rng.integers(1, 15))
        cube.scramble(d, rng=rng)
        s = cube.colors.copy()
        opt = solve_optimal(cube, s)
        if opt is None:
            continue
        true_d.append(len(opt))
        learned_v.append(float(net.value(encode_batch(s)).item()))
    true_d = np.array(true_d); learned_v = np.array(learned_v)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(true_d + rng.normal(0, 0.06, len(true_d)), learned_v,
               s=14, alpha=0.35, color="#2a6fdb", label="etats")
    xs = np.arange(true_d.min(), true_d.max() + 1)
    means = [learned_v[true_d == x].mean() for x in xs]
    ax.plot(xs, means, "o-", color="#d6336c", lw=2, label="moyenne apprise")
    ax.plot(xs, xs, "--", color="gray", label="ideal (y = x)")
    corr = np.corrcoef(true_d, learned_v)[0, 1]
    ax.set_xlabel("Distance REELLE a la solution (BFS optimal)")
    ax.set_ylabel("Valeur APPRISE par le reseau")
    ax.set_title(f"Le reseau a-t-il appris la 'distance' ?  (correlation = {corr:.3f})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)
    return corr


def ia_vs_optimal(cube, net, out, n=60, depth=13, seed=9):
    rng = np.random.default_rng(seed)
    gaps, solved = [], 0
    for _ in range(n):
        cube.reset()
        cube.scramble(depth, rng=rng)
        s = cube.colors.copy()
        sol = astar_solve(cube, net, s, weight=1.2)
        opt = solve_optimal(cube, s)
        if sol is None:
            continue
        chk = s.copy()
        for m in sol:
            chk = chk[cube.perms[m]]
        if cube.is_solved(chk):
            solved += 1
            gaps.append(len(sol) - len(opt))
    gaps = np.array(gaps)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    vals, counts = np.unique(gaps, return_counts=True)
    ax.bar(vals, counts, width=0.6, color="#2a9d8f", edgecolor="black")
    for v, c in zip(vals, counts):                       # étiquette de comptage
        ax.text(v, c + 0.5, str(int(c)), ha="center", fontweight="bold")
    pct_opt = 100.0 * np.mean(gaps == 0) if len(gaps) else 0
    xmax = max(2, int(vals.max()) if len(vals) else 2)
    ax.set_xlim(-0.8, xmax + 0.8)
    ax.set_xticks(range(0, xmax + 1))
    ax.set_ylim(0, counts.max() * 1.12 if len(counts) else 1)
    ax.set_xlabel("Ecart a l'optimal (coups en trop)  —  0 = solution optimale")
    ax.set_ylabel("Nombre de cubes")
    ax.set_title(f"Qualite IA sur {n} melanges de {depth} coups\n"
                 f"resolus: {solved}/{n} ({100*solved/n:.0f}%)  |  "
                 f"optimaux: {pct_opt:.0f}%")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)
    return solved, n, pct_opt


def training_curve(history_path, out):
    with open(history_path) as f:
        h = json.load(f)
    its = [d["iter"] for d in h]
    loss = [d["loss"] for d in h]
    sr = [100 * d["solve_rate"] for d in h]
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(its, loss, "o-", color="#e76f51", label="perte (loss)")
    ax1.set_xlabel("Iteration d'entrainement")
    ax1.set_ylabel("Perte", color="#e76f51")
    ax1.tick_params(axis="y", labelcolor="#e76f51")
    ax2 = ax1.twinx()
    ax2.plot(its, sr, "s-", color="#264653", label="taux de resolution")
    ax2.set_ylabel("Taux de resolution glouton (%)", color="#264653")
    ax2.tick_params(axis="y", labelcolor="#264653")
    ax2.set_ylim(-5, 105)
    ax1.set_title("Apprentissage : l'IA s'ameliore au fil des iterations")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_2x2.pt")
    ap.add_argument("--history", default=None)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cube, net = load_model(args.model)
    o = args.outdir

    print("1/4 cube_filmstrip ...")
    L = cube_filmstrip(cube, net, os.path.join(o, "cube_filmstrip.png"))
    print(f"     -> resolu en {L} coups")
    print("2/4 value_vs_distance ...")
    c = value_vs_distance(cube, net, os.path.join(o, "value_vs_distance.png"))
    print(f"     -> correlation = {c:.3f}")
    print("3/4 ia_vs_optimal ...")
    s, n, p = ia_vs_optimal(cube, net, os.path.join(o, "ia_vs_optimal.png"))
    print(f"     -> {s}/{n} resolus, {p:.0f}% optimaux")
    if args.history and os.path.exists(args.history):
        print("4/4 training_curve ...")
        training_curve(args.history, os.path.join(o, "training_curve.png"))
    else:
        print("4/4 training_curve ... (ignore : pas d'historique fourni)")
    print(f"\nImages sauvegardees dans : {os.path.abspath(o)}")


if __name__ == "__main__":
    main()
