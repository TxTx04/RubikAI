"""
Moteur de Rubik's Cube générique (2x2 et 3x3).

Idée centrale (robuste et sans table d'adjacence à la main) :
- Chaque autocollant ("sticker") est repéré par sa POSITION 3D et sa NORMALE
  (la direction vers laquelle il "regarde").
- Tourner une face = appliquer une ROTATION 3D de 90° à tous les stickers
  de la couche extérieure correspondante.
- On précalcule, pour chaque mouvement, une PERMUTATION des indices de stickers.
  Appliquer un mouvement devient alors une simple indexation NumPy (très rapide).

Convention des faces (id de face = couleur résolue) :
    0 = R (+X, droite)      1 = L (-X, gauche)
    2 = U (+Y, haut)        3 = D (-Y, bas)
    4 = F (+Z, avant)       5 = B (-Z, arrière)
"""

from __future__ import annotations
import numpy as np

# Normale -> id de face (= couleur à l'état résolu)
NORMALS = {
    (1, 0, 0): 0,   # +X  R
    (-1, 0, 0): 1,  # -X  L
    (0, 1, 0): 2,   # +Y  U
    (0, -1, 0): 3,  # -Y  D
    (0, 0, 1): 4,   # +Z  F
    (0, 0, -1): 5,  # -Z  B
}

# Définition d'une face : (axe, signe du côté, sens de rotation pour le coup direct)
# Sens choisi pour correspondre à la notation standard (sens horaire vu de l'extérieur).
FACE_DEF = {
    "R": (0, +1, -1),
    "L": (0, -1, +1),
    "U": (1, +1, -1),
    "D": (1, -1, +1),
    "F": (2, +1, -1),
    "B": (2, -1, +1),
}

# Jeux de coups (quarts de tour uniquement = QTM)
MOVESET_3 = ["U", "U'", "D", "D'", "L", "L'", "R", "R'", "F", "F'", "B", "B'"]
# Pour le 2x2 on fixe le coin arrière-bas-gauche (BDL) : on n'utilise que R, U, F.
# Cela supprime la redondance des 24 orientations du cube entier.
MOVESET_2 = ["R", "R'", "U", "U'", "F", "F'"]


def _rot_vec(v, ax, d):
    """Rotation de 90° du vecteur v autour de l'axe ax (0=X,1=Y,2=Z), sens d=+1/-1."""
    x, y, z = v
    if ax == 0:
        return (x, -z, y) if d > 0 else (x, z, -y)
    if ax == 1:
        return (z, y, -x) if d > 0 else (-z, y, x)
    return (-y, x, z) if d > 0 else (y, -x, z)


def _build_geometry(N):
    """Construit la liste des stickers (position, normale) et l'index inverse."""
    vals = [2 * c - (N - 1) for c in range(N)]  # coords centrées entières
    slots = []
    for normal in NORMALS:                      # itère dans l'ordre d'insertion
        ax = next(i for i in range(3) if normal[i] != 0)
        sign = normal[ax]
        free = [i for i in range(3) if i != ax]
        for a in vals:
            for b in vals:
                pos = [0, 0, 0]
                pos[ax] = sign * (N - 1)
                pos[free[0]] = a
                pos[free[1]] = b
                slots.append((tuple(pos), normal))
    index = {slot: i for i, slot in enumerate(slots)}
    return slots, index


def _build_perm(N, slots, index, ax, side, d):
    """Permutation P telle que: nouvelles_couleurs = anciennes_couleurs[P]."""
    P = np.arange(len(slots), dtype=np.int64)
    layer = side * (N - 1)
    for s, (pos, normal) in enumerate(slots):
        if pos[ax] == layer:                    # sticker dans la couche tournée
            npos = _rot_vec(pos, ax, d)
            nnormal = _rot_vec(normal, ax, d)
            t = index[(npos, nnormal)]          # destination du sticker
            P[t] = s                            # la couleur de s arrive en t
    return P


class Cube:
    """Représente l'état d'un cube NxN par un tableau de couleurs (un int par sticker)."""

    def __init__(self, N=3, moveset=None):
        self.N = N
        self.moveset = moveset or (MOVESET_2 if N == 2 else MOVESET_3)
        self.slots, self.index = _build_geometry(N)
        self.n_stickers = len(self.slots)       # 6 * N*N
        # Couleur résolue de chaque sticker = id de face de sa normale
        self.solved = np.array([NORMALS[nrm] for (_, nrm) in self.slots], dtype=np.int8)
        # Précalcul des permutations pour tous les coups du jeu
        self.perms = {}
        for name in self.moveset:
            base = name[0]
            prime = name.endswith("'")
            ax, side, d = FACE_DEF[base]
            if prime:
                d = -d
            self.perms[name] = _build_perm(N, self.slots, self.index, ax, side, d)
        self.colors = self.solved.copy()

    # --- état ---------------------------------------------------------------
    def reset(self):
        self.colors = self.solved.copy()
        return self

    def copy(self):
        c = Cube.__new__(Cube)
        c.__dict__.update(self.__dict__)        # partage géométrie/perms (immuables)
        c.colors = self.colors.copy()
        return c

    def is_solved(self, colors=None):
        c = self.colors if colors is None else colors
        faces = c.reshape(6, self.N * self.N)
        return bool(np.all(faces == faces[:, :1]))

    # --- coups --------------------------------------------------------------
    def apply(self, name):
        self.colors = self.colors[self.perms[name]]
        return self

    def apply_seq(self, seq):
        for m in seq:
            self.apply(m)
        return self

    @staticmethod
    def inverse_move(name):
        return name[:-1] if name.endswith("'") else name + "'"

    def scramble(self, k, rng=None, avoid_trivial=True):
        """Mélange en k coups; renvoie la séquence appliquée."""
        rng = rng or np.random.default_rng()
        seq = []
        last = None
        for _ in range(k):
            m = self.moveset[rng.integers(len(self.moveset))]
            # évite d'annuler immédiatement le coup précédent (A puis A')
            while avoid_trivial and last is not None and m == self.inverse_move(last):
                m = self.moveset[rng.integers(len(self.moveset))]
            self.apply(m)
            seq.append(m)
            last = m
        return seq

    # --- encodage pour le réseau de neurones --------------------------------
    def one_hot(self, colors=None):
        """Encodage one-hot des couleurs -> vecteur de taille n_stickers*6."""
        c = self.colors if colors is None else colors
        oh = np.zeros((self.n_stickers, 6), dtype=np.float32)
        oh[np.arange(self.n_stickers), c] = 1.0
        return oh.reshape(-1)

    def key(self, colors=None):
        """Clé hashable (bytes) de l'état, pour BFS / tables."""
        c = self.colors if colors is None else colors
        return c.tobytes()
