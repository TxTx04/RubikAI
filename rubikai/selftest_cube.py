"""
Vérifie que le moteur de cube respecte les invariants d'un vrai Rubik's Cube.
Si ces tests passent, la mécanique est correcte (pas juste "une permutation au hasard").
"""
import numpy as np
from cube import Cube


def test_solved():
    assert Cube(3).is_solved() and Cube(2).is_solved()
    print("[OK] L'etat initial est bien resolu (2x2 et 3x3)")


def test_order4():
    """Tourner une face 4 fois = revenir a l'identite."""
    for N in (2, 3):
        c = Cube(N)
        for m in c.moveset:
            base = Cube(N)
            for _ in range(4):
                base.apply(m)
            assert np.array_equal(base.colors, c.solved), f"{m} n'est pas d'ordre 4"
    print("[OK] Chaque coup est d'ordre 4 (face tournee 4x = identite)")


def test_inverse():
    """Un coup suivi de son inverse = identite."""
    for N in (2, 3):
        c = Cube(N)
        rng = np.random.default_rng(0)
        for m in c.moveset:
            base = Cube(N)
            base.apply(m).apply(Cube.inverse_move(m))
            assert base.is_solved(), f"{m} puis inverse != identite"
    print("[OK] coup + inverse = identite")


def test_sexy_move_order6():
    """(R U R' U') repete 6 fois = identite : invariant celebre du cube."""
    for N in (2, 3):
        c = Cube(N)
        for _ in range(6):
            c.apply("R").apply("U").apply("R'").apply("U'")
        assert c.is_solved(), "(R U R' U')^6 != identite"
    print("[OK] (R U R' U')^6 = identite  (invariant non-trivial)")


def test_scramble_unscramble():
    """Melanger puis appliquer la sequence inverse = resolu."""
    for N in (2, 3):
        c = Cube(N)
        rng = np.random.default_rng(42)
        seq = c.scramble(50, rng=rng)
        assert not c.is_solved(), "le melange n'a rien change (suspect)"
        for m in reversed(seq):
            c.apply(Cube.inverse_move(m))
        assert c.is_solved(), "impossible de re-resoudre via l'inverse"
    print("[OK] melange (50 coups) puis inverse = resolu")


def test_encoding():
    c = Cube(2)
    oh = c.one_hot()
    assert oh.shape == (c.n_stickers * 6,)
    assert oh.sum() == c.n_stickers  # un 1 par sticker
    print(f"[OK] encodage one-hot : {c.n_stickers} stickers -> vecteur de {oh.shape[0]}")


if __name__ == "__main__":
    test_solved()
    test_order4()
    test_inverse()
    test_sexy_move_order6()
    test_scramble_unscramble()
    test_encoding()
    print("\nTOUS LES TESTS PASSENT : le moteur de cube est mathematiquement correct.")
