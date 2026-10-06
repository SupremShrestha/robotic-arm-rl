"""Seeded position pairs in opposite parts of the supported table workspace."""
import numpy as np


def sample_cross_workspace(rng):
    left=rng.uniform([.35,-.23],[.40,-.14])
    right=rng.uniform([.50,.16],[.55,.26])
    if rng.random()<.5:
        left=rng.uniform([.50,-.23],[.55,-.14])
        right=rng.uniform([.35,.16],[.40,.26])
    return {"source_xy":left.tolist(),"destination_xy":right.tolist()}


THREE_TASKS = [
    ((.45,-.10),(.45,.13)),
    ((.51,-.18),(.39,.20157568056677826)),
    ((.30,-.30),(.55,.30)),
]
