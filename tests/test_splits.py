"""Test the per-n dataset-split allocation used by the shared-model sweep."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "experiments"))

from multi_prover import per_n_counts  # noqa: E402

NS = [4, 5, 6, 7, 8, 9]


def test_equal_split_is_uniform():
    c = per_n_counts(NS, 60000, "equal")
    assert len(set(c.values())) == 1
    assert sum(c.values()) == 60000


def test_linear_split_monotone_in_n():
    c = per_n_counts(NS, 48000, "linear")
    vals = [c[n] for n in NS]
    assert vals == sorted(vals)                 # grows with n
    assert c[9] > c[4]


def test_factorial_split_favors_large_n_with_floor():
    c = per_n_counts(NS, 48000, "factorial")
    assert c[9] == max(c.values())
    assert min(c.values()) >= 200               # small n floored, not starved to 0


def test_paper_split_matches_ratios_up_to_floor():
    c = per_n_counts(NS, 48000, "paper")
    # n=8, n=9 dominate (paper Table-4 sizes 10000, 20000)
    assert c[9] > c[8] > c[7]
    assert all(v >= 200 for v in c.values())
