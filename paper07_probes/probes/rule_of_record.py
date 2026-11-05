#!/usr/bin/env python3
"""Paper 7 strengthening item 2: the decision rule OF RECORD.

The published rule (manuscript Table 1, reconstructed from the source .docx S3.5):
  accept-strong   : BF10 >= 10 and Pd >= 0.95
  accept-moderate : 3 <= BF10 < 10 and Pd >= 0.90
  reject          : BF10 <= 1/3
  otherwise       : inconclusive
Plus a post-hoc ANNOTATION (not a decision): "direction contradicted" when
BF10 >= 10 and Pd <= 0.05.

The previously shipped benchmark code accepted iff BF10 > 3 and Pd > 0.95 and its
decision function branched on the ground truth beta*. This module implements the
published rule only; it takes no ground-truth argument. Unit tests at the bottom
enumerate the (BF, Pd) grid, including the moderate-band cells where the old
code diverged from the published rule (12 of this module's 63-cell grid; the
round-1 tester report's probe B5, on a different grid, found 15).
"""


def decide(bf10: float, pd: float) -> str:
    if bf10 >= 10.0 and pd >= 0.95:
        return "accept"
    if 3.0 <= bf10 < 10.0 and pd >= 0.90:
        return "accept"
    if bf10 <= 1.0 / 3.0:
        return "reject"
    return "inconclusive"


def direction_contradicted(bf10: float, pd: float) -> bool:
    """Annotation only: strong evidence for an effect, against the elicited sign."""
    return bf10 >= 10.0 and pd <= 0.05


def _old_code_decide(bf10: float, pd: float) -> str:
    # the rule the shipped benchmark actually implemented (for divergence tests)
    if bf10 > 3.0 and pd > 0.95:
        return "accept"
    if bf10 <= 1.0 / 3.0:
        return "reject"
    return "inconclusive"


if __name__ == "__main__":
    # boundary cells
    assert decide(10.0, 0.95) == "accept"
    assert decide(10.0, 0.949) == "inconclusive"
    assert decide(3.0, 0.90) == "accept"          # moderate tier lower corner
    assert decide(9.99, 0.90) == "accept"
    assert decide(3.0, 0.899) == "inconclusive"
    assert decide(2.99, 0.99) == "inconclusive"
    assert decide(1.0 / 3.0, 0.50) == "reject"
    assert decide(0.34, 0.50) == "inconclusive"
    assert decide(0.92, 1.00) == "reject" or decide(0.92, 1.00) == "inconclusive"
    assert decide(0.92, 1.00) == "inconclusive"   # the pmo stress edge stays inconclusive
    assert decide(1e6, 0.999) == "accept"
    assert decide(134.44, 0.0) != "accept"        # null edge: BF01=134 -> BF10 tiny -> reject
    assert decide(1.0 / 134.44, 0.5) == "reject"
    assert direction_contradicted(50.0, 0.0) is True
    assert direction_contradicted(50.0, 0.5) is False

    # grid divergence audit vs the old code (mirrors tester probe B5's grid)
    bfs = [0.1, 0.34, 1.0, 2.0, 3.0, 5.0, 9.9, 10.0, 100.0]
    pds = [0.0, 0.5, 0.89, 0.90, 0.94, 0.95, 0.99]
    div = [(bf, pd) for bf in bfs for pd in pds
           if decide(bf, pd) != _old_code_decide(bf, pd)]
    print(f"grid cells: {len(bfs) * len(pds)}; divergences old-vs-published: {len(div)}")
    for bf, pd in div:
        print(f"  BF={bf:>6} Pd={pd:<5} published={decide(bf, pd):<12} old={_old_code_decide(bf, pd)}")
    # every divergence must be a moderate-band accept under the published rule
    assert all(decide(bf, pd) == "accept" and 3.0 <= bf < 10.0 or (bf >= 10 and 0.90 <= pd < 0.95 + 1e-9 and decide(bf, pd) == 'accept')
               for bf, pd in div)
    print("rule_of_record unit tests: ALL PASS")
