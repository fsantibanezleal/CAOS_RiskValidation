"""The packaged labels on the Contraste 1.0 corpus (Contraste backlog BL-096, engine 1.2.0).

Every test row of the Contraste case documents (``data/derived/C*/*.json``, key ``tests``) whose ``policy_version``
names a packaged policy resolves its thresholds through the packaged entry (``Policy.threshold`` and
``Threshold.for_row``) and reproduces the light the document committed (``Policy.classify_row``). A row whose
``policy_version`` is an inline battery string (``contraste-c12-1.0.0``, the hyphen form, not a packaged label) is
excluded and counted apart. The documents are read-only; the tests are skipped when ``CONTRASTE_ROOT`` is unset, as
those of ``tests/test_packaged_policies.py``.

The counts are asserted, so a corpus change (a re-bake, a new case) is seen here and not absorbed: update them with
the corpus.
"""

from __future__ import annotations

import copy
import glob
import json
import math
import os
from collections import Counter
from pathlib import Path

import pytest

from riskvalidation.core.policy import packaged_labels, packaged_policy

ROOT = os.environ.get("CONTRASTE_ROOT")
needs_root = pytest.mark.skipif(not ROOT, reason="CONTRASTE_ROOT is unset")

ROWS = 6330  # every test row of the corpus
PACKAGED_ROWS = 6295  # those whose policy_version is a packaged label
INLINE_ROWS = 35  # contraste-c12-1.0.0: the inline hyphen form, no packaged policy of that spelling
WITH_ENTRY = 5605  # packaged rows whose test has a threshold entry in its policy
RULE_ONLY_ROWS = 690  # packaged rows of tests that state no threshold by contract: the rule classifies them


def _documents():
    for path in sorted(glob.glob(str(Path(ROOT) / "data" / "derived" / "C*" / "*.json"))):
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        if isinstance(doc, dict) and "tests" in doc:
            rel = Path(path).relative_to(Path(ROOT) / "data" / "derived").as_posix()
            for row in doc["tests"]:
                yield rel, row


def _view(row: dict) -> dict:
    """The row as the policy reads it. Two fields the committed documents state as null or leave out are derived from
    the document: the Euler basis from the segment (``var-*`` is the kernel VaR, everything else an exact allocation)
    and the observed rates of a lifetime-PD back-test from the stored Brier score and signed gap.
    """
    view = copy.deepcopy(row)
    extras = view.get("extras")
    if not isinstance(extras, dict):
        extras = view["extras"] = {}
    if row["test_id"] == "portfolio.euler_additivity" and not extras.get("basis"):
        extras["basis"] = "kernel-var" if str(row.get("segment") or "").startswith("var-") else "exact-allocation"
    if row["test_id"] == "ifrs9.lifetime_pd_backtest" and "band_low" in extras and not extras.get("observed"):
        obs = []
        for brier, gap in zip(extras["brier"], extras["signed_gap"], strict=True):
            disc = (1 - 2 * gap) ** 2 - 4 * (brier - gap)
            p = ((1 - 2 * gap) - math.sqrt(max(disc, 0.0))) / 2.0
            o = p + gap
            obs.append(0.0 if abs(o) < 1e-7 else o)  # a zero rate carries the rounding of the stored 9 digits
        extras["observed"] = obs
    return view


def _split(rows):
    """``(packaged, inline)``: the rows whose policy_version is a packaged label, and the others."""
    labels = set(packaged_labels())
    packaged = [(rel, row) for rel, row in rows if row["policy_version"] in labels]
    inline = [(rel, row) for rel, row in rows if row["policy_version"] not in labels]
    return packaged, inline


def _reproduce(packaged):
    """``(checked, failures, with_entry, unresolved)`` over the packaged rows: each row is classified by its packaged
    policy and compared with its committed light; ``with_entry`` counts the rows whose test has an entry, and
    ``unresolved`` those whose entry gives no effective threshold for the row.
    """
    checked, failures, with_entry, unresolved = 0, [], 0, []
    for rel, row in packaged:
        policy = packaged_policy(row["policy_version"])
        view = _view(row)
        entry = policy.threshold(row["test_id"])
        if entry is not None:
            with_entry += 1
            if entry.for_row(view) is None:
                unresolved.append((rel, row["test_id"], row.get("segment")))
        light, _ = policy.classify_row(row["test_id"], view)
        checked += 1
        if light.value != row["light"]:
            failures.append((rel, row["test_id"], row.get("segment"), row["light"], light.value))
    return checked, failures, with_entry, unresolved


@needs_root
def test_the_corpus_splits_into_packaged_and_inline_rows_with_the_stated_counts():
    rows = list(_documents())
    packaged, inline = _split(rows)
    assert len(rows) == ROWS
    assert len(packaged) == PACKAGED_ROWS and len(inline) == INLINE_ROWS
    # the inline rows are excluded, never silently dropped: each is an inline battery string, not a packaged label
    assert {row["policy_version"] for _, row in inline} == {"contraste-c12-1.0.0"}
    assert Counter(row["policy_version"] for _, row in packaged).keys() <= set(packaged_labels())


@needs_root
def test_every_packaged_row_reproduces_its_committed_light_through_its_packaged_entry():
    packaged, _ = _split(list(_documents()))
    checked, failures, with_entry, unresolved = _reproduce(packaged)
    print(f"packaged rows checked: {checked}; with an entry: {with_entry}")
    assert checked == PACKAGED_ROWS  # a test that checked no row would pass: the count is the proof it did not
    assert not failures, f"{len(failures)} rows not reproduced, first: {failures[:6]}"
    assert with_entry == WITH_ENTRY and checked - with_entry == RULE_ONLY_ROWS
    assert not unresolved, f"entries that give no threshold for the row: {unresolved[:6]}"


@needs_root
def test_every_packaged_label_of_the_corpus_is_used_and_every_light_is_met():
    packaged, _ = _split(list(_documents()))
    labels = Counter(row["policy_version"] for _, row in packaged)
    # the corpus writes C12's battery in its inline hyphen form, so that one packaged label is not met by any row
    assert set(labels) == set(packaged_labels()) - {"contraste-c12 1.0.0"}, labels
    assert len(labels) == 11 and min(labels.values()) > 0
    lights = {row["light"] for _, row in packaged}
    assert {"green", "amber", "red"} <= lights, lights


@needs_root
def test_the_check_detects_a_light_that_the_policy_does_not_give():
    """Must refute: a corrupted committed light is reported, so a passing run is not a check that saw nothing."""
    packaged, _ = _split(list(_documents()))
    sample = [(rel, copy.deepcopy(row)) for rel, row in packaged if row["test_id"] == "market.desk_backtest"][:5]
    assert len(sample) == 5
    flip = {"green": "red", "amber": "green", "red": "green", "not_evaluated": "green", "error": "green"}
    for _, row in sample:
        row["light"] = flip[row["light"]]
    checked, failures, _, _ = _reproduce(sample)
    assert checked == 5 and len(failures) == 5


def test_an_empty_selection_checks_nothing_and_says_so():
    """The helper counts what it checked: an empty corpus gives zero, which the asserted counts above reject."""
    assert _reproduce([]) == (0, [], 0, [])
    assert _split([]) == ([], [])
