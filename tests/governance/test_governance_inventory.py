"""Requirements RV-1550 to RV-1581 (docs/design/features/u15-governance/requirements.md): the inventory, its
dependency graph and its completeness (contract `engine-governance-api` 1.0.0, module inventory)."""

from __future__ import annotations

import dataclasses

import pytest

from riskvalidation.governance.inventory import (
    Component,
    Edge,
    Entry,
    Inventory,
    assign_variant,
    attribute_completeness,
    build_inventory,
    graph_metrics,
    topological_order,
)

from .governance_support import ATTR_IDS, attributes, edges_of, entries_of, make_entry


def _keyed(*cases: str) -> list[Entry]:
    """Entries keyed `<case>/f`, one component each."""
    return [make_entry(c, "f", components=(Component(c, "f", "P0", "champion"),)) for c in cases]


# ---- Component and Entry


def test_component_key_is_the_triple_and_the_role_is_one_of_four():
    c = Component("C01", "german", "P0", "champion")
    assert c.key == "C01/german/P0"
    for role in ("champion", "challenger", "benchmark", "reference"):
        Component("C01", "german", "P0", role)
    with pytest.raises(ValueError):
        Component("C01", "german", "P0", "approved")
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.role = "benchmark"  # type: ignore[misc]


def test_entry_key_and_valid_entry():
    e = make_entry()
    assert e.key == "C01/german"
    assert len(e.attributes) == 40


def test_entry_names_the_attribute_that_breaks_a_rule():
    bad = {
        "A05": {"value": 1, "status": "guessed", "basis": "x" * 10},  # an unknown status
        "A06": {"value": 1, "status": "present", "basis": "read somewhere"},  # present without a pointer
        "A07": {"value": None, "status": "not-applicable", "basis": "n/a"},  # a reason shorter than eight characters
        "A08": {"value": 3, "status": "missing", "basis": "no field"},  # missing with a value
        "A09": {"value": 3, "status": "not-applicable", "basis": "does not apply here"},  # not-applicable with a value
        "A10": {"value": 3, "status": "derivable", "basis": " "},  # derivable without a basis
        "A11": {"value": 3, "status": "convention", "basis": ""},  # convention without a basis
    }
    for aid, attr in bad.items():
        with pytest.raises(ValueError, match=aid):
            make_entry(attrs=attributes(**{aid: attr}))


def test_entry_accepts_a_present_attribute_with_a_pointer_and_absent_ones_with_none():
    ok = attributes(
        A01={
            "value": "C01/german",
            "status": "present",
            "basis": "read at data/derived/C01/models-german.json#/case_id",
        },
        A02={"value": None, "status": "missing", "basis": "no field"},
        A03={"value": None, "status": "not-applicable", "basis": "a reason of enough characters"},
    )
    assert make_entry(attrs=ok).attributes["A02"]["value"] is None


def test_entry_approver_and_approval_date_are_not_applicable_with_no_value():
    for aid in ("A32", "A33"):
        for attr in (
            {"value": "CRO", "status": "not-applicable", "basis": "Contraste approves no model"},
            {"value": None, "status": "missing", "basis": "no field"},
            {"value": None, "status": "convention", "basis": "a basis"},
        ):
            with pytest.raises(ValueError, match=aid):
                make_entry(attrs=attributes(**{aid: attr}))


def test_entry_needs_the_forty_attributes_and_an_attribute_has_a_value_status_and_basis():
    attrs = attributes()
    del attrs["A40"]
    with pytest.raises(ValueError, match="A40"):
        make_entry(attrs=attrs)
    with pytest.raises(ValueError, match="A12"):
        make_entry(attrs=attributes(A12={"value": 1, "status": "convention"}))


def test_entry_variants_are_listed_with_a_basis():
    e = make_entry(
        variants=(("v1", "the single models document of the case"), ("v2", "the single models document of the case"))
    )
    assert [v for v, _ in e.variants] == ["v1", "v2"]
    with pytest.raises(ValueError):
        make_entry(variants=(("v1", "a basis"), ("v1", "a basis")))
    with pytest.raises(ValueError):
        make_entry(variants=(("v1", " "),))


def test_repeated_component_key_inside_one_entry_is_refused():
    comps = (Component("A", "f", "P0", "champion"), Component("A", "f", "P0", "challenger"))
    with pytest.raises(ValueError, match="repeats"):
        build_inventory([make_entry("A", "f", components=comps)], [])


# ---- Edge, Inventory, build_inventory


def test_edge_needs_a_stated_source_and_one_of_two_kinds():
    Edge("A", "B", "document", "file.py:3")
    Edge("A", "B", "code", "file.py:3")
    with pytest.raises(ValueError):
        Edge("A", "B", "document", "")
    with pytest.raises(ValueError):
        Edge("A", "B", "related", "file.py:3")


def test_build_inventory_keeps_the_order_and_is_frozen():
    ents = _keyed("B", "A", "C")
    edges = [Edge("B/f", "A/f", "document", "s1"), Edge("A/f", "C/f", "code", "s2")]
    inv = build_inventory(ents, edges)
    assert isinstance(inv, Inventory)
    assert [e.key for e in inv.entries] == ["B/f", "A/f", "C/f"]
    assert [(e.upstream, e.downstream) for e in inv.edges] == [("B/f", "A/f"), ("A/f", "C/f")]
    with pytest.raises(dataclasses.FrozenInstanceError):
        inv.edges = ()  # type: ignore[misc]


def test_cycle_is_refused_naming_the_cycle_in_dependency_order():
    ents = _keyed("A", "B", "C")
    edges = [
        Edge("A/f", "B/f", "document", "s"),
        Edge("B/f", "C/f", "document", "s"),
        Edge("C/f", "A/f", "document", "s"),
    ]
    with pytest.raises(ValueError) as exc:
        build_inventory(ents, edges)
    assert str(exc.value) == "dependency cycle: A/f, B/f, C/f, A/f"


def test_cycle_goldens_on_plain_keys_through_topological_order():
    with pytest.raises(ValueError) as exc:
        topological_order(["A", "B", "C"], [("A", "B"), ("B", "C"), ("C", "A")])
    assert str(exc.value) == "dependency cycle: A, B, C, A"
    with pytest.raises(ValueError) as exc:
        topological_order(["C", "B", "A"], [("C", "A"), ("A", "B"), ("B", "C")])
    assert str(exc.value) == "dependency cycle: A, B, C, A"  # from its smallest key


def test_cycle_downstream_nodes_are_not_part_of_the_named_cycle():
    with pytest.raises(ValueError) as exc:
        topological_order(["A", "B", "C", "D"], [("B", "C"), ("C", "B"), ("C", "D"), ("A", "B")])
    assert str(exc.value) == "dependency cycle: B, C, B"


def test_unknown_entries_in_edges_and_related_links_are_refused():
    ents = _keyed("A")
    with pytest.raises(ValueError) as exc:
        build_inventory(ents, [Edge("A/f", "Z", "document", "s")])
    assert str(exc.value) == "edge A/f to Z: unknown entry Z"
    with pytest.raises(ValueError) as exc:
        build_inventory(ents, [], related=[("A/f", "Z")])
    assert str(exc.value) == "related link A/f and Z: unknown entry Z"
    build_inventory(ents, [], related=[("A/f", "A/f")])


def test_repeated_component_key_across_entries_is_refused():
    a = make_entry("C01", "german", components=(Component("C01", "german", "P0", "champion"),))
    b = make_entry("C01", "german2", components=(Component("C01", "german", "P0", "benchmark"),))
    with pytest.raises(ValueError) as exc:
        build_inventory([a, b], [])
    assert str(exc.value) == "component key 'C01/german/P0' repeats"


def test_inventory_invariants():
    ents = _keyed("A", "B")
    with pytest.raises(ValueError):
        build_inventory(ents + ents[:1], [])  # a repeated entry key
    with pytest.raises(ValueError):
        build_inventory(
            ents, [Edge("A/f", "B/f", "document", "s"), Edge("A/f", "B/f", "document", "t")]
        )  # a repeated edge
    build_inventory(
        ents, [Edge("A/f", "B/f", "document", "s"), Edge("A/f", "B/f", "code", "t")]
    )  # another kind is another edge
    with pytest.raises(ValueError, match="the inventory holds no entry"):
        build_inventory([], [])
    with pytest.raises(ValueError, match="itself"):
        build_inventory(ents, [Edge("A/f", "A/f", "document", "s")])  # a self edge
    with pytest.raises(ValueError):
        build_inventory(ents, [Edge("riskvalidation", "A/f", "document", "s")])  # the engine node is not an entry


# ---- assign_variant


def test_assign_variant_goldens():
    fits = {"german": ["uci-german"], "taiwan": ["uci-taiwan"]}
    assert assign_variant(["uci-german"], fits) == (
        "german",
        "the variant's provenance reads uci-german, the sources of the fit german",
    )
    fit, reason = assign_variant([], fits)
    assert fit is None and reason
    assert assign_variant(["x"], {"only": ["y"]}) == ("only", "the single models document of the case")


def test_assign_variant_rules():
    fits = {"a": ["s1", "s2"], "b": ["s2", "s3"]}
    assert assign_variant(["s1"], fits)[0] == "a"
    assert assign_variant(["s1", "s2"], fits)[0] == "a"
    fit, reason = assign_variant(["s2"], fits)  # both fits contain it: never assigned by guess
    assert fit is None and "s2" in reason
    fit, reason = assign_variant(["s9"], fits)
    assert fit is None and reason
    assert assign_variant(["s2", "s3"], fits)[0] == "b"
    assert assign_variant(["s1"], fits) == assign_variant(["s1"], fits)  # deterministic


def test_assign_variant_gives_a_variant_to_one_fit_only():
    fits = {"a": ["s1"], "b": ["s2"], "c": ["s1", "s2"]}
    for sources in (["s1"], ["s2"], ["s1", "s2"]):
        fit, _ = assign_variant(sources, fits)
        holders = [f for f, src in fits.items() if all(s in src for s in sources)]
        assert (fit in holders) if len(holders) == 1 else (fit is None)


# ---- topological_order, graph_metrics


def test_topological_order_goldens():
    assert topological_order(["A", "B", "C"], [("A", "B"), ("B", "C")]) == [["A"], ["B"], ["C"]]
    assert topological_order(["A", "B", "C"], [("A", "C"), ("B", "C")]) == [["A", "B"], ["C"]]


def test_topological_order_depth_is_the_longest_path():
    gens = topological_order(["A", "B", "C", "D"], [("A", "B"), ("B", "C"), ("A", "C"), ("A", "D")])
    depth = {k: i for i, layer in enumerate(gens) for k in layer}
    assert depth == {"A": 0, "B": 1, "C": 2, "D": 1}
    assert topological_order(["B", "A"], []) == [["A", "B"]]


def test_graph_metrics_golden_on_the_fixture_inventory(inventory_doc):
    inv = build_inventory(entries_of(inventory_doc), edges_of(inventory_doc))
    assert graph_metrics(inv) == {
        "C01/german": (0, 0, 1),
        "C07/banco-andino": (0, 0, 1),
        "C18/banco-andino": (1, 1, 0),
        "C22/generators": (1, 1, 0),
    }


def test_graph_metrics_counts_distinct_entries():
    inv = build_inventory(_keyed("A", "B"), [Edge("A/f", "B/f", "document", "s"), Edge("A/f", "B/f", "code", "t")])
    assert graph_metrics(inv) == {"A/f": (0, 0, 1), "B/f": (1, 1, 0)}


# ---- attribute_completeness


def test_attribute_completeness_goldens_on_the_fixture_entries(inventory_doc):
    comp = attribute_completeness(entries_of(inventory_doc))
    assert comp["A32"] == {"present": 0, "derivable": 0, "convention": 0, "not_applicable": 4, "missing": 0}
    assert comp["A10"] == {"present": 0, "derivable": 0, "convention": 0, "not_applicable": 0, "missing": 4}
    assert comp == inventory_doc["completeness"]["attributes"]
    assert list(comp) == ATTR_IDS


def test_attribute_completeness_counts_sum_to_the_entries():
    ents = [
        make_entry("A", "f", components=(Component("A", "f", "P0", "champion"),)),
        make_entry(
            "B",
            "f",
            attrs=attributes(A09={"value": None, "status": "missing", "basis": "no field"}),
            components=(Component("B", "f", "P0", "champion"),),
        ),
    ]
    comp = attribute_completeness(ents)
    assert all(sum(c.values()) == 2 for c in comp.values())
    assert comp["A09"]["missing"] == 1 and comp["A09"]["convention"] == 1
