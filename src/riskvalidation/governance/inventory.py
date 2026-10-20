"""The model inventory (dossier 06 B.2 l.539-551, I.2): entries keyed ``case_id/fit_id`` (W15-1), their components,
the dependency edges (W15-8) and the graph checks.

An :class:`Entry` holds the forty attributes of B.2, each as a value, a status and a basis, never a bare value. A
dependency is an edge between two entries with a stated source; page statements, shared methods and shared sources
are "related" links, never edges. The engine node is not an entry: every entry depends on it implicitly.
:func:`build_inventory` is the only producer of an :class:`Inventory`; it refuses a repeated key, a cycle (named
in dependency order) and an edge to an absent entry.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

ROLES = ("champion", "challenger", "benchmark", "reference")
STATUSES = ("present", "derivable", "convention", "not-applicable", "missing")
ATTRIBUTE_IDS = tuple(f"A{i:02d}" for i in range(1, 41))
EDGE_KINDS = ("document", "code")
_POINTER = re.compile(r"\S+#/\S*")
_MIN_REASON = 8


@dataclass(frozen=True)
class Component:
    """A model record of an entry; the key is the triple, because model ids repeat across cases."""

    case_id: str
    fit_id: str
    model_id: str
    role: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"component {self.key}: role {self.role!r} is not one of {ROLES}")

    @property
    def key(self) -> str:
        return f"{self.case_id}/{self.fit_id}/{self.model_id}"


def _check_attributes(key: str, attributes: Mapping[str, Any]) -> None:
    for aid in ATTRIBUTE_IDS:
        if aid not in attributes:
            raise ValueError(f"{key} {aid}: the attribute is missing; an entry holds A01 to A40")
    extra = sorted(set(attributes) - set(ATTRIBUTE_IDS))
    if extra:
        raise ValueError(f"{key} {extra[0]}: not an attribute of B.2 (A01 to A40)")
    for aid in ATTRIBUTE_IDS:
        a = attributes[aid]
        if not isinstance(a, Mapping) or not {"value", "status", "basis"} <= set(a):
            raise ValueError(f"{key} {aid}: an attribute is an object with a value, a status and a basis")
        status, value, basis = a["status"], a["value"], a["basis"]
        text = basis if isinstance(basis, str) else ""
        if status not in STATUSES:
            raise ValueError(f"{key} {aid}: unknown status {status!r}")
        if status == "present" and not _POINTER.search(text):
            raise ValueError(f"{key} {aid}: status present without a pointer (a path#/pointer in the basis)")
        if status == "not-applicable" and len(text.strip()) < _MIN_REASON:
            raise ValueError(
                f"{key} {aid}: status not-applicable without a reason of at least {_MIN_REASON} characters"
            )
        if status in ("missing", "not-applicable") and value is not None:
            raise ValueError(f"{key} {aid}: status {status} holds a value (an absent attribute holds none)")
        if status in ("derivable", "convention") and not text.strip():
            raise ValueError(f"{key} {aid}: status {status} without a basis")
    # SDD non-goal 1: Contraste approves no model, so the approver (A32) and the date and authority (A33) are
    # not applicable
    for aid, what in (("A32", "approver"), ("A33", "date and authority")):
        a = attributes[aid]
        if a["status"] != "not-applicable" or a["value"] is not None:
            raise ValueError(
                f"{key} {aid}: the {what} must be not-applicable with no value (SDD non-goal 1: Contraste approves "
                f"no model)"
            )


@dataclass(frozen=True)
class Entry:
    """One fit of one case: the unit of the inventory."""

    case_id: str
    fit_id: str
    attributes: dict[str, dict]  # "A01" to "A40" -> {"value": ..., "status": ..., "basis": ...}
    variants: tuple[tuple[str, str], ...]  # (variant_id, basis of the assignment to this fit)
    components: tuple[Component, ...]

    def __post_init__(self) -> None:
        _check_attributes(self.key, self.attributes)
        seen: set[str] = set()
        for vid, basis in self.variants:
            if vid in seen:
                raise ValueError(f"{self.key}: variant {vid!r} is listed twice")
            seen.add(vid)
            if not str(basis).strip():
                raise ValueError(f"{self.key}: variant {vid!r} has no basis for its assignment to the fit")
        keys: set[str] = set()
        for c in self.components:
            if c.key in keys:
                raise ValueError(f"component key {c.key!r} repeats")
            keys.add(c.key)

    @property
    def key(self) -> str:
        return f"{self.case_id}/{self.fit_id}"


@dataclass(frozen=True)
class Edge:
    """A dependency: ``upstream`` is read or imported by ``downstream``."""

    upstream: str
    downstream: str
    kind: str  # "document" (a document read) or "code" (an import)
    stated_source: str  # the file and line, or the document field, that states it

    def __post_init__(self) -> None:
        if self.kind not in EDGE_KINDS:
            raise ValueError(
                f"edge {self.upstream} to {self.downstream}: kind {self.kind!r} is not one of {EDGE_KINDS}"
            )
        if not self.stated_source.strip():
            raise ValueError(f"edge {self.upstream} to {self.downstream}: the stated source is empty")


@dataclass(frozen=True)
class Inventory:
    """Entries and edges in the order given; produced by :func:`build_inventory`, which checks every invariant."""

    entries: tuple[Entry, ...]
    edges: tuple[Edge, ...]


def assign_variant(variant_sources: Sequence[str], fits: Mapping[str, Sequence[str]]) -> tuple[str | None, str]:
    """The fit a variant document belongs to, with the basis, or ``(None, reason)`` (W15-15).

    A case with one fit gives its variants to that fit; otherwise the fit whose sources contain all the variant's
    sources, when exactly one does. Any other variant is never assigned by guess.
    """
    if not fits:
        return None, "the case has no fit to assign the variant to"
    if len(fits) == 1:
        return next(iter(fits)), "the single models document of the case"
    sources = list(variant_sources)
    if not sources:
        return None, "the variant's provenance names no source, so no fit can be determined"
    holders = [fit for fit, src in fits.items() if all(s in src for s in sources)]
    if len(holders) == 1:
        return holders[0], f"the variant's provenance reads {', '.join(sources)}, the sources of the fit {holders[0]}"
    if not holders:
        return None, f"no fit holds all of the variant's sources ({', '.join(sources)})"
    return (
        None,
        f"{len(holders)} fits hold all of the variant's sources ({', '.join(sources)}): {', '.join(sorted(holders))}",
    )


def _kahn(keys: Sequence[str], edges: Sequence[tuple[str, str]]) -> tuple[list[list[str]], set[str]]:
    keyset = set(keys)
    pairs = sorted({(u, d) for u, d in edges})
    for u, d in pairs:
        for x in (u, d):
            if x not in keyset:
                raise ValueError(f"edge {u} to {d}: unknown entry {x}")
    indegree = {k: 0 for k in keyset}
    for _, d in pairs:
        indegree[d] += 1
    generations: list[list[str]] = []
    remaining = set(keyset)
    while True:
        layer = sorted(k for k in remaining if indegree[k] == 0)
        if not layer:
            break
        generations.append(layer)
        remaining -= set(layer)
        done = set(layer)
        for u, d in pairs:
            if u in done:
                indegree[d] -= 1
    return generations, remaining


def find_cycle(keys: Sequence[str], edges: Sequence[tuple[str, str]]) -> list[str] | None:
    """The cycle in dependency order, from its smallest key and closed on that key (``["A", "B", "C", "A"]``), or
    ``None``.

    The cycle through the smallest key that lies on any cycle, the shortest one, successors taken in sorted order:
    deterministic. A self edge is the cycle ``[k, k]``.
    """
    _, remaining = _kahn(keys, [(u, d) for u, d in edges])
    pairs = sorted({(u, d) for u, d in edges if u in remaining and d in remaining})
    succ: dict[str, list[str]] = {k: [] for k in remaining}
    for u, d in pairs:
        succ[u].append(d)
    for start in sorted(remaining):
        parent: dict[str, str | None] = {start: None}
        queue = [start]
        found = None
        while queue and found is None:
            node = queue.pop(0)
            for nxt in sorted(succ[node]):
                if nxt == start:
                    found = node
                    break
                if nxt not in parent:
                    parent[nxt] = node
                    queue.append(nxt)
        if found is not None:
            path = [found]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])  # type: ignore[arg-type]
            path.reverse()
            return [*path, start]
    return None


def nodes_on_cycles(keys: Sequence[str], edges: Sequence[tuple[str, str]]) -> set[str]:
    """The keys that lie on a cycle (a node merely downstream of a cycle is not one)."""
    _, remaining = _kahn(keys, edges)
    succ: dict[str, set[str]] = {k: set() for k in remaining}
    for u, d in edges:
        if u in remaining and d in remaining:
            succ[u].add(d)
    out = set()
    for start in remaining:
        seen: set[str] = set()
        stack = list(succ[start])
        while stack:
            node = stack.pop()
            if node == start:
                out.add(start)
                break
            if node not in seen:
                seen.add(node)
                stack.extend(succ[node])
    return out


def topological_order(keys: Sequence[str], edges: Sequence[tuple[str, str]]) -> list[list[str]]:
    """The generations of Kahn's algorithm, each sorted; the index of a key's generation is its depth (the longest
    path to it).
    """
    generations, remaining = _kahn(keys, edges)
    if remaining:
        cycle = find_cycle(keys, edges)
        raise ValueError("dependency cycle: " + ", ".join(cycle or sorted(remaining)))
    return generations


def build_inventory(
    entries: Sequence[Entry], edges: Sequence[Edge], *, related: Sequence[tuple[str, str]] = ()
) -> Inventory:
    """Check the entries and edges and return the :class:`Inventory`, in the order given.

    ``related`` is the optional list of (a, b) entry keys of the "related" links (never edges): each must name two
    entries.
    """
    if not entries:
        raise ValueError("the inventory holds no entry")
    keys: list[str] = []
    for e in entries:
        if e.key in keys:
            raise ValueError(f"entry key {e.key!r} repeats")
        keys.append(e.key)
    comp_seen: set[str] = set()
    for e in entries:
        for c in e.components:
            if c.key in comp_seen:
                raise ValueError(f"component key {c.key!r} repeats")
            comp_seen.add(c.key)
    known = set(keys)
    seen_edges: set[tuple[str, str, str]] = set()
    for ed in edges:
        for x in (ed.upstream, ed.downstream):
            if x not in known:
                raise ValueError(f"edge {ed.upstream} to {ed.downstream}: unknown entry {x}")
        if ed.upstream == ed.downstream:
            raise ValueError(f"edge {ed.upstream} to itself")
        triple = (ed.upstream, ed.downstream, ed.kind)
        if triple in seen_edges:
            raise ValueError(f"edge {ed.upstream} to {ed.downstream} ({ed.kind}) repeats")
        seen_edges.add(triple)
    for a, b in related:
        for x in (a, b):
            if x not in known:
                raise ValueError(f"related link {a} and {b}: unknown entry {x}")
    topological_order(keys, [(ed.upstream, ed.downstream) for ed in edges])
    return Inventory(entries=tuple(entries), edges=tuple(edges))


def graph_metrics(inventory: Inventory) -> dict[str, tuple[int, int, int]]:
    """Entry key to (depth, fan-in, fan-out): the longest dependency path to the entry and the counts of distinct
    entries.
    """
    keys = [e.key for e in inventory.entries]
    pairs = sorted({(ed.upstream, ed.downstream) for ed in inventory.edges})
    depth = {k: i for i, layer in enumerate(topological_order(keys, pairs)) for k in layer}
    return {k: (depth[k], len({u for u, d in pairs if d == k}), len({d for u, d in pairs if u == k})) for k in keys}


def attribute_completeness(entries: Sequence[Any]) -> dict[str, dict[str, int]]:
    """For each of A01 to A40 the number of entries per status; the five counts sum to the number of entries.

    An entry is an :class:`Entry` or its document form (a mapping with ``attributes``).
    """
    out = {
        aid: {"present": 0, "derivable": 0, "convention": 0, "not_applicable": 0, "missing": 0} for aid in ATTRIBUTE_IDS
    }
    for e in entries:
        attributes = e.attributes if hasattr(e, "attributes") else e["attributes"]
        for aid in ATTRIBUTE_IDS:
            out[aid][attributes[aid]["status"].replace("-", "_")] += 1
    return out
