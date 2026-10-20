# 01 Scope and inventory

Modules `riskvalidation.governance.scope` and `riskvalidation.governance.inventory`. Sources: dossier 06 B.1 (l.523-537), B.2 (l.539-551), I.2;
the U15 research addendum D1 (the B.2 attribute map), D4 (the dependencies) and D6 (the regulation check); decisions W15-1, W15-4, W15-8 and W15-15.

## The B.1 scope test

Four answers (`ScopeAnswers`) and a jurisdiction profile give an in-scope decision with its rationale (`ScopeDecision`):

| Answer | Question |
|---|---|
| (a) `applies_theory` | does the model apply statistical, economic, financial or mathematical theory |
| (b) `output_is_estimate` | is the output an estimate |
| (c) `deterministic` | is it a deterministic rule set |
| (d) `generative_or_agentic` | is it a generative or agentic AI model |

`scope_test(answers, profile)` reads the answers with the rule of the profile (a reading of the B.1 table, **UNVERIFIED** beyond it):

| Profile | Source | In scope when |
|---|---|---|
| `us-2026` | SR 26-2 | (a) and (b) hold and not (d): a deterministic rule set without theory and a generative or agentic model are out |
| `pra` | SS1/23 Principle 1.1 | (a) holds; the output need not be quantitative |
| `osfi-2027` | OSFI E-23 | (a) and (b) hold |
| `ecb` | ECB | (a) and (b) hold |

The rationale names the answers that decided it; all four answers are recorded, including the one a profile does not read. An unknown profile
raises `ValueError`. `scope_record(answers)` returns the four decisions in the order `us-2026`, `pra`, `osfi-2027`, `ecb` (the inventory field
`scope.profiles`). Goldens: `scope_test(ScopeAnswers(True, True, False, False), "us-2026").in_scope` is true; with `generative_or_agentic`
true it is false; `ScopeAnswers(True, False, False, False)` is in scope under `pra` and out under `ecb`. The rationale is English; the
bake adds the Spanish text.

## The inventory

An `Entry` is one fit of one case, keyed `case_id/fit_id` (W15-1: a variation warranting separate validation is a separate model, SR 11-7).
Its `attributes` are the forty attributes of B.2, each an object with a `value`, a `status` and a `basis`, never a bare value:

| Block | Attributes |
|---|---|
| Identity (7) | A01 model ID, A02 name, A03 version, A04 description, A05 model type, A06 origin, A07 status (lifecycle state) |
| Purpose and use (6) | A08 purpose, A09 products and portfolios, A10 intended use against actual use, A11 approved uses, A12 restrictions on use, A13 model operating boundaries |
| Construction (8) | A14 input data sources, A15 alternative or unstructured data flag, A16 upstream and downstream model dependencies, A17 outputs and intended use, A18 methodology, A19 assumptions and limitations, A20 use of ML, A21 dynamic or self-updating flag |
| Risk (6) | A22 materiality rating, A23 complexity rating, A24 tier, A25 explainability rating, A26 PMAs in force, A27 exceptions to policy |
| Governance (13) | A28 owner, A29 developer, A30 users, A31 validator, A32 approver, A33 date and authority of an approval, A34 last validation date and type, A35 next validation date, A36 monitoring status, A37 open findings by severity, A38 remediation due dates, A39 time frame of validity, A40 decommission date and rationale |

The status is one of `present` (read in a document: the basis names a `path#/pointer`), `derivable` (computed from documents: the basis says
how), `convention` (set by a stated convention: the basis says which), `not-applicable` (the basis is a reason of at least eight characters) and
`missing` (no document holds it). A `missing` or `not-applicable` attribute has the value `None`, never an invented one; `Entry` raises
`ValueError` naming the attribute that breaks a rule. A32 (approver) and A33 are `not-applicable` with the value `None` because Contraste
approves no model (SDD non-goal 1).

A `Component` is a model record of the entry, keyed `case_id/fit_id/model_id` because model ids repeat across cases (P0-constant and
P1-scorecard repeat in C01 and C03); its role is `champion`, `challenger`, `benchmark` or `reference`.

`Entry.variants` lists the variant documents of the entry with the basis of each assignment (W15-15). `assign_variant(variant_sources, fits)`
assigns a variant to the fit whose sources contain all the variant's sources, when exactly one does, and gives the variants of a one-fit case
to that fit ("the single models document of the case"); any other variant returns `(None, reason)` and goes to the inventory's `excluded`
list, never assigned by guess. Golden: `assign_variant(['uci-german'], {'german': ['uci-german'], 'taiwan': ['uci-taiwan']})` is
`('german', "the variant's provenance reads uci-german, the sources of the fit german")`.

## Edges and the graph

An `Edge` joins two entry keys with a kind, `document` (a document read) or `code` (an import), and the source that states it (W15-8). Page
statements, shared methods and shared sources are "related" links, never edges. The engine node `riskvalidation` is not an entry: every entry
depends on it implicitly.

`build_inventory(entries, edges)` returns the `Inventory` in the order given and refuses: zero entries (`the inventory holds no entry`), a
repeated entry key, a repeated component key (`component key 'C01/german/P0' repeats`), an edge to an absent entry (`edge A to Z: unknown entry
Z`), a related link to an absent entry (`related link A and Z: unknown entry Z`, through the optional `related` argument), a repeated edge, a
self edge and a cycle, named in dependency order from its smallest key and closed on it (`dependency cycle: A, B, C, A`).

`topological_order(keys, edges)` returns the generations of Kahn's algorithm, each sorted; the index of a generation is the depth, the length
of the longest dependency path to the entry: `[['A', 'B'], ['C']]` for the edges A to C and B to C. `graph_metrics(inventory)` returns, per entry,
(depth, fan-in, fan-out) counting distinct entries. On the fixture inventory, C01/german is (0, 0, 1), C07/banco-andino (0, 0, 1),
C18/banco-andino (1, 1, 0) and C22/generators (1, 1, 0).

`attribute_completeness(entries)` counts, for each of A01 to A40, the entries per status; the five counts sum to the number of entries. On the
four fixture entries A32 is four `not_applicable` and A10 four `missing`.

## Tests

The tests of this page are `tests/governance/test_governance_scope.py` and `tests/governance/test_governance_inventory.py`; the fixture inventory is reproduced
field by field by `tests/governance/test_governance_fixtures.py`. The completeness of the record is the `gov.` test
[`gov.inventory_completeness`](06_validation-tests.md) and the acyclic graph is `gov.edges_acyclic`.
