# Design: U5-E2 local statutory provisions

Requirements: [`requirements.md`](requirements.md). Research: Contraste dossier 03 (sections D and K). Contract:
`engine-provisions-api` 1.0.0 (Contraste `docs/contracts/engine/provisions-api.md`), modules `local_cl` and `local_es`.

## Package layout

| Module | Holds |
|---|---|
| `riskvalidation.provisions.local_cl` | `b1_individual_category`, `b1_individual_provision`, `b1_default_provision`, `b1_mortgage_pe`, `b1_leasing_pe`, `b1_student_loan_pe`, `b1_commercial_pe`, `b1_consumer_pe`, `b1_group_provision`, `b1_minimum_normal_check` (`MinimumCheck`), `b1_group_max_rule`, `b1_additional_provisions` |
| `riskvalidation.provisions.local_es` | `annex9_uncovered_amount`, `annex9_normal_coverage`, `annex9_doubtful_coverage`, `annex9_category_to_stage`, `annex9_coverage` |

## Decisions

- The tables are module constants written as decimals (1.0916 per cent is 0.010916), one tuple per table row, each with its
  source paragraph in a comment; the fixtures under `tests/fixtures/provisions_local/` repeat them in per cent as the dossier
  prints them, and every cell is asserted against the fixture (divided by 100 in the test).
- The mortgage PE is the transcribed cell (it equals PI x PDI within the table's own rounding, 5e-7); the other PEs are
  computed as PI x PDI, as the contract states.
- Bucket comparisons round the tested ratio to 1e-9 first (TPE, PVG, PVB, PTVG), so a float at a boundary lands in its inclusive
  bucket; the contract states this for TPE and the rule is applied uniformly.
- The 0.50% minimum lives in `b1_minimum_normal_check` only. The module has no B-1 total and no function that takes the
  additional provisions, so they cannot enter a comparable total; it imports nothing from the IFRS 9 modules.
- Invalid inputs raise with the category and the rule: `KeyError` for `b1_individual_category` (contract), `ValueError`
  elsewhere. A doubtful Annex 9 account without arrears above 90 days raises `ValueError`; the caller reports it as not
  evaluated (the contract's rule), since no coverage is transcribed for it.
- The contract defines no B-1 total or bridge function; none is added.

## Limits

No compliance claim. The August 2026 CMF reform is not in force and is not implemented. The Annex 9 point numbers after
Circular 1/2025 are UNVERIFIED. The collateral haircut table of Annex 9 and the MP factor of subsidised mortgages are not
transcribed in the dossier and not implemented.
