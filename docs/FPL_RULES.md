# FPL scoring rules by season (recovered from data, not typed from memory)

Source: `pipeline/09_validate_fpl.py` regresses `total_points` on event counts per
season (`FPL_RULES_RECOVERED.csv`). In every season 2016-17 to 2026-27 the fit is
exact (R^2 = 1, 100% of 250k+ player-matches reproduced), so the table below is the
rule set that was actually applied. 2026-27 coefficients also match the rules the
API publishes (`data/fpl/fpl_scoring_rules.csv`).

| event | points | seasons |
|---|---|---|
| appearance 1-59 min / 60+ min | 1 / 2 | all |
| goal: DEF / MID / FWD | 6 / 5 / 4 | all |
| goal: GKP | 6 (one observation, 2020-21); **API publishes 10 for 2026-27** | see caveat 1 |
| assist | 3 | all |
| clean sheet (60+ min): GKP, DEF / MID / FWD | 4 / 1 / 0 | all |
| goals conceded: -1 per 2 (GKP, DEF) | -1 | all |
| saves: +1 per 3 (GKP) | 1 | all |
| penalty saved / missed | +5 / -2 | all |
| yellow / red card | -1 / -3 | all |
| own goal | -2 | all |
| bonus | 1-3 (stat `bonus`) | all |
| defensive contribution: DEF >= 10; MID, FWD >= 12 | +2 | 2025-26 onward only |

Managers (`position = MNG`) were scored in 2024-25 only and are excluded from player models.

## Caveats
1. **GKP goal value 10 (2026-27)** is published by the API but unobserved in the data
   (no goalkeeper has scored this season). Verify when one does; until then treat as unconfirmed.
2. **FWD defensive-contribution threshold (2026-27)**: only one forward has reached any threshold
   so far, so it cannot be recovered. 2025-26 data say 12. The API lists the points value (2) only.
3. The API publishes points per event but not the thresholds, which here were inferred from data.
4. Rules are unchanged 2016-17 to 2024-25 apart from the manager entries.
