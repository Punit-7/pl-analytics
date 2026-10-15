# P5 findings

One dated line per observation: amber performance levels, drift alerts, gate decisions worth noting, and measurements. Newest last. See [RUNBOOK.md](../RUNBOOK.md) for when to write here.

| Date | Check | Value | Note |
|---|---|---|---|
| 2026-10-09 | Gate | gain 0.0002 RPS | Below the 0.001 required; challenger (`dc_xi = 0.0030`) refused, champion kept |
| 2026-10-09 | Drift (PSI, 4 columns) | max 0.0742 (`total_goals`) | All below the 0.25 alert level |
| 2026-10-09 | Live performance | waiting | No scored matches yet |

## Open measurements

- Docker image size (`docker images`) and container memory (`docker stats`), from P5 Stage 8. Not yet recorded; the memory figure must stay well under Render's 512 MB.
