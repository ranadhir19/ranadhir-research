# MEMENTO — retention vs plasticity under distribution shift

Seed: 20261019; n = 2000; boundary angles 45/90/135 deg at t = 700/1400; 5% label noise.

| method | overall | 0-700 | 700-1400 | 1400-2000 |
|--------|---------|-------|----------|-----------|
| sliding-window refit (fine-tune analogue) | 0.567 | 0.526 | 0.299 | 0.927 |
| prototype bank (no-forgetting analogue) | 0.715 | 0.914 | 0.661 | 0.543 |

Reading: the prototype bank retains the early segments (never deleted, so old knowledge survives) but its frozen compromise lags after the shifts; the sliding-window refit tracks the current distribution and dominates the latest segment while forgetting the earlier ones. Neither dominates everywhere: a retention/plasticity trade-off, not a clean win.
