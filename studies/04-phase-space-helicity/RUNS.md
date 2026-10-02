# Run sets

The agent launches production runs itself, through `loop/modal_launch.py` and nothing else (decision 1, 1 October 2026; `LOOP.md`). Before a launch, the run set gets a row here with its configs, purpose, the gate it depends on, the A100-hours estimated from measured wall times, and the timeout per run. The launcher refuses a run set that has no row. After the launch the row gets the launch ID, the run IDs and the data path. After `modal_launch.py reconcile`, it gets the actual A100-hours. Local runs on the Mac are not listed here; they go in the log.

The cap is 200 A100-hours for the whole study (`loop/config.env`). `compute_ledger.json` is the authoritative record of reservations and charges, and only the launcher writes it. `uv run python studies/04-phase-space-helicity/loop/modal_launch.py status` prints the hours used, reserved and left.

| Set | Configs | Purpose | Depends on | Est. A100-h | Timeout/run (h) | Status | Launch | Run IDs | Data path | Actual A100-h |
|---|---|---|---|---|---|---|---|---|---|---|
