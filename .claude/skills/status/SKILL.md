---
name: status
description: Use when asked for the status of an autonomous loop, workflow or study. Fixed six-item report format and where each loop keeps its checkpoint.
---
1. Read the state from the loop's checkpoint file instead of reconstructing it from logs:
   - Study 04: `studies/04-phase-space-helicity/STATUS.md`
   - Other loops: their `.loop/STATE.md` (e.g. `paper/phase-velocity-slices/.loop/STATE.md`), plus the tail of the loop's log if it has one.
2. Check whether the loop process is running (ps / job list).
3. Answer in this fixed format:
   1. Running / stopped
   2. Current phase & gate
   3. Last completed step
   4. Blockers
   5. What I need from you
   6. Next action
4. If it stopped unexpectedly, show the exact relaunch command but do NOT relaunch without asking.
