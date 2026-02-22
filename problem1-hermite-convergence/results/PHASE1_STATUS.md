# Phase 1 Status

## Completed Runs (6/9)

| M | ν=0.001 | ν=0.01 | ν=0.1 |
|---|---------|--------|-------|
| 8 | ✅ | ✅ | ✅ |
| 16 | ✅ | ✅ | ✅ |
| 32 | ❌ (OOM) | ⏳ | ⏳ |

## Issue: M=32 Memory

M=32 runs killed (SIGKILL) - likely out of memory on local machine.

## Options for M=32:
1. Run on Modal (cloud GPU, more memory)
2. Reduce resolution to 32³ for M=32 runs
3. Proceed with M=8,16 analysis first, then M=32 later

## Results Location
- `phase1_M08_nu*/` - M=8 results (3 runs)
- `phase1_M16_nu*/` - M=16 results (3 runs)
