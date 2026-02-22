#!/usr/bin/env python3
"""Generate Phase 1 YAML configs for GANDALF."""
import yaml
from pathlib import Path

M_values = [8, 16, 32]
nu_values = [0.001, 0.01, 0.1]

for M in M_values:
    for nu in nu_values:
        config = {
            'name': f'phase1_M{M:02d}_nu{nu:.3f}',
            'description': f'Hermite convergence: M={M}, nu={nu}',
            'grid': {
                'Nx': 64, 'Ny': 64, 'Nz': 64,
                'Lx': 6.283185307179586,
                'Ly': 6.283185307179586,
                'Lz': 6.283185307179586
            },
            'physics': {
                'v_A': 1.0,
                'eta': 0.01,
                'nu': nu,
                'beta_i': 1.0,
                'hyper_r': 2,
                'hyper_n': 2
            },
            'initial_condition': {
                'type': 'random_spectrum',
                'amplitude': 0.1,
                'alpha': 1.667,
                'k_min': 1.0,
                'k_max': 10.0,
                'M': M
            },
            'forcing': {
                'enabled': True,
                'amplitude': 0.3,
                'k_min': 1.0,
                'k_max': 3.0
            },
            'time_integration': {
                'n_steps': 1000,  # ~5 τ_A
                'cfl_safety': 0.3,
                'save_interval': 50
            },
            'io': {
                'output_dir': f'results/phase1_M{M:02d}_nu{nu:.3f}',
                'save_spectra': True,
                'save_energy_history': True,
                'save_fields': False,
                'save_final_state': True,
                'overwrite': True
            }
        }
        
        outfile = Path(f'configs/phase1_M{M:02d}_nu{nu:.3f}.yaml')
        with open(outfile, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Created: {outfile}")

print(f"\nGenerated {len(M_values) * len(nu_values)} configs")
