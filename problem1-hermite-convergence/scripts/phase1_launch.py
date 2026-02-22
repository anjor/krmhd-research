#!/usr/bin/env python3
"""
Phase 1 Hermite Convergence Survey - Coarse Parameter Sweep
============================================================

Launch Phase 1 of Problem 1: M ∈ {8, 16, 32} × ν ∈ {0.001, 0.01, 0.1}
Total: 9 runs, ~12 GPU-hours estimated

Based on Anjor's direction (2026-02-22):
- Convergence threshold: 1e-3 (empirical discovery)
- Validation metrics: Track all three (let data reveal sensitivity)  
- Key physics: Capture ν-independent dissipation at low ν
- Forcing: Single ion, balanced (σ_c ≈ 0)
- Resolution: 64³
"""

import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
import subprocess
import shutil

# Add GANDALF to path
sys.path.insert(0, str(Path.home() / '.openclaw/workspace-physics/gandalf/src'))

def create_phase1_configs():
    """Generate all Phase 1 configuration files."""
    
    # Phase 1 parameter matrix
    M_values = [8, 16, 32]
    nu_values = [0.001, 0.01, 0.1]
    
    # Base configuration template
    base_config = {
        # Grid and resolution
        'Nx': 64, 'Ny': 64, 'Nz': 64,
        'Lx': 1.0, 'Ly': 1.0, 'Lz': 1.0,
        
        # Time integration
        'dt': 0.005,
        'max_time': 5.0,  # 5 τ_A total (2 spinup + 3 statistics)
        'save_interval': 0.1,  # Save every 0.1 τ_A for monitoring
        
        # Physics
        'beta': 0.1,  # Plasma beta
        'closure': 'closure_zero',  # Standard closure
        'species': 'single_ion',  # Single ion species
        
        # Forcing (balanced)
        'force_kmin': 1, 'force_kmax': 2,
        'force_rate': 1.0,
        'sigma_c_target': 0.0,  # Balanced forcing
        
        # Hypercollision (quadratic, as planned)
        'hyper_n': 2,
        'hyper_coeff': 1e-3,
        
        # Diagnostics (enhanced for convergence study)
        'diagnostics': {
            'energy_spectrum': True,
            'heat_flux': True, 
            'cross_helicity': True,
            'hermite_moments': True,
            'convergence_monitor': True
        }
    }
    
    configs = []
    config_dir = Path.home() / '.openclaw/workspace-physics/configs/phase1'
    config_dir.mkdir(parents=True, exist_ok=True)
    
    run_id = 0
    for M in M_values:
        for nu in nu_values:
            run_id += 1
            
            # Create configuration
            config = base_config.copy()
            config.update({
                'M_max': M,
                'nu': nu,
                'run_name': f'phase1_M{M:02d}_nu{nu:.3f}',
                'run_id': run_id,
                'phase': 1
            })
            
            # Save configuration
            config_file = config_dir / f'phase1_M{M:02d}_nu{nu:.3f}.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            configs.append({
                'config_file': config_file,
                'M': M, 
                'nu': nu,
                'run_name': config['run_name'],
                'estimated_gpu_hours': estimate_gpu_time(M, 64**3, 5.0)
            })
    
    return configs

def estimate_gpu_time(M, grid_points, sim_time):
    """Estimate GPU time based on scaling M × N³ × T."""
    # Base: M=16, 64³, 5τ_A ≈ 0.8 GPU-hours
    base_M = 16
    base_N3 = 64**3  
    base_T = 5.0
    base_hours = 0.8
    
    scaling = (M / base_M) * (grid_points / base_N3) * (sim_time / base_T)
    return base_hours * scaling

def launch_run(config_info):
    """Launch a single GANDALF run."""
    print(f"\n🚀 Launching: {config_info['run_name']}")
    print(f"   M = {config_info['M']}, ν = {config_info['nu']}")
    print(f"   Estimated time: {config_info['estimated_gpu_hours']:.1f} GPU-hours")
    
    # Change to GANDALF directory
    gandalf_dir = Path.home() / '.openclaw/workspace-physics/gandalf'
    os.chdir(gandalf_dir)
    
    # Activate virtual environment and run
    cmd = [
        'source', '.venv/bin/activate', '&&',
        'python', '-m', 'krmhd.main',
        '--config', str(config_info['config_file']),
        '--output-dir', f'results/problem1/phase1/{config_info["run_name"]}'
    ]
    
    try:
        # Run in background, capture output to log
        log_file = Path(f'logs/phase1_{config_info["run_name"]}.log')
        log_file.parent.mkdir(exist_ok=True)
        
        with open(log_file, 'w') as f:
            result = subprocess.run(
                ' '.join(cmd), 
                shell=True, 
                stdout=f, 
                stderr=subprocess.STDOUT,
                cwd=gandalf_dir
            )
        
        if result.returncode == 0:
            print(f"   ✅ Completed successfully")
            return True
        else:
            print(f"   ❌ Failed (return code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def create_monitoring_script():
    """Create a script to monitor Phase 1 progress."""
    
    monitor_script = Path.home() / '.openclaw/workspace-physics/scripts/monitor_phase1.py'
    
    monitor_code = '''#!/usr/bin/env python3
"""Monitor Phase 1 progress and convergence."""

import os
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def check_phase1_status():
    """Check status of all Phase 1 runs."""
    
    results_dir = Path.home() / '.openclaw/workspace-physics/gandalf/results/problem1/phase1'
    
    print("Phase 1 Status Check")
    print("=" * 50)
    print(f"Time: {datetime.now()}")
    
    # Expected runs
    M_values = [8, 16, 32]
    nu_values = [0.001, 0.01, 0.1]
    
    completed = 0
    failed = 0
    running = 0
    
    for M in M_values:
        for nu in nu_values:
            run_name = f'phase1_M{M:02d}_nu{nu:.3f}'
            run_dir = results_dir / run_name
            
            if not run_dir.exists():
                status = "❌ Not started"
            elif (run_dir / 'COMPLETED').exists():
                status = "✅ Completed"
                completed += 1
            elif (run_dir / 'FAILED').exists():
                status = "💥 Failed"
                failed += 1
            else:
                status = "🔄 Running"
                running += 1
                
            print(f"  {run_name:<25} {status}")
    
    print()
    print(f"Summary: {completed} completed, {running} running, {failed} failed")
    
    return completed, running, failed

def plot_early_convergence():
    """Plot early convergence results as they come in."""
    
    results_dir = Path.home() / '.openclaw/workspace-physics/gandalf/results/problem1/phase1'
    
    # Find completed runs with convergence data
    convergence_data = []
    
    for run_dir in results_dir.glob('phase1_*'):
        if not (run_dir / 'COMPLETED').exists():
            continue
            
        # Parse M and nu from directory name
        parts = run_dir.name.split('_')
        M = int(parts[1][1:])  # M08 -> 8
        nu = float(parts[2][2:])  # nu0.001 -> 0.001
        
        # Load convergence data
        conv_file = run_dir / 'convergence.npz'
        if conv_file.exists():
            data = np.load(conv_file)
            final_ratio = data['E_M_ratio'][-1]  # Final E_M/E_total
            convergence_data.append((M, nu, final_ratio))
    
    if not convergence_data:
        print("No convergence data available yet.")
        return
    
    # Create convergence plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    nu_unique = sorted(set(d[1] for d in convergence_data))
    colors = plt.cm.viridis(np.linspace(0, 1, len(nu_unique)))
    
    for i, nu in enumerate(nu_unique):
        M_vals = [d[0] for d in convergence_data if d[1] == nu]
        ratios = [d[2] for d in convergence_data if d[1] == nu]
        
        ax.loglog(M_vals, ratios, 'o-', color=colors[i], 
                 label=f'ν = {nu}', markersize=8)
    
    ax.axhline(1e-3, color='red', linestyle='--', alpha=0.7, 
               label='Threshold (10⁻³)')
    
    ax.set_xlabel('Hermite Moments (M)')
    ax.set_ylabel('E_M / E_total')
    ax.set_title('Phase 1: Hermite Convergence (Preliminary)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plot_file = Path.home() / '.openclaw/workspace-physics/results/phase1_preliminary.png'
    plot_file.parent.mkdir(exist_ok=True)
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Convergence plot saved: {plot_file}")
    
    plt.close()

if __name__ == '__main__':
    check_phase1_status()
    plot_early_convergence()
'''
    
    with open(monitor_script, 'w') as f:
        f.write(monitor_code)
    
    # Make executable
    os.chmod(monitor_script, 0o755)
    
    return monitor_script

def main():
    """Main execution: Generate configs and launch Phase 1."""
    
    print("Phase 1 Hermite Convergence Survey")
    print("=" * 50)
    print("Anjor's Direction (2026-02-22):")
    print("  • Convergence threshold: 10⁻³ (empirical discovery)")
    print("  • Track all validation metrics (E(k⊥), Q, σ_c)")
    print("  • Capture ν-independent dissipation at low ν")
    print("  • Single ion, balanced forcing")
    print("  • 64³ resolution")
    print()
    
    # Generate configurations
    print("🔧 Generating Phase 1 configurations...")
    configs = create_phase1_configs()
    
    total_gpu_hours = sum(c['estimated_gpu_hours'] for c in configs)
    print(f"   Generated {len(configs)} configurations")
    print(f"   Estimated total: {total_gpu_hours:.1f} GPU-hours")
    print()
    
    # Create monitoring script  
    print("📊 Creating monitoring script...")
    monitor_script = create_monitoring_script()
    print(f"   Monitor script: {monitor_script}")
    print(f"   Usage: python {monitor_script}")
    print()
    
    # Launch runs
    print("🚀 Launching Phase 1 simulations...")
    print(f"   Parameter matrix: M ∈ {{8, 16, 32}} × ν ∈ {{0.001, 0.01, 0.1}}")
    print()
    
    successes = 0
    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}]", end=" ")
        if launch_run(config):
            successes += 1
    
    print()
    print("=" * 50)
    print(f"Phase 1 Launch Summary:")
    print(f"  • {successes}/{len(configs)} runs launched successfully")
    print(f"  • Estimated completion: {total_gpu_hours:.1f} GPU-hours")
    print(f"  • Monitor progress: python {monitor_script}")
    print()
    print("🔬 Key Physics to Watch:")
    print("  • ν-independent M_crit at low ν (collisionless regime)")
    print("  • Transition from M_crit ~ ν⁻ᵅ to saturation")
    print("  • Validation metric sensitivity")

if __name__ == '__main__':
    main()