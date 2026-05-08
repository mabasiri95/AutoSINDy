# -*- coding: utf-8 -*-
"""
run_sweep_v2.py  —  AutoSINDy Full Publication Sweep
=====================================================
Set SWEEP_GROUP to control what runs:
    'main'      — all systems x all noise x all seeds
    'ablation'  — expansion / library / optimizer ablations
    'chunks'    — chunk-count sensitivity
    'all'       — everything

Crash Recovery
--------------
A sidecar file (STARTED_LOG) is written *before* each run and stamped
completed=True *after* it succeeds.  On restart:

  • log entry says completed=True   → SKIP (already done)
  • log entry says completed=False  → kernel died mid-run → retry with
                                       fallback seed (nominal + 100, cumulative)
  • no log entry at all             → fresh run, use seed as normal

Entries are NEVER deleted, only stamped.  This means a nominal seed whose
fallback finished is permanently marked done, so future restarts never attempt
the bad original seed again.

Isolation: the log key is (system | noise | nominal_seed | run_tag), so every
combination of system, noise, seed, and sweep group is tracked independently —
main, ablation, and chunk sweeps never interfere with each other.
"""

import copy
import systems
import AutoSINDy
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime

# ─── Choose what to run ───────────────────────────────────────────────────────
SWEEP_GROUP = 'main'   # 'main' | 'ablation' | 'chunks' | 'all'

# ─── Main sweep parameters ────────────────────────────────────────────────────
SYSTEMS_TO_TEST = [
    'harmonic_oscillator',
    'vanderpol',
    'damped_pendulum',
    'duffing',
    'modulated_oscillator',
    'complex_lorenz',
    # 'michaelis_menten',
    # 'exponential_system',
    # 'lorenz',
]

ABLATION_SYSTEMS     = ['harmonic_oscillator', 'vanderpol', 'damped_pendulum', 'lorenz']
NOISE_LEVELS         = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
N_TRIALS             = 5          # seeds 32 … 36
CSV_FILE             = "autosindy_results_log_tidy.csv"
STARTED_LOG          = "autosindy_started_log.json"
FALLBACK_SEED_OFFSET = 100        # crashed seed N → retry with N+100


# ═══════════════════════════════════════════════════════════════════════════════
# Started-log helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_log():
    if not os.path.exists(STARTED_LOG):
        return {}
    try:
        with open(STARTED_LOG, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_log(log):
    with open(STARTED_LOG, 'w') as f:
        json.dump(log, f, indent=2)


def _log_key(sys_name, noise, nominal_seed, run_tag):
    """One permanent key per sweep-loop slot.  Uses the NOMINAL seed, not the
    actual seed, so the slot is recognised correctly across fallback retries."""
    return f"{sys_name}|{round(noise, 4)}|{nominal_seed}|{run_tag}"


def _mark_started(sys_name, noise, nominal_seed, run_tag, actual_seed):
    """Write the entry before a run begins (completed=False)."""
    log = _load_log()
    log[_log_key(sys_name, noise, nominal_seed, run_tag)] = {
        "system_name":  sys_name,
        "noise_level":  round(noise, 4),
        "nominal_seed": nominal_seed,
        "actual_seed":  actual_seed,
        "run_tag":      run_tag,
        "completed":    False,
        "started_at":   datetime.now().isoformat(timespec='seconds'),
    }
    _save_log(log)


def _mark_done(sys_name, noise, nominal_seed, run_tag, actual_seed):
    """Stamp the entry completed=True after a run succeeds.
    Never deletes — keeping the entry is what prevents future restarts from
    re-attempting a bad nominal seed whose fallback already finished."""
    log = _load_log()
    log[_log_key(sys_name, noise, nominal_seed, run_tag)] = {
        "system_name":  sys_name,
        "noise_level":  round(noise, 4),
        "nominal_seed": nominal_seed,
        "actual_seed":  actual_seed,
        "run_tag":      run_tag,
        "completed":    True,
        "completed_at": datetime.now().isoformat(timespec='seconds'),
    }
    _save_log(log)


# ═══════════════════════════════════════════════════════════════════════════════
# CSV helper
# ═══════════════════════════════════════════════════════════════════════════════

def _csv_completed():
    """Return set of (system_name, noise, actual_seed, run_tag) from the CSV."""
    if not os.path.exists(CSV_FILE):
        return set()
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception:
        return set()
    out = set()
    for _, row in df.iterrows():
        out.add((
            row['system_name'],
            round(float(row['noise_level']), 4),
            int(row.get('global_seed', 32)),
            str(row.get('run_tag', 'main')),
        ))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Core run function
# ═══════════════════════════════════════════════════════════════════════════════

def run_single(sys_name, noise, seed, run_tag='main', overrides=None):
    """
    Run one experiment slot identified by (sys_name, noise, seed, run_tag).

    `seed` is always the NOMINAL seed (what the sweep loop asks for).
    The actual seed used may differ if a crash recovery is active.

    overrides: dict of config keys to change.
      top-level:  {'discovery_chunks': 10}
      nested:     {('curation_params', 'expansion_strategy'): 'severe'}
    """
    log      = _load_log()
    csv_done = _csv_completed()
    key      = _log_key(sys_name, noise, seed, run_tag)
    entry    = log.get(key)

    # ── 1. Already stamped done in the log ────────────────────────────────────
    if entry and entry["completed"]:
        actual = entry["actual_seed"]
        label  = f"seed={actual} [fallback for {seed}]" if actual != seed else f"seed={seed}"
        print(f"  SKIP (already done): {sys_name} noise={noise} {label} tag={run_tag}")
        return

    # ── 2. Entry exists but not completed → kernel died → crash recovery ──────
    if entry and not entry["completed"]:
        prev_actual = entry["actual_seed"]
        actual_seed = prev_actual + FALLBACK_SEED_OFFSET
        print(
            f"\n  ⚠  CRASH DETECTED: {sys_name} noise={noise} nominal_seed={seed}"
            f" (previous attempt used seed={prev_actual}, kernel died)."
            f"\n  ↳  Retrying with FALLBACK seed={actual_seed} …\n"
        )

    # ── 3. No log entry → completely fresh run ────────────────────────────────
    else:
        actual_seed = seed

    # ── 4. Safety check: actual seed already in the CSV? ─────────────────────
    #    (guards against the rare case where CSV write succeeded but log stamp
    #     failed, e.g. a power cut between the two writes)
    if (sys_name, round(noise, 4), actual_seed, run_tag) in csv_done:
        print(f"  SKIP (already done): {sys_name} noise={noise} seed={actual_seed} tag={run_tag}")
        _mark_done(sys_name, noise, seed, run_tag, actual_seed)
        return

    # ── 5. Announce and record start ──────────────────────────────────────────
    if actual_seed != seed:
        print(f"--- {sys_name} | noise={noise} | seed={actual_seed}"
              f" (fallback for {seed}) | tag={run_tag} ---")
    else:
        print(f"--- {sys_name} | noise={noise} | seed={seed} | tag={run_tag} ---")

    _mark_started(sys_name, noise, seed, run_tag, actual_seed)

    # ── 6. Build config ───────────────────────────────────────────────────────
    cfg = copy.deepcopy(AutoSINDy.config)
    cfg["system_to_run"]                           = sys_name
    cfg["data_params"][sys_name]["noise_level"]    = noise
    cfg["global_seed"]                             = actual_seed
    cfg["pysr_params"]["random_state"]             = actual_seed
    cfg["data_params"][sys_name]["noise_seed"]     = actual_seed
    cfg["pysr_params"]["deterministic"]            = True
    cfg["pysr_params"]["parallelism"]              = "serial"
    cfg["simulation_params"]["models_to_simulate"] = [
        'AutoSINDy', 'Standard SINDy', 'Standard PySR'
    ]
    cfg["run_tag"] = run_tag

    if overrides:
        for k, v in overrides.items():
            if isinstance(k, tuple):    # nested: ('section', 'param')
                cfg[k[0]][k[1]] = v
            else:
                cfg[k] = v

    # ── 7. Run ────────────────────────────────────────────────────────────────
    try:
        AutoSINDy.run_experiment(cfg)
    except Exception as e:
        print(f"  FAILED: {e}")
        # Leave entry as completed=False so the next restart detects the crash
        return

    # ── 8. Verify the row actually landed in the CSV before stamping done ─────
    #    If the CSV was locked (e.g. open in Excel), run_experiment may have
    #    returned without error but without writing anything.  In that case we
    #    leave the log entry as completed=False so the next restart re-runs it.
    csv_after = _csv_completed()
    if (sys_name, round(noise, 4), actual_seed, run_tag) in csv_after:
        _mark_done(sys_name, noise, seed, run_tag, actual_seed)
    else:
        print(
            f"  ⚠  WARNING: run finished but row is missing from CSV"
            f" ({sys_name} noise={noise} seed={actual_seed} tag={run_tag})."
            f"\n  ↳  CSV may have been locked (e.g. open in Excel)."
            f"\n  ↳  Entry left as incomplete — will be retried on next run."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Crash summary — printed at the top of every sweep group
# ═══════════════════════════════════════════════════════════════════════════════

def _crash_summary():
    log     = _load_log()
    crashed = [e for e in log.values() if not e["completed"]]
    if crashed:
        print(f"\n  ℹ  {len(crashed)} previously-crashed run(s) detected"
              " — fallback seeds will be used:")
        for e in crashed:
            fb = e['actual_seed'] + FALLBACK_SEED_OFFSET
            print(f"     • {e['system_name']}  noise={e['noise_level']}"
                  f"  nominal_seed={e['nominal_seed']}  tag={e['run_tag']}"
                  f"  → fallback seed={fb}")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# Sweep groups
# ═══════════════════════════════════════════════════════════════════════════════

def sweep_main():
    print("\n" + "="*60)
    print("GROUP: main — all systems × noise levels × seeds")
    print("="*60)
    _crash_summary()
    for sys_name in SYSTEMS_TO_TEST:
        for noise in NOISE_LEVELS:
            for trial in range(N_TRIALS):
                run_single(sys_name, noise, 32 + trial, run_tag='main')
    print("\nMain sweep done.")


def sweep_ablation_expansion():
    print("\n" + "="*60)
    print("GROUP: ablation — expansion strategy")
    print("="*60)
    _crash_summary()
    for strategy in ['gentle', 'severe', 'none']:
        for sys_name in ABLATION_SYSTEMS:
            for noise in [0.0, 0.04]:
                for trial in range(3):
                    run_single(sys_name, noise, 32 + trial,
                               run_tag=f'ablation_expansion_{strategy}',
                               overrides={('curation_params', 'expansion_strategy'): strategy})


def sweep_ablation_pruning():
    print("\n" + "="*60)
    print("GROUP: ablation — pruning method")
    print("="*60)
    _crash_summary()
    for method in ['correlation']:
        for sys_name in ABLATION_SYSTEMS:
            for noise in [0.0, 0.04]:
                for trial in range(3):
                    run_single(sys_name, noise, 32 + trial,
                               run_tag=f'ablation_pruning_{method}',
                               overrides={('curation_params', 'pruning_method'): method})


def sweep_ablation_library():
    print("\n" + "="*60)
    print("GROUP: ablation — library strategy")
    print("="*60)
    _crash_summary()
    for sys_name in ABLATION_SYSTEMS:
        for noise in [0.0, 0.04]:
            for trial in range(3):
                run_single(sys_name, noise, 32 + trial,
                           run_tag='ablation_library_unified',
                           overrides={'use_unified_library': True})


def sweep_ablation_optimizer():
    print("\n" + "="*60)
    print("GROUP: ablation — optimizer")
    print("="*60)
    _crash_summary()
    for opt in ['SR3']:
        for sys_name in ABLATION_SYSTEMS:
            for noise in [0.0, 0.04, 0.08]:
                for trial in range(3):
                    run_single(sys_name, noise, 32 + trial,
                               run_tag=f'ablation_optimizer_{opt}',
                               overrides={('optimizer_params', 'name'): opt})


def sweep_chunks():
    print("\n" + "="*60)
    print("GROUP: chunks — discovery chunk count sensitivity")
    print("="*60)
    _crash_summary()
    for n_chunks in [2, 5, 10, 20]:
        for sys_name in ABLATION_SYSTEMS[:2]:
            for noise in [0.01, 0.04]:
                for trial in range(3):
                    run_single(sys_name, noise, 32 + trial,
                               run_tag=f'ablation_chunks_{n_chunks}',
                               overrides={'discovery_chunks': n_chunks})


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point  —  set SWEEP_GROUP at the top of this file
# ═══════════════════════════════════════════════════════════════════════════════

if SWEEP_GROUP in ('main', 'all'):
    sweep_main()

if SWEEP_GROUP in ('ablation', 'all'):
    sweep_ablation_expansion()
    sweep_ablation_pruning()
    sweep_ablation_library()
    sweep_ablation_optimizer()

if SWEEP_GROUP in ('chunks', 'all'):
    sweep_chunks()

print("\nSweep done. Run plot_results_v2.py to generate figures.")