# ==============================================================================
# AutoSINDy: A Data-Driven Framework for Automated Basis Function Generation
#
# This script implements and compares two different discovery methodologies:
#
# 1. Unified Library (use_unified_library = True):
#    - Discovers terms for all state variables.
#    - Pools them into a single, unified library.
#    - Fits one SINDy model for the entire system.
#
# 2. Separate Libraries (use_unified_library = False):
#    - For each state variable, discovers and curates a specific library.
#    - Fits a separate SINDy model for each state variable independently.
#
# UPDATES:
# - Now supports SR3 (Sparse Relaxed Regularized Regression) via config.
# - Now supports Ensemble SINDy (Bagging) via config.
# ==============================================================================

import numpy as np
import pandas as pd
import pysindy as ps
import sympy
from scipy.integrate import solve_ivp
from pysr import PySRRegressor
import matplotlib.pyplot as plt
import time
import systems
from sklearn.metrics import mean_squared_error, r2_score
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression
import threading


# Suppress PySR info messages for cleaner output
import logging
logging.basicConfig()
logging.getLogger('pysr').setLevel(logging.WARNING)
import random
GLOBAL_SEED = 32
os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)



# --- Data Splitting Function ---
def split_data(X, X_dot, t, train_ratio, strategy):
    """
    Splits the trajectory data into training and testing sets based on a chosen strategy.

    Args:
        X (np.ndarray): State variable data.
        X_dot (np.ndarray): Derivative data.
        t (np.ndarray): Time vector.
        train_ratio (float): The proportion of the data to be used for training (e.g., 0.8).
        strategy (str): 'end' or 'middle'.
            - 'end': Train on the first part of the trajectory, test on the last part.
            - 'middle': Train on the middle part, test on the first and last parts combined.

    Returns:
        A tuple containing (X_train, X_dot_train, t_train, X_test, X_dot_test, t_test).
    """
    n_samples = X.shape[0]
    train_size = int(n_samples * train_ratio)
    test_size = n_samples - train_size

    if strategy == 'end':
        print(f"  Splitting data with 'end' strategy: {train_size} train samples, {test_size} test samples.")
        X_train = X[:train_size]
        X_dot_train = X_dot[:train_size]
        t_train = t[:train_size]
        X_test = X[train_size:]
        X_dot_test = X_dot[train_size:]
        t_test = t[train_size:]
    elif strategy == 'middle':
        start_test_size = test_size // 2
        end_test_size = test_size - start_test_size
        train_start_idx = start_test_size
        train_end_idx = n_samples - end_test_size

        print(f"  Splitting data with 'middle' strategy: {train_end_idx - train_start_idx} train samples, {test_size} test samples.")

        X_train = X[train_start_idx:train_end_idx]
        X_dot_train = X_dot[train_start_idx:train_end_idx]
        t_train = t[train_start_idx:train_end_idx]

        X_test = np.concatenate((X[:start_test_size], X[train_end_idx:]))
        X_dot_test = np.concatenate((X_dot[:start_test_size], X_dot[train_end_idx:]))
        t_test = np.concatenate((t[:start_test_size], t[train_end_idx:]))
    else:
        raise ValueError(f"Unknown split strategy: {strategy}. Use 'end' or 'middle'.")

    return X_train, X_dot_train, t_train, X_test, X_dot_test, t_test


def calculate_complexity_metrics(model):
    """
    Robustly calculates complexity by rebuilding the equation from SINDy's 
    internal coefficients and feature names. Uses ABS(coeff) to ensure 
    negative signs don't artificially inflate complexity compared to PySR.
    Robustly calculates complexity by forcing all equations into a canonical 
    (expanded) form. This levels the playing field between PySR (factored) 
    and SINDy (expanded).
    Returns a dictionary with two complexity metrics:
    1. 'structural': Complexity of the equation as written (Standard PySR wins here).
    2. 'canonical': Complexity after expansion (Fair comparison).
    """
    metrics = {'structural': 0, 'canonical': 0}
    exprs_to_measure = []

    # --- CASE 1: SINDy Models ---
    if hasattr(model, 'coefficients') and hasattr(model, 'get_feature_names'):
        coeffs = model.coefficients()
        features = model.get_feature_names()
        for i in range(coeffs.shape[0]):
            eq_accum = 0
            for j in range(coeffs.shape[1]):
                c = coeffs[i, j]
                if abs(c) > 1e-10:
                    feat_name = features[j].replace(" ", "*")
                    try:
                        term = c * sympy.sympify(feat_name)
                        eq_accum += term
                    except:
                        metrics['structural'] += 1
                        metrics['canonical'] += 1
            if eq_accum != 0: exprs_to_measure.append(eq_accum)

    # --- CASE 2: SeparateSINDyModel Wrapper (Recursive Fix) ---
    elif hasattr(model, 'models'):
        for sub_model in model.models:
            # Recursive call: get metrics for the sub-model
            sub_metrics = calculate_complexity_metrics(sub_model)
            # Manually sum the dictionaries
            metrics['structural'] += sub_metrics['structural']
            metrics['canonical'] += sub_metrics['canonical']
        return metrics

    # --- CASE 3: PySR Wrapper ---
    elif hasattr(model, 'equations_'): 
         try:
             # PySR provides string equations directly
             best_eq = model.get_best()['sympy_format']
             exprs_to_measure.append(best_eq)
         except: pass

    # --- CALCULATE SCORES ---
    for e in exprs_to_measure:
        try:
            if isinstance(e, str):
                e = sympy.sympify(e)
    
            # ── Structural score ──────────────────────────────────────────────
            metrics['structural'] += sympy.count_ops(e)
    
            # ── Canonical score ───────────────────────────────────────────────
            canonical = sympy.expand(e)
            # canonical = sympy.expand_trig(canonical)
    
            # Normalise sign: -2x - 3y and 2x + 3y are equally complex.
            # could_extract_minus_sign() is True when the leading term is negative.
            if canonical.could_extract_minus_sign():
                canonical = -canonical
    
            metrics['canonical'] += sympy.count_ops(canonical)
        except:
            pass

    return metrics


# --- Step 2: Discover Candidate Functions with PySR ---

def discover_basis_functions(X, X_dot_target, feature_names, n_chunks=10, chunk_size_divisor=3, pysr_params=None):
    """
    Runs PySR on random data chunks to find recurring functional forms for a specific target.
    This is the core "discovery" engine of the framework.
    """
    n_samples, n_features = X.shape
    chunk_size = int(n_samples / chunk_size_divisor)

    raw_function_strings = set()

    print(f"\n> Step 2: Running PySR on {n_chunks} data chunks...")

    # Loop over the specified number of chunks
    for i in range(n_chunks):
        # Select a random subset of the data for this chunk
        indices = np.random.choice(n_samples, chunk_size, replace=False)
        print(f"\n> CHECK RANDOM {indices[0]} ...")

        X_chunk, X_dot_chunk = X[indices], X_dot_target[indices]

        # Loop over all target state variables (this will be 1 for separate, n_features for unified)
        for target_idx in range(X_dot_target.shape[1]):
            # PySR requires a pandas DataFrame to correctly map feature names
            X_chunk_df = pd.DataFrame(X_chunk, columns=feature_names)

            # Initialize and run the PySR model
            model = PySRRegressor(**pysr_params, verbosity=0)
            model.fit(X_chunk_df, X_dot_chunk[:, target_idx])

            # After the search, collect all equations found
            if hasattr(model, 'equations_') and len(model.equations_) > 0:
                print(f"    --- Chunk {i+1}, Target Var {target_idx} Results ---")
                for eq in model.equations_.itertuples():
                    print(f"        Complexity: {eq.complexity:<2} | Loss: {eq.loss:<.4f} | Score: {eq.score:<.6f} | Eq: {str(eq.sympy_format)}")
                    raw_function_strings.add(str(eq.sympy_format))

        print(f"  ...Chunk {i+1}/{n_chunks} complete.")

    print(f"✓ Discovered {len(raw_function_strings)} total raw symbolic expressions.")
    return sorted(list(raw_function_strings))


# --- Step 3: Curate the Discovered Library ---

def curate_library(function_strings, X, feature_names, curation_config):
    """
    Curates the library based on the expansion strategy defined in config.
    Includes full logging of pruned terms.
    """
    # Load settings from config
    expansion_strategy = curation_config.get('expansion_strategy', 'hybrid')
    pruning_method = curation_config.get('pruning_method', 'vif')
    if pruning_method == 'vif':
        # Load VIF threshold (default 10.0)
        threshold = curation_config.get('vif_threshold', 10.0)
        print(f"\n> Step 3: Curation (Strategy: '{expansion_strategy}', Method: 'VIF', Threshold: {threshold})")
    else:
        # Load Correlation threshold (default 0.99)
        threshold = curation_config.get('correlation_threshold', 0.99)
        print(f"\n> Step 3: Curation (Strategy: '{expansion_strategy}', Method: 'Correlation', Threshold: {threshold})")    
    
    print(f"  Starting with {len(function_strings)} raw expressions.")
    
    atomic_expressions = set()

    # --- 1. DECOMPOSITION ---
    for s in function_strings:
        try:
            raw_expr = sympy.sympify(s)
            expressions_to_process = []

            # Hybrid is usually best: gives SINDy both the block (x+y)^2 and pieces x^2, y^2
            if expansion_strategy == 'hybrid':
                expressions_to_process.append(sympy.expand(raw_expr, multinomial=False)) # Gentle
                severe = sympy.expand(raw_expr)
                severe = sympy.expand_trig(severe)
                expressions_to_process.append(severe) # Severe
            elif expansion_strategy == 'gentle':
                expressions_to_process.append(sympy.expand(raw_expr, multinomial=False))
            elif expansion_strategy == 'severe':
                severe = sympy.expand(raw_expr)
                severe = sympy.expand_trig(severe)
                severe = sympy.expand(severe)
                expressions_to_process.append(severe)
            elif expansion_strategy == 'none':
                expressions_to_process.append(raw_expr)

            for expr in expressions_to_process:
                terms = expr.as_ordered_terms() if isinstance(expr, sympy.Add) else [expr]
                for term in terms:
                    term_no_coeffs = term
                    if term.is_Mul:
                        non_numeric_args = [arg for arg in term.args if not arg.is_Number]
                        if not non_numeric_args: term_no_coeffs = sympy.S.One
                        else: term_no_coeffs = sympy.Mul(*non_numeric_args)

                    if term_no_coeffs.is_Number: continue
                    atomic_expressions.add(term_no_coeffs)
        except: 
            pass

    print(f"  Decomposed into {len(atomic_expressions)} unique atomic expressions.")

    # --- 2. SORT BY COMPLEXITY ---
    def get_complexity(expr):
        return sympy.count_ops(expr)

    print(f"\n  Pruning for collinearity with simplicity bias...")
    symbols = sympy.symbols(feature_names)
    sorted_expressions = sorted(list(atomic_expressions), key=get_complexity)

    kept_expressions = []
    kept_features = [] 

    # Helper to evaluate expression safely
    def evaluate_expr(expr, X_data):
        try:
            f = sympy.lambdify(symbols, expr, 'numpy')
            res = f(*X_data.T)
            if isinstance(res, (int, float)):
                res = np.full(X_data.shape[0], res)
            return res
        except:
            return None

    # --- 3. PRUNING LOOP ---
    for expr in sorted_expressions:
        new_feature = evaluate_expr(expr, X)
        
        if new_feature is None or np.std(new_feature) < 1e-9:
            print(f"    - Discarding '{expr}' (constant or error).")
            continue
            
        should_discard = False

        if len(kept_features) > 0:
            if pruning_method == 'vif':
                # --- OPTION A: VIF / LINEAR DEPENDENCE ---
                X_existing = np.column_stack(kept_features)
                reg = LinearRegression().fit(X_existing, new_feature)
                r_squared = reg.score(X_existing, new_feature)
                
                # VIF = 1 / (1 - R^2)
                if r_squared > 0.9999999:
                    implied_vif = float('inf')
                else:
                    implied_vif = 1.0 / (1.0 - r_squared)
                
                if implied_vif > threshold:
                    print(f"    - Discarding '{expr}' (VIF={implied_vif:.2f} > {threshold}). Redundant.")
                    should_discard = True

            else:
                # --- OPTION B: PAIRWISE CORRELATION ---
                for i, existing_feat in enumerate(kept_features):
                    corr = abs(np.corrcoef(new_feature, existing_feat.flatten())[0, 1])
                    if corr > threshold:
                        prev_expr = kept_expressions[i]
                        print(f"    - Discarding '{expr}' (corr={corr:.3f} with '{prev_expr}').")
                        should_discard = True
                        break

        if not should_discard:
            kept_expressions.append(expr)
            kept_features.append(new_feature)

    callable_functions = [sympy.lambdify(symbols, expr) for expr in kept_expressions]
    function_names_list = [str(expr) for expr in kept_expressions]

    print("\n  Final curated library of atomic expressions:")
    for name in function_names_list: print(f"    {name}")
    print(f"✓ Curated to {len(callable_functions)} unique functions.")
    
    return callable_functions, function_names_list

# --- Step 4: Run SINDy and Evaluate the Model ---

# === NEW FACTORY FUNCTION FOR OPTIMIZERS ===
def get_sindy_optimizer(optimizer_config, n_samples):
    """
    Factory function to create a PySINDy optimizer based on config.
    Handles STLSQ, SR3, and Ensemble wrapping.
    
    Args:
        optimizer_config (dict): The dictionary from config['optimizer_params']
        n_samples (int): Used to calculate bagging subset size.
    """
    name = optimizer_config.get("name", "STLSQ")
    threshold = optimizer_config.get("threshold", 0.1)
    
    # 1. Instantiate the Base Optimizer
    if name == "SR3":
        print(f"  > Configuring SR3 Optimizer (threshold={threshold})...")
        base_opt = ps.SR3(
            threshold=threshold, 
            nu=optimizer_config.get("sr3_nu", 1.0),
            max_iter=optimizer_config.get("sr3_max_iter", 30)
        )
    elif name == "STLSQ":
        print(f"  > Configuring STLSQ Optimizer (threshold={threshold})...")
        base_opt = ps.STLSQ(threshold=threshold, alpha=0.05)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

    # 2. Check for Ensemble requirement
    if optimizer_config.get("use_ensemble", False):
        n_models = optimizer_config.get("n_models", 10)
        fraction = optimizer_config.get("bagging_fraction", 0.6)
        n_subset = int(n_samples * fraction)
        
        print(f"  > Wrapping in ENSEMBLE Optimizer ({n_models} bags, {n_subset} samples/bag)...")
        final_opt = ps.EnsembleOptimizer(
            opt=base_opt,
            bagging=True,
            n_models=n_models,
            n_subset=n_subset,
            replace=False
        )
        return final_opt
    
    return base_opt
# ===========================================


# # === NEW HELPER: Apply Ensemble Filtering Logic ===
# def apply_ensemble_filter(model, optimizer_config):
#     """
#     Applies the 'Hard Cut' filter to an ensemble model, handles 1D/2D shape 
#     inconsistencies, and conditionally aggregates the coefficients ONLY from 
#     models where they were included.
#     """
#     if not optimizer_config.get("use_ensemble", False):
#         return  # early exit if no ensemble

#     print("  > Applying Robust Ensemble Filtering (Conditional Aggregation)...")
    
#     # 1. Get the history of all bootstrap models
#     coef_list = np.array(model.optimizer.coef_list)
#     n_models = coef_list.shape[0]
#     print(f"  coef_list.shape {coef_list.shape}")
#     # --- FIX 1: Normalize to always be 3D (n_models, n_targets, n_features) ---
#     if coef_list.ndim == 2:
#         coef_list = coef_list[:, np.newaxis, :]  
    
#     n_targets = coef_list.shape[1]
#     n_features = coef_list.shape[2]
    
#     # 2. Calculate Inclusion Probabilities
#     inclusion_counts = np.count_nonzero(np.abs(coef_list) > 1e-10, axis=0)
#     inclusion_probs = inclusion_counts / n_models
    
#     # 3. Create the Mask
#     cut_off = optimizer_config.get("inclusion_cut_off", 0.8)
#     robust_mask = inclusion_probs >= cut_off
    
#     # --- FIX 2: Calculate Conditional Median ---
#     new_coef = np.zeros((n_targets, n_features))
    
#     for i in range(n_targets):
#         for j in range(n_features):
#             if robust_mask[i, j]:
#                 # Extract this specific term across all ensemble models
#                 term_values = coef_list[:, i, j]
#                 # Filter out the zeros (only keep where it was actually included)
#                 included_values = term_values[np.abs(term_values) > 1e-10]
                
#                 # Take the median of ONLY the included values
#                 if len(included_values) > 0:
#                     new_coef[i, j] = np.median(included_values)

#     # 4. Report what we are doing
#     print(f"  new_coef.shape {new_coef.shape}")
#     n_kept = np.sum(robust_mask)
#     n_total = robust_mask.size
#     print(f"    - Cut-off: {cut_off*100:.0f}%. Kept {n_kept}/{n_total} terms.")

#     # 5. Overwrite PySINDy's default coefficients
#     # We reshape it back to whatever original shape PySINDy was expecting 
#     # to prevent downstream shape mismatch errors in the separate-library case.
#     # original_shape = model.optimizer.coef_.shape
#     # model.optimizer.coef_ = new_coef.reshape(original_shape)
#     # model.optimizer.coef_ = new_coef

        
    # =========================================================
    

def fit_sindy_model(X_train, X_dot_train, t_train, curated_functions, curated_function_names, feature_names, optimizer_config):
    """
    Fits a SINDy model using the custom library and CONFIGURABLE optimizer.
    NOW INCLUDES: Robust Ensemble Filtering (Hard Cut-off).
    """
    print("\n> Step 4: Fitting SINDy model...")

    constant_library = ps.PolynomialLibrary(degree=0, include_bias=True)
    custom_library = ps.CustomLibrary(
        library_functions=curated_functions,
        function_names=[(lambda name: lambda *x: name)(name) for name in curated_function_names]
    )

    full_library = constant_library + custom_library

    sindy_library_names = ['1'] + curated_function_names
    print(f"  SINDy is using a library with {len(sindy_library_names)} total features: {sindy_library_names}")

    # === USE THE FACTORY FUNCTION ===
    optimizer = get_sindy_optimizer(optimizer_config, n_samples=X_train.shape[0])

    model = ps.SINDy(feature_library=full_library, optimizer=optimizer, feature_names=feature_names)

    # Fit the model (quiet=True to suppress ensemble noise)
    model.fit(X_train, t=t_train, x_dot=X_dot_train, quiet=True)
    
    # apply_ensemble_filter(model, optimizer_config)



    print("✓ SINDy model fitting complete.")
    return model

def evaluate_and_print_performance(model, title, X_train, X_dot_train, t_train, X_test, X_dot_test, t_test, X_new, X_dot_new, t_new, sim_scores=None):
    """
    Calculates performance scores, prints them, and returns them in a dictionary.
    """
    print("\n" + "="*80)
    print(f"||{title.center(78)}||")
    print("="*80)

    # --- Standard Prediction Scores ---
    train_score_r2 = model.score(X_train, t=t_train, x_dot=X_dot_train)
    train_score_mse = mean_squared_error(X_dot_train, model.predict(X_train))

    test_score_r2 = model.score(X_test, t=t_test, x_dot=X_dot_test)
    test_score_mse = mean_squared_error(X_dot_test, model.predict(X_test))

    new_traj_score_r2 = model.score(X_new, t=t_new, x_dot=X_dot_new)
    new_traj_score_mse = mean_squared_error(X_dot_new, model.predict(X_new))
    
    # --- Complexity Metrics ---
    complexities = calculate_complexity_metrics(model)

    # --- Print Report ---
    print(f"  Complexity (Structural): {complexities['structural']} ops")
    print(f"  Complexity (Canonical):  {complexities['canonical']} ops (Expanded)")
    print("-" * 80)
    print("  Derivative Prediction Performance:")
    print(f"  Training Set:   R^2 = {train_score_r2:<.6f}  |  MSE = {train_score_mse:<.6f}")
    print(f"  Test Set:       R^2 = {test_score_r2:<.6f}  |  MSE = {test_score_mse:<.6f}")
    print(f"  New Trajectory: R^2 = {new_traj_score_r2:<.6f}  |  MSE = {new_traj_score_mse:<.6f}")

    if sim_scores and not np.isnan(sim_scores.get('r2', np.nan)):
        print("-" * 80)
        print("  Simulation Performance (State Variables):")
        print(f"  Simulation:     R^2 = {sim_scores['r2']:<.6f}  |  MSE = {sim_scores['mse']:<.6f}")

    print("="*80)

    # Return ALL metrics for logging
    performance_data = {
        'train_r2': train_score_r2, 'train_mse': train_score_mse,
        'test_r2': test_score_r2, 'test_mse': test_score_mse,
        'new_traj_r2': new_traj_score_r2, 'new_traj_mse': new_traj_score_mse,
        'sim_r2': sim_scores.get('r2', np.nan) if sim_scores else np.nan,
        'sim_mse': sim_scores.get('mse', np.nan) if sim_scores else np.nan,
        'structural_complexity': complexities['structural'],
        'canonical_complexity': complexities['canonical']
    }
    return performance_data

# ==========================================================================
# ===== CENTRALIZED CONFIGURATION DICTIONARY =====
# All hyperparameters are defined here for easy tracking and modification.
# ==========================================================================
config = {
    "global_seed": GLOBAL_SEED,
    "system_to_run": 'complex_lorenz', # Options: 'harmonic_oscillator', 'vanderpol', 'damped_pendulum', 'lorenz', 'duffing', 'michaelis_menten', 'modulated_oscillator', 'exponential_system'

    "data_params": {
        'harmonic_oscillator': {'k1': 5.0, 'k2': 1.0, 'x0': [1.0, 0.0], 't_end': 10, 'n_samples': 5000, 'noise_level': 0.05, 'noise_seed': GLOBAL_SEED, 'new_trajectory_params': {'x0': [0.5, 0.5], 't_end': 15, 'n_samples': 7500}},
        'vanderpol': {'mu': 2.0, 'x0': [0.5, 0.5], 't_end': 25, 'n_samples': 5000, 'noise_level': 0.1, 'noise_seed': GLOBAL_SEED, 'new_trajectory_params': {'x0': [2.0, -1.0], 't_end': 30, 'n_samples': 6000}},
        'damped_pendulum': {'b': 0.25, 'c': 5.0, 'x0': [np.pi - 0.1, 0], 't_end': 10, 'n_samples': 5000, 'noise_level': 0.01, 'noise_seed': GLOBAL_SEED, 'new_trajectory_params': {'x0': [np.pi - 0.5, 0.2], 't_end': 12, 'n_samples': 6000}},
        'lorenz': {'sigma': 10.0, 'rho': 28.0, 'beta': 8./3., 'x0': [-8.0, 8.0, 27.0], 't_end': 20, 'n_samples': 5000, 'noise_level': 0.05, 'noise_seed': GLOBAL_SEED, 'new_trajectory_params': {'x0': [10, 10, 20], 't_end': 20, 'n_samples': 6000}},
        'complex_lorenz': {'sigma': 10.0, 'rho': 28.0, 'beta': 8./3., 'gamma': 1.5, 'x0': [-8.0, 8.0, 27.0], 't_end': 20, 'n_samples': 5000, 'noise_level': 0.04, 'noise_seed': GLOBAL_SEED, 'new_trajectory_params': {'x0': [10, 10, 20], 't_end': 20, 'n_samples': 6000}},
        'duffing': {'delta': 0.3, 'alpha': -1.0, 'beta': 1.0, 'x0': [0.5, 0.5], 't_end': 15, 'n_samples': 5000, 'noise_level': 0.02, 'noise_seed': GLOBAL_SEED, 'new_trajectory_params': {'x0': [1.0, -1.0], 't_end': 12, 'n_samples': 5500}},
        'michaelis_menten': {'vmax': 1.5, 'km': 0.3, 'x0': [1.0], 't_end': 10, 'n_samples': 5000, 'noise_level': 0.01, 'noise_seed': GLOBAL_SEED, 'new_trajectory_params': {'x0': [1.5], 't_end': 12, 'n_samples': 6000}},
        'modulated_oscillator': {'b': 0.25, 'k': 5.0, 'x0': [1.0, 0.0], 't_end': 10, 'n_samples': 5000, 'noise_level': 0.1, 'noise_seed': GLOBAL_SEED, 'new_trajectory_params': {'x0': [0.8, 0.2], 't_end': 15, 'n_samples': 7500}},
        'exponential_system': {'a': 0.5, 'b': 0.5, 'x0': [0.5, 0.5], 't_end': 10, 'n_samples': 5000, 'noise_level': 0.01, 'noise_seed': GLOBAL_SEED, 'new_trajectory_params': {'x0': [0.7, 0.3], 't_end': 12, 'n_samples': 6000}},
    },

    # --- Optimizer Configuration (NEW) ---
    "optimizer_params": {
        "name": "STLSQ",         # Options: 'STLSQ', 'SR3'
        "threshold": 0.21,      # The lambda (sparsity threshold) # for STLSQ : 0.21 # for Sr3 : 0.1 or 0.21
        
        # SR3 Specifics
        "sr3_nu": 1.0,         # Relaxation parameter (usually 0.1 to 10)
        "sr3_max_iter": 30,    # Iterations for SR3
        
        # Ensemble Specifics
        "use_ensemble": True,  # True = Wrap optimizer in Ensemble (Bagging)
        "n_models": 20,        # Number of models in ensemble (bags)
        "bagging_fraction": 1, # Portion of data used per model
        "inclusion_cut_off": 0.80, # Only keep terms found in >80% of bootstraps
    },

    "use_unified_library": False, # True = Pool all terms; False = Separate libraries per variable

    "pysr_params": {
        "niterations": 30,
        "binary_operators": ["+", "-", "*"],
        "unary_operators": ["sin", "cos", "square", "cube"], #, "exp" "sin", "cos"
        "model_selection": "best",
        "maxsize": 8, # previously 10
        "populations": 16,
        "procs": 4,
        "temp_equation_file": True,
        "nested_constraints": {"sin": {"sin": 0, "cos": 0}, "cos": {"sin": 0, "cos": 0}},
        "random_state": GLOBAL_SEED,      # Your fixed seed
        # "parallelism": "serial", # Replaces 'procs' to strictly enforce a single thread
        # "deterministic": True,   # Enforces strict determinism 
        # 
    },

    # Simplified performance settings
    "train_test_split_ratio": 0.8,
    "split_strategy": 'middle',

    "discovery_chunks": 3,
    "chunk_size_divisor": 10,
    "curation_params": {
        # Options: 
        # 'severe'  -> Fully expands everything (x+y)^2 becomes x^2 + 2xy + y^2
        # 'gentle'  -> Keeps powers intact (x+y)^2 stays (x+y)^2
        # 'hybrid'  -> Adds BOTH severe and gentle versions (Best for discovery)
        # 'none'    -> No expansion, keeps raw PySR output
        "expansion_strategy": "gentle", 
        
        "pruning_method": "correlation",   # Options: 'vif' or 'correlation'
        "vif_threshold": 10.0,            # Standard for VIF , 10 for VIF # Threshold 10.0 implies R^2 > 0.9. 
        # Increase to 20.0 or 50.0 if you want to allow more similar terms.
        "correlation_threshold": 0.985 #0.998, 0.985 for correlation
    },
    
    # Note: 'sindy_threshold' is now handled inside "optimizer_params" above, but we keep this
    # just for the Standard SINDy baseline comparison.
    "sindy_threshold": 0.21, 
    "sindy_poly_degree": 3, # 3
    "sindy_fourier_freqs": 1, # 3

    "simulation_params": {
        "t_sim_end": 5,
        "n_sim_samples": 500,
        "models_to_simulate": ['AutoSINDy', 'Standard SINDy', 'Standard PySR'] # Options: 'AutoSINDy', 'Standard SINDy', 'Standard PySR'
    },

    "results_csv_file": "autosindy_results_log_tidy.csv"
}
# ==========================================================================


# --- Plotting Functions ---

def plot_derivative_comparison(t, X, X_dot, t_train, models_dict, feature_names, split_strategy, run_tag="default"):
    """
    Plots true derivatives vs predictions.
    UPDATED: Y-axis limits are now fixed to the True Derivative's range to ignore exploding predictions.
    """
    n_features = X.shape[1]
    fig, axs = plt.subplots(n_features, 1, figsize=(12, 3 * n_features), sharex=True, dpi=300)

    if n_features == 1:
        axs = [axs]

    fig.suptitle('Derivative Comparison (Train/Test Trajectory)', fontsize=16)

    # Styles
    styles = {
        'AutoSINDy': {'color': 'red', 'linestyle': '--', 'alpha': 0.8, 'zorder': 4, 'linewidth': 3.5},
        'Standard SINDy': {'color': 'green', 'linestyle': ':', 'alpha': 0.8, 'zorder': 3, 'linewidth': 3.5},
        'Standard PySR': {'color': 'blue', 'linestyle': '-.', 'alpha': 0.8, 'zorder': 2, 'linewidth': 3.5}
    }

    for i in range(n_features):
        # 1. Get True Derivative Data
        true_deriv = X_dot[:, i]
        
        # 2. Plot True Derivative
        axs[i].plot(t, true_deriv, color='gray', label='True Derivative', linewidth=4.0, zorder=1)

        # 3. Plot Predictions
        for name, model in models_dict.items():
            x_dot_pred = model.predict(X)
            style = styles.get(name, {'color': 'gray', 'linestyle': '--'})
            pred_to_plot = x_dot_pred[:, i] if x_dot_pred.ndim > 1 else x_dot_pred
            axs[i].plot(t, pred_to_plot, label=name, **style)

        axs[i].set_ylabel(f'$d{feature_names[i]}/dt$')

        # --- KEY UPDATE: Set Y-Limits based on True Derivative ---
        y_min, y_max = np.min(true_deriv), np.max(true_deriv)
        y_range = y_max - y_min
        
        # Add a 20% margin (handle flat lines where range is 0)
        if y_range == 0: 
            margin = 1.0 
        else: 
            margin = y_range * 0.2

        axs[i].set_ylim(y_min - margin, y_max + margin)
        # ---------------------------------------------------------

        # Add split markers
        if split_strategy == 'end':
            axs[i].axvline(t_train[-1], color='k', linestyle='--', lw=1.5, alpha=0.5, label='Train/Test Split')
        elif split_strategy == 'middle':
            axs[i].axvline(t_train[0], color='k', linestyle='--', lw=1.5, alpha=0.5, label='Train Start')
            axs[i].axvline(t_train[-1], color='k', linestyle='--', lw=1.5, alpha=0.5, label='Train End')

    # Legend handling
    handles, labels = axs[0].get_legend_handles_labels()
    # Remove duplicates in legend caused by the loop
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper right', framealpha=0.9)
    
    axs[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.savefig(f'figures/fig_derivatives_{split_strategy}_{run_tag}.svg', dpi=300, bbox_inches='tight')  # ← ADD THIS
    plt.show()

def plot_new_derivative_comparison(t, X, X_dot, models_dict, feature_names, run_tag="default"):
    """
    Plots new true derivatives vs predictions.
    UPDATED: Y-axis limits fixed to True Derivative range.
    """
    n_features = X.shape[1]
    fig, axs = plt.subplots(n_features, 1, figsize=(12, 3 * n_features), sharex=True, dpi=300)

    if n_features == 1:
        axs = [axs]

    fig.suptitle('Derivative Comparison (New Trajectory)', fontsize=16)

    styles = {
        'AutoSINDy': {'color': 'red', 'linestyle': ':', 'alpha': 0.8, 'zorder': 4, 'linewidth': 3.5},
        'Standard SINDy': {'color': 'green', 'linestyle': '-.', 'alpha': 0.8, 'zorder': 3, 'linewidth': 3.5},
        'Standard PySR': {'color': 'blue', 'linestyle': '--', 'alpha': 0.8, 'zorder': 2, 'linewidth': 3.5}
    }

    for i in range(n_features):
        true_deriv = X_dot[:, i]
        
        # Plot True
        axs[i].plot(t, true_deriv, color='gray', label='True Derivative', linewidth=6.0, alpha = 0.5, zorder=1)

        # Plot Predictions
        for name, model in models_dict.items():
            x_dot_pred = model.predict(X)
            style = styles.get(name, {'color': 'gray', 'linestyle': '--'})
            pred_to_plot = x_dot_pred[:, i] if x_dot_pred.ndim > 1 else x_dot_pred
            axs[i].plot(t, pred_to_plot, label=name, **style)

        axs[i].set_ylabel(f'$d{feature_names[i]}/dt$')

        # --- KEY UPDATE: Set Y-Limits based on True Derivative ---
        y_min, y_max = np.min(true_deriv), np.max(true_deriv)
        y_range = y_max - y_min
        
        if y_range == 0: 
            margin = 1.0 
        else: 
            margin = y_range * 0.2

        axs[i].set_ylim(y_min - margin, y_max + margin)
        # ---------------------------------------------------------

    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper right', framealpha=0.9)
    
    axs[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.savefig(f'figures/fig_derivatives_validation_{run_tag}.svg', dpi=300, bbox_inches='tight')  # ← ADD THIS
    plt.show()
    
def simulate_with_timeout(model, x0, t_sim, integrator_kws, timeout_seconds=200):
    """
    Runs model.simulate() in a thread. Returns result, or None if it times out.
    Prevents any single simulation from hanging the entire sweep.
    """
    result = [None]
    exception = [None]

    def target():
        try:
            result[0] = model.simulate(x0, t=t_sim, integrator_kws=integrator_kws)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        return None, TimeoutError(f"Simulation exceeded {timeout_seconds}s wall-clock limit")
    if exception[0] is not None:
        return None, exception[0]
    return result[0], None

def plot_simulation_comparison(t_sim, X_true, models_dict, feature_names, run_tag="default"):
    """
    Plots the true trajectory, compares it with simulations, and returns simulation scores.
    Includes a 'Safety Brake' to stop unstable simulations quickly.
    """
    n_features = X_true.shape[1]
    
    # --- 1. Setup Time Series Figure ---
    fig, axs = plt.subplots(n_features, 1, figsize=(12, 2 * n_features), sharex=True, dpi=300)
    if n_features == 1:
        axs = [axs]
    fig.suptitle('State Variable Simulation Comparison', fontsize=16)

    styles = {
        'AutoSINDy': {'color': 'red', 'linestyle': '--', 'alpha': 0.8, 'zorder': 4, 'linewidth': 3.5},
        'Standard SINDy': {'color': 'green', 'linestyle': ':', 'alpha': 0.8, 'zorder': 3, 'linewidth': 3.5},
        'Standard PySR': {'color': 'blue', 'linestyle': '-.', 'alpha': 0.8, 'zorder': 2, 'linewidth': 3.5}
    }

    # Plot true trajectory (Time Series)
    for i in range(n_features):
        axs[i].plot(t_sim, X_true[:, i], color='black', label='True Trajectory', linewidth=6.0, alpha = 0.4, zorder=1)

    # --- SAFETY BRAKE FUNCTION ---
    limit_val = np.max(np.abs(X_true)) * 100
    def divergence_check(t, y):
        return limit_val - np.max(np.abs(y))
    divergence_check.terminal = True 
    # -----------------------------

    simulation_times = {}
    simulation_scores = {}
    
    # NEW: Dictionary to store successful simulations for the Phase Plot later
    stored_results = {} 

    # --- 2. Main Simulation Loop ---
    for name, model in models_dict.items():
        print(f"\n> Simulating model: {name}")
        start_time = time.time()
        x_sim = None
        
        try:
            x_sim, sim_error = simulate_with_timeout(
                model, X_true[0], t_sim,
                integrator_kws={'method': 'Radau', 'events': divergence_check, 
                                'max_step': (t_sim[-1] - t_sim[0]) / 100,   # coarse cap
                                'rtol': 1e-4,   # loosen — you don't need precision on a diverging trajectory
                                'atol': 1e-6,},
                timeout_seconds=200   # 1 minutes max per model per run
            )
            if sim_error is not None:
                raise sim_error  # caught by the existing except block below

            end_time = time.time()
            simulation_times[name] = end_time - start_time
            print(f"  Simulation time: {simulation_times[name]:.4f} seconds")

            # VALIDATION checks
            if x_sim.shape[0] != X_true.shape[0]:
                print("  ! Instability detected: Simulation stopped early (values exploded).")
                simulation_scores[name] = {'mse': np.nan, 'r2': np.nan}
            elif np.isnan(x_sim).any() or np.isinf(x_sim).any():
                print("  ! Instability detected: Simulation contains NaN/Inf.")
                simulation_scores[name] = {'mse': np.nan, 'r2': np.nan}
            else:
                # SUCCESS CASE
                # A. Plot on Time Series
                style = styles.get(name, {'color': 'gray', 'linestyle': '--'})
                for i in range(n_features):
                    axs[i].plot(t_sim, x_sim[:, i], label=name, **style)
                
                # B. Calculate Scores
                mse = mean_squared_error(X_true, x_sim)
                r2 = r2_score(X_true, x_sim)
                simulation_scores[name] = {'mse': mse, 'r2': r2}
                print(f"  Simulation scores: R^2={r2:.4f}, MSE={mse:.4f}")
                
                # C. Store for Phase Plot (NEW STEP)
                stored_results[name] = x_sim

        except Exception as e:
            end_time = time.time()
            simulation_times[name] = end_time - start_time
            print(f"  ! Simulation crashed for {name}: {e}")
            simulation_scores[name] = {'mse': np.nan, 'r2': np.nan}

    # Finalize Time Series Plot
    for i in range(n_features):
        axs[i].set_ylabel(f'${feature_names[i]}$')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', framealpha=0.5)
    axs[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.savefig(f'figures/fig_simulation_timeseries_{run_tag}.svg', dpi=300, bbox_inches='tight')  # ← ADD THIS
    plt.show() # Show the first figure
    
    # --- 3. Phase Portrait Plot (Uses stored results) ---
    if X_true.shape[1] >= 2:
        plt.figure(figsize=(8, 8), dpi=300)
        plt.title(f"Phase Portrait ({feature_names[0]} vs {feature_names[1]})")
        
        # Plot Truth
        plt.plot(X_true[:, 0], X_true[:, 1], 'k-', label='True/Underlying model', linewidth=7.0, alpha=0.3)
        
        # Plot Models (Iterating over the STORED results, not re-simulating)
        for name, x_sim in stored_results.items():
             style = styles.get(name, {'color': 'gray', 'linestyle': '--'})
             plt.plot(x_sim[:, 0], x_sim[:, 1], label=name, **style)
             
        print(" Phase Portrait is about to be Plotted")
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figures/fig_phase_portrait_{run_tag}.svg', dpi=300, bbox_inches='tight')  # ← ADD THIS
        plt.show() # Show the second figure
        print(" Phase Portrait is Plotted")

    # --- 4. Print Times ---
    print("\n" + "="*80)
    print("||                    RESULTS: Simulation Times                      ||")
    print("="*80)
    for name, sim_time in simulation_times.items():
        print(f"{name}: {sim_time:.4f} seconds")
    print("="*80)

    return simulation_scores, simulation_times

# --- Model Wrapper Classes ---

class SeparateSINDyModel:
    """
    A wrapper to unify a list of single-output SINDy models into a single
    multi-output model, making it compatible with PySINDy's simulation
    and evaluation functions.
    """
    def __init__(self, models, feature_names):
        self.models = models
        self.feature_names = feature_names
        self.complexity = sum(model.complexity for model in self.models)

    def predict(self, x):
        """Predicts derivatives by combining predictions from each model."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        predictions = [np.asarray(model.predict(x)) for model in self.models]
        return np.hstack(predictions)

    def simulate(self, x0, t, integrator_kws=None, **kwargs):
        """Simulates the system's trajectory."""
        def ode_func(t, x):
            x_reshaped = x.reshape(1, -1)
            return self.predict(x_reshaped).flatten()

        kws = integrator_kws if integrator_kws is not None else {}
        sol = solve_ivp(ode_func, (t[0], t[-1]), x0, t_eval=t, **kws)
        if sol.status != 0:
            print(f"Warning: Simulation integration failed with status {sol.status}: {sol.message}")
            return np.full((len(t), len(x0)), np.nan)
        return sol.y.T

    def score(self, x, t=None, x_dot=None, u=None):
        """Calculates the R^2 score for the unified model."""
        if x_dot is None:
            raise ValueError("x_dot must be provided to compute the score.")
        return r2_score(x_dot, self.predict(x))

    # === METHOD THAT WAS MISSING ===
    def equations(self, **kwargs):
        """Returns a list of equations from each sub-model."""
        return [model.equations(**kwargs)[0] for model in self.models]
    # ===============================

    def print(self, precision=10):
        """Prints the equations for each sub-model."""
        for i, model in enumerate(self.models):
            print(f"(x{i})' = {model.equations(precision=precision)[0]}")

class PySRWrapper:
    """A wrapper for a list of PySR models to make them act like a single multi-output model."""
    def __init__(self, models, feature_names):
        self.models = models
        self.feature_names = feature_names
        self.complexity = 0
        for model_j in self.models:
            best_eq_str = str(model_j.get_best().get('sympy_format', ''))
            try:
                self.complexity += sympy.count_ops(sympy.sympify(best_eq_str))
            except (SyntaxError, TypeError):
                pass

    def predict(self, x):
        """Predicts derivatives by combining predictions from each PySR model."""
        if not isinstance(x, pd.DataFrame):
            x_df = pd.DataFrame(x, columns=self.feature_names)
        else:
            x_df = x
        predictions = [model.predict(x_df).reshape(-1, 1) for model in self.models]
        return np.hstack(predictions)

    def simulate(self, x0, t, integrator_kws=None, **kwargs):
        """Simulates the system's trajectory using the PySR equations."""
        def ode_func(t, x):
            x_df = pd.DataFrame([x], columns=self.feature_names)
            return self.predict(x_df).flatten()

        kws = integrator_kws if integrator_kws is not None else {}
        sol = solve_ivp(ode_func, (t[0], t[-1]), x0, t_eval=t, dense_output=True, **kws)
        if sol.status != 0:
            print(f"Warning: Simulation integration failed with status {sol.status}: {sol.message}")
            return np.full((len(t), len(x0)), np.nan)
        return sol.y.T

    def score(self, x, t=None, x_dot=None, u=None):
        """Calculates the R^2 score for the unified PySR model."""
        if x_dot is None:
            raise ValueError("x_dot must be provided to compute the score.")
        return r2_score(x_dot, self.predict(x))

    def equations(self, **kwargs):
        return [model.equations(**kwargs)[0] for model in self.models]


    def print(self, precision=10):
        """Prints the best equation found by each PySR model."""
        for i, model in enumerate(self.models):
            best_eq = model.get_best()
            print(f"(x{i})' = {str(best_eq.get('sympy_format', 'No equation found'))}")



# ==========================================================================
# 2. RUN EXPERIMENT FUNCTION
# ==========================================================================
def run_experiment(cfg=None):
    """
    Main logic to run one full experiment based on the current global 'config'.
    UPDATED: 
    - Training data retains noise (to test robustness).
    - New Trajectory & Simulation data are forced to be CLEAN (to test physical accuracy).
    """
    if cfg is None:
        cfg = config
        

    np.random.seed(cfg.get("global_seed", GLOBAL_SEED))
    system_to_run = cfg["system_to_run"]
    noise_level   = cfg["data_params"][system_to_run]["noise_level"]
    run_tag       = f"{system_to_run}_noise{noise_level:.3f}_seed{cfg.get('global_seed', 42)}"
    os.makedirs("figures", exist_ok=True)
    
    print("\n" + "#"*80)
    print(f"##### RUNNING TEST ON: {system_to_run.upper()} SYSTEM #####")
    print("#"*80)

    # --- Main Execution Block ---

    # --- Generate Primary Data (NOISY - For Discovery/Training) ---
    system_map = {
        'harmonic_oscillator': systems.generate_harmonic_oscillator_data,
        'vanderpol': systems.generate_vanderpol_data,
        'damped_pendulum': systems.generate_damped_pendulum_data,
        'lorenz': systems.generate_lorenz_data,
        'complex_lorenz': systems.generate_complex_lorenz_data,
        'duffing': systems.generate_duffing_data,
        'michaelis_menten': systems.generate_michaelis_menten_data,
        'modulated_oscillator': systems.generate_modulated_oscillator_data,
        'exponential_system': systems.generate_exponential_system_data,
    }

    if system_to_run not in system_map:
        raise ValueError(f"Invalid system specified in config: {system_to_run}")

    system_generator = system_map[system_to_run]
    params = cfg["data_params"][system_to_run]
    
    # 1. Generate Training Data (Uses config noise_level)
    print(f"\n> Step 1a: Generating Training Data (Noise Level: {params['noise_level']})...")
    X, X_dot, t = system_generator(**{k: v for k, v in params.items() if k != 'new_trajectory_params'})

    n_features = X.shape[1]
    feature_names = [f'x{i}' for i in range(n_features)]

    # --- Data Splitting and Preparation ---
    print("\n> Step 1b: Splitting data for performance evaluation...")
    X_train, X_dot_train, t_train, X_test, X_dot_test, t_test = split_data(
        X, X_dot, t,
        train_ratio=cfg["train_test_split_ratio"],
        strategy=cfg["split_strategy"]
    )
    print(f"  Train data shape: X={X_train.shape}, X_dot={X_dot_train.shape}")
    print(f"  Test data shape:  X={X_test.shape}, X_dot={X_dot_test.shape}")

    # --- Generate New Trajectory (CLEAN - For Validation) ---
    print("\n> Step 1c: Generating New Trajectory for Validation (CLEAN/NOISELESS)...")
    new_params = {k: v for k, v in params.items() if k != 'new_trajectory_params'}
    new_params.update(params['new_trajectory_params'])
    
    # !!! KEY UPDATE: Force noise to 0.0 for validation !!!
    new_params['noise_level'] = 0.0 
    
    X_new, X_dot_new, t_new = system_generator(**new_params)
    print(f"  New trajectory shape: X={X_new.shape}, X_dot={X_new.shape} (Noise: 0.0)")

    # --- AutoSINDy (Your Method) ---
    start_time_autosindy = time.time()

    if cfg["use_unified_library"]:
        print("\n--- Running with UNIFIED Library Approach ---")
        raw_library_strings = discover_basis_functions(
            X_train, X_dot_train, feature_names,
            n_chunks=cfg["discovery_chunks"],
            chunk_size_divisor=cfg["chunk_size_divisor"],
            pysr_params=cfg["pysr_params"]
        )
        final_functions, final_names = curate_library(
            raw_library_strings, X_train, feature_names,
            curation_config=cfg["curation_params"]
        )
        
        auto_sindy_model = fit_sindy_model(
            X_train, X_dot_train, t_train,
            final_functions, final_names, feature_names,
            optimizer_config=cfg["optimizer_params"] 
        )
    else:
        print("\n--- Running with SEPARATE Library Approach ---")
        separate_models = []
        for j in range(n_features):
            print(f"\n----- Processing State Variable x{j} -----")
            raw_library_strings_j = discover_basis_functions(
                X_train, X_dot_train[:, j:j+1], feature_names,
                n_chunks=cfg["discovery_chunks"],
                chunk_size_divisor=cfg["chunk_size_divisor"],
                pysr_params=cfg["pysr_params"]
            )
            final_functions_j, final_names_j = curate_library(
                raw_library_strings_j, X_train, feature_names,
                curation_config=cfg["curation_params"]
            )
            
            model_j = fit_sindy_model(
                X_train, X_dot_train[:, j:j+1], t_train,
                final_functions_j, final_names_j, feature_names,
                optimizer_config=cfg["optimizer_params"] 
            )
            separate_models.append(model_j)

        # Instantiate the wrapper for the separate models
        auto_sindy_model = SeparateSINDyModel(separate_models, feature_names)

    end_time_autosindy = time.time()
    autosindy_time = end_time_autosindy - start_time_autosindy

    # --- Standard PySR Baseline ---
    print("\n" + "="*80)
    print("||               COMPARISON: Standard PySR (No SIndy)                   ||")
    print("="*80)
    start_time_pysr = time.time()
    standard_pysr_models = []
    for j in range(n_features):
        print(f"\n----- Running Standard PySR on State Variable x{j} -----")
        pysr_model = PySRRegressor(**cfg["pysr_params"], verbosity=0)
        pysr_model.fit(pd.DataFrame(X_train, columns=feature_names), X_dot_train[:, j])
        standard_pysr_models.append(pysr_model)
        if hasattr(pysr_model, 'equations_') and len(pysr_model.equations_) > 0:
            for eq in pysr_model.equations_.itertuples():
                print(f"        Complexity: {eq.complexity:<2} | Loss: {eq.loss:<.4f} | Score: {eq.score:<.6f} | Eq: {str(eq.sympy_format)}")

    end_time_pysr = time.time()
    pysr_time = end_time_pysr - start_time_pysr

    standard_pysr_wrapper = PySRWrapper(standard_pysr_models, feature_names)
    print("\nStandard PySR Model:")
    standard_pysr_wrapper.print()


    # --- Standard SINDy Baseline ---
    print("\n" + "="*80)
    print("||               COMPARISON: Standard SINDy with Enriched Library        ||")
    print("="*80)
    start_time_sindy = time.time()
    poly_library = ps.PolynomialLibrary(degree=cfg["sindy_poly_degree"])
    fourier_library = ps.FourierLibrary(n_frequencies=cfg["sindy_fourier_freqs"])
    combined_library = poly_library + (poly_library * fourier_library)
    standard_optimizer = get_sindy_optimizer(cfg["optimizer_params"], n_samples=X_train.shape[0])
    standard_sindy_model = ps.SINDy(
        feature_library=combined_library,
        optimizer=standard_optimizer, 
        feature_names=feature_names
    )
    standard_sindy_model.fit(X_train, t=t_train, x_dot=X_dot_train, quiet=True)
    # apply_ensemble_filter(standard_sindy_model, cfg["optimizer_params"])
    
    end_time_sindy = time.time()
    sindy_time = end_time_sindy - start_time_sindy
    print("Standard Enriched Model:")
    standard_sindy_model.print(precision=10)

    # --- Final AutoSINDy Result Printout ---
    print("\n" + "="*80)
    print("||                   RESULTS: AutoSINDy (Your Method)                       ||")
    print("="*80)
    print("Discovered Model:")
    auto_sindy_model.print(precision=10)

    # --- Final Timing Comparison ---
    print("\n" + "="*80)
    print("||                       RESULTS: Discovery Times                         ||")
    print("="*80)
    print(f"AutoSINDy (Your Method): {autosindy_time:.2f} seconds")
    print(f"Standard PySR:           {pysr_time:.2f} seconds")
    print(f"Standard SINDy:          {sindy_time:.2f} seconds")
    print("="*80)

    # --- Create a dictionary of models to plot and simulate ---
    all_models = {
        'AutoSINDy': auto_sindy_model,
        'Standard SINDy': standard_sindy_model,
        'Standard PySR': standard_pysr_wrapper
    }

    # --- Visual Comparison ---
    # 1. Training/Test Split (Plotting NOISY data as requested)
    plot_derivative_comparison(t, X, X_dot, t_train, all_models, feature_names, cfg["split_strategy"], run_tag)
    
    # 2. New Trajectory (Plotting CLEAN data for validation)
    plot_new_derivative_comparison(t_new, X_new, X_dot_new, all_models, feature_names, run_tag)

    # --- Simulation Comparison (only for selected models) ---
    models_to_simulate_names = cfg["simulation_params"]["models_to_simulate"]
    models_to_simulate_dict = {name: model for name, model in all_models.items() if name in models_to_simulate_names}

    simulation_scores = {}
    simulation_times = {}
    if models_to_simulate_dict:
        print(f"\nModels selected for simulation: {list(models_to_simulate_dict.keys())}")
        t_sim_end = cfg["simulation_params"]["t_sim_end"]
        n_sim_samples = cfg["simulation_params"]["n_sim_samples"]
        t_sim = np.linspace(0, t_sim_end, n_sim_samples)

        # Start simulation from the clean initial condition of the new trajectory
        x0_sim = X_new[0]

        sim_params = {k: v for k, v in params.items() if k != 'new_trajectory_params'}
        sim_params['t_end'] = t_sim_end
        sim_params['n_samples'] = n_sim_samples
        sim_params['x0'] = x0_sim
        
        # !!! KEY UPDATE: Force noise to 0.0 for Simulation Truth !!!
        sim_params['noise_level'] = 0.0
        
        X_true_sim, _, _ = system_generator(**sim_params)

        simulation_scores, simulation_times = plot_simulation_comparison(t_sim, X_true_sim, models_to_simulate_dict, feature_names, run_tag)

    # --- Final Performance Reports and Data Collection ---
    performance_results = {}
    for name, model in all_models.items():
        title = f"PERFORMANCE: {name}"
        sim_scores_for_model = simulation_scores.get(name)
        
        # Evaluates:
        # Train/Test on NOISY data (from split)
        # New Traj on CLEAN data (from X_new)
        performance_results[name] = evaluate_and_print_performance(
            model, title,
            X_train, X_dot_train, t_train,
            X_test, X_dot_test, t_test,
            X_new, X_dot_new, t_new,
            sim_scores=sim_scores_for_model
        )

    # === NEW TIDY FORMAT: Assemble and Save the Final Log ===
    print("\n> Assembling final results for tidy logging...")

    # Create a list to hold the data for each row in the CSV
    results_list = []
    
    if cfg["curation_params"]["pruning_method"] == 'vif':
        active_threshold = cfg["curation_params"]["vif_threshold"]
    else:
        active_threshold = cfg["curation_params"]["correlation_threshold"]


    # 1. Define the base information common to this entire experimental run
    base_log = {
        "global_seed": cfg.get("global_seed", 32),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system_name": cfg["system_to_run"],
        "noise_level": cfg["data_params"][system_to_run]["noise_level"],
        "library_strategy": "Unified" if cfg["use_unified_library"] else "Separate",
        "optimizer": cfg["optimizer_params"]["name"],
        "use_ensemble": cfg["optimizer_params"]["use_ensemble"],
        "threshold": cfg["optimizer_params"]["threshold"],
        
        # Correctly pointing to curation_params now
        "pruning_method": cfg["curation_params"]["pruning_method"],
        "active_curation_threshold": active_threshold,
        "expansion_strategy": cfg["curation_params"]["expansion_strategy"],
        
        "split_strategy": cfg["split_strategy"],
    }

    # 2. Define a helper function to format equations into a single string
    def format_equations(eqs):
        """Joins a list of equation strings with a semicolon."""
        return "; ".join(eqs)

    # 3. Store equations and discovery times for each model in dictionaries for easy lookup
    model_equations = {
        'AutoSINDy': format_equations(auto_sindy_model.equations(precision=10)),
        'Standard SINDy': format_equations(standard_sindy_model.equations(precision=10)),
        'Standard PySR': format_equations([str(m.get_best().get('sympy_format', 'No eq found')) for m in standard_pysr_wrapper.models])
    }

    model_times = {
        'AutoSINDy': autosindy_time,
        'Standard SINDy': sindy_time,
        'Standard PySR': pysr_time
    }

    # 4. Loop through each model's performance results to create a specific row for it
    for model_name, perf_data in performance_results.items():
        # Start with a copy of the common information for this run
        model_log = base_log.copy()

        # Add model-specific information
        model_log["model_name"] = model_name
        model_log["discovery_time_s"] = model_times.get(model_name, np.nan)
        model_log["equations"] = model_equations.get(model_name, "N/A")

        # Add all performance metrics (r2, mse, complexity) from the evaluation step
        for metric, value in perf_data.items():
            model_log[metric] = value

        # Add the simulation time for this specific model
        model_log["sim_time_s"] = simulation_times.get(model_name, np.nan)

        # Add the completed dictionary (which represents one row) to our list
        results_list.append(model_log)

    # 5. Save the list of dictionaries to the CSV file
    df_to_save = pd.DataFrame(results_list)

    # Define a consistent column order for a clean and readable CSV file
    # REMOVED: 'sparsity_l0', 'coef_error'
    column_order = [
        'global_seed', 'timestamp', 'system_name', 'model_name', 'noise_level', 
        'library_strategy', 'optimizer', 'use_ensemble', 'threshold', 
        'active_curation_threshold', 'expansion_strategy', 'split_strategy', 'discovery_time_s', 
        'structural_complexity', 'canonical_complexity',
        'train_r2', 'train_mse', 'test_r2', 'test_mse',
        'new_traj_r2', 'new_traj_mse', 'sim_time_s', 'sim_r2', 'sim_mse', 'equations'
    ]
    # Reorder the dataframe to match our desired column order
    df_to_save = df_to_save.reindex(columns=column_order)
    
    # Append to the CSV file, creating it with a header if it's the first run
    try:
        file_exists = os.path.exists(cfg["results_csv_file"])
        if file_exists:
            df_to_save.to_csv(cfg["results_csv_file"], mode='a', header=False, index=False)
            print(f"\n✓ Tidy results appended to {cfg['results_csv_file']}")
        else:
            df_to_save.to_csv(cfg["results_csv_file"], mode='w', header=True, index=False)
            print(f"\n✓ Tidy results saved to new file: {cfg['results_csv_file']}")
    except IOError as e:
        print(f"\n✗ Error saving tidy results to {cfg['results_csv_file']}: {e}")
        
        
        
        
        
# ==========================================================================
# 3. CLEAN MAIN BLOCK
# ==========================================================================
if __name__ == "__main__":
    # Now this block is very simple
    run_experiment(config)