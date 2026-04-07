"""
viterbi_extended.py

Extension of viterbi_pa_eval.py to test two new formula hypotheses
when transition matrices are NOT uniform.

ORIGINAL formula (paper, Section 3.1):
    P_a = (1 - y) * (n-1)/n + 1/n
    where y = mean pairwise cosine similarity of EMISSION rows

NEW formulas being tested:
    Guess A (weighted average):
        P_a = (1 - w*y - (1-w)*x) * (n-1)/n + 1/n
        where x = mean cosine of TRANSITION rows

    Guess B (product):
        P_a = (1 - y*x) * (n-1)/n + 1/n

This script runs a grid across different HMM sizes (n).
"""

from __future__ import annotations

import csv
import itertools
import math
import random
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


# ============================================================
# Helpers
# ============================================================

def make_labels(prefix: str, n: int) -> List[str]:
    return [f"{prefix}{i+1}" for i in range(n)]

def normalize_row(row: Sequence[float]) -> List[float]:
    total = sum(row)
    return [x / total for x in row]

def uniform_distribution(labels: Sequence[str]) -> Dict[str, float]:
    p = 1.0 / len(labels)
    return {label: p for label in labels}

def sample_from_distribution(prob_dict: Dict[str, float]) -> str:
    r = random.random()
    cumulative = 0.0
    last_key = None
    for key, prob in prob_dict.items():
        cumulative += prob
        last_key = key
        if r <= cumulative:
            return key
    return last_key

def safe_log(x: float) -> float:
    return float("-inf") if x <= 0.0 else math.log(x)

def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a*a for a in v1))
    n2 = math.sqrt(sum(b*b for b in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)

def mean_pairwise_cosine(matrix: Sequence[Sequence[float]]) -> float:
    pairs = list(itertools.combinations(range(len(matrix)), 2))
    if not pairs:
        return 0.0
    cosines = [cosine_similarity(matrix[i], matrix[j]) for i, j in pairs]
    return sum(cosines) / len(cosines)


# ============================================================
# Matrix generators
# ============================================================

def blended_matrix(n: int, sharpness: float, rng: random.Random) -> List[List[float]]:
    base = 1.0 / n
    matrix = []
    for i in range(n):
        row = [base * (1.0 - sharpness) for _ in range(n)]
        row[i] += sharpness
        noise = [0.02 * rng.random() for _ in range(n)]
        row = [max(1e-9, a + b) for a, b in zip(row, noise)]
        matrix.append(normalize_row(row))
    return matrix


# ============================================================
# HMM builder & Viterbi
# ============================================================

@dataclass
class HMM:
    states: List[str]
    observations: List[str]
    start_prob: Dict[str, float]
    transition_prob: Dict[str, Dict[str, float]]
    emission_prob: Dict[str, Dict[str, float]]

def build_hmm(n: int, emission_matrix: List[List[float]], transition_matrix: List[List[float]]) -> HMM:
    states = make_labels("S", n)
    observations = make_labels("E", n)
    start_prob = uniform_distribution(states)

    transition_prob = {}
    for i, state in enumerate(states):
        row = normalize_row(transition_matrix[i])
        transition_prob[state] = {s: p for s, p in zip(states, row)}

    emission_prob = {}
    for i, state in enumerate(states):
        row = normalize_row(emission_matrix[i])
        emission_prob[state] = {o: p for o, p in zip(observations, row)}

    return HMM(states, observations, start_prob, transition_prob, emission_prob)

def generate_sequence(hmm: HMM, length: int) -> Tuple[List[str], List[str]]:
    state = sample_from_distribution(hmm.start_prob)
    states, obs = [state], [sample_from_distribution(hmm.emission_prob[state])]
    for _ in range(1, length):
        state = sample_from_distribution(hmm.transition_prob[state])
        states.append(state)
        obs.append(sample_from_distribution(hmm.emission_prob[state]))
    return states, obs

def viterbi(hmm: HMM, observations: Sequence[str]) -> List[str]:
    states = hmm.states
    dp = [{s: safe_log(hmm.start_prob[s]) + safe_log(hmm.emission_prob[s][observations[0]]) for s in states}]
    bp = [{s: "" for s in states}]

    for t in range(1, len(observations)):
        obs_t = observations[t]
        dp_t, bp_t = {}, {}
        for curr in states:
            em = safe_log(hmm.emission_prob[curr][obs_t])
            best_score, best_prev = float("-inf"), states[0]
            for prev in states:
                score = dp[t-1][prev] + safe_log(hmm.transition_prob[prev][curr]) + em
                if score > best_score:
                    best_score, best_prev = score, prev
            dp_t[curr] = best_score
            bp_t[curr] = best_prev
        dp.append(dp_t)
        bp.append(bp_t)

    final = max(states, key=lambda s: dp[-1][s])
    path = [final]
    for t in range(len(observations)-1, 0, -1):
        final = bp[t][final]
        path.append(final)
    path.reverse()
    return path

def run_trials(hmm: HMM, num_trials: int = 1000, seq_length: int = 15) -> float:
    correct_total = 0
    total = 0
    for _ in range(num_trials):
        true_states, obs = generate_sequence(hmm, seq_length)
        pred_states = viterbi(hmm, obs)
        correct_total += sum(a == b for a, b in zip(true_states, pred_states))
        total += seq_length
    return correct_total / total


# ============================================================
# The three formulas
# ============================================================

def formula_original(y: float, n: int) -> float:
    return (1 - y) * ((n-1)/n) + (1/n)

def formula_guess_a(y: float, x: float, n: int, w: float = 0.5) -> float:
    combined = w * y + (1 - w) * x
    return (1 - combined) * ((n-1)/n) + (1/n)

def formula_guess_b(y: float, x: float, n: int) -> float:
    return (1 - y * x) * ((n-1)/n) + (1/n)


# ============================================================
# Main Multi-Size Experiment
# ============================================================

def run_experiment(sizes: List[int] = [3, 4, 6], num_trials: int = 1000, seq_length: int = 50, seed: int = 42):
    rng = random.Random(seed)
    random.seed(seed)

    sharpness_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    all_results = []
    summaries = []

    print(f"\n{'#'*100}")
    print(f"STARTING MULTI-SIZE VITERBI EXPERIMENT")
    print(f"Testing HMM sizes: {sizes}")
    print(f"{'#'*100}")

    for n in sizes:
        print(f"\n{'='*100}")
        print(f"Running {n}x{n} HMM experiments: {len(sharpness_levels)**2} combinations")
        print("Running Viterbi trials...")
        print(f"{'='*100}\n")

        raw_data = []
        # 1. Collect Baseline Accuracy Data
        for em_sharp in sharpness_levels:
            for tr_sharp in sharpness_levels:
                emission_matrix = blended_matrix(n, em_sharp, rng)
                transition_matrix = blended_matrix(n, tr_sharp, rng)

                y = mean_pairwise_cosine(emission_matrix)
                x = mean_pairwise_cosine(transition_matrix)

                hmm = build_hmm(n, emission_matrix, transition_matrix)
                aa = run_trials(hmm, num_trials=num_trials, seq_length=seq_length)

                raw_data.append({
                    "em_sharp": em_sharp, "tr_sharp": tr_sharp,
                    "y": y, "x": x, "aa": aa
                })

        # 2. Grid Search for the optimal 'w' for THIS specific 'n'
        best_w = 0.5
        best_avg_err = float('inf')

        for w_int in range(101):
            w_test = w_int / 100.0
            total_err = sum(abs(formula_guess_a(d["y"], d["x"], n, w=w_test) - d["aa"]) for d in raw_data)
            avg_err = total_err / len(raw_data)
            if avg_err < best_avg_err:
                best_avg_err = avg_err
                best_w = w_test

        print(f"★ Grid Search Complete: Optimal 'w' for {n}x{n} found at {best_w:.2f} (Error: {best_avg_err:.2%}) ★\n")

        # 3. Generate detailed results table
        header = (f"{'Em.Sharp':>10} {'Tr.Sharp':>10} {'y(EmCos)':>10} {'x(TrCos)':>10} "
                  f"{'AA':>8} {'PA_orig':>8} {'PA_A(opt)':>9} {'PA_B':>8} "
                  f"{'Err_orig':>10} {'Err_A':>10} {'Err_B':>10}")
        print(header)
        print("-" * len(header))

        n_results = []
        for d in raw_data:
            pa_orig = formula_original(d["y"], n)
            pa_a    = formula_guess_a(d["y"], d["x"], n, w=best_w)
            pa_b    = formula_guess_b(d["y"], d["x"], n)

            err_orig = abs(pa_orig - d["aa"])
            err_a    = abs(pa_a - d["aa"])
            err_b    = abs(pa_b - d["aa"])

            row = {
                "n_states": n,
                "em_sharpness": d["em_sharp"],
                "tr_sharpness": d["tr_sharp"],
                "y_emission_cosine": round(d["y"], 4),
                "x_transition_cosine": round(d["x"], 4),
                "actual_accuracy": round(d["aa"], 4),
                "pa_original": round(pa_orig, 4),
                "pa_guess_a_opt": round(pa_a, 4), 
                "pa_guess_b": round(pa_b, 4),
                "error_original": round(err_orig, 4),
                "error_guess_a_opt": round(err_a, 4),
                "error_guess_b": round(err_b, 4),
                "optimal_w_used": best_w
            }
            n_results.append(row)
            all_results.append(row)

            print(
                f"{d['em_sharp']:>10.1f} {d['tr_sharp']:>10.1f} {d['y']:>10.4f} {d['x']:>10.4f} "
                f"{d['aa']:>8.2%} {pa_orig:>8.2%} {pa_a:>9.2%} {pa_b:>8.2%} "
                f"{err_orig:>10.2%} {err_a:>10.2%} {err_b:>10.2%}"
            )

        # 4. Save summary for the final printout
        avg_err_orig = sum(r["error_original"] for r in n_results) / len(n_results)
        avg_err_a    = sum(r["error_guess_a_opt"] for r in n_results) / len(n_results)
        avg_err_b    = sum(r["error_guess_b"] for r in n_results) / len(n_results)

        best_for_n = min(
            [("Original", avg_err_orig), (f"Guess A (w={best_w:.2f})", avg_err_a), ("Guess B", avg_err_b)],
            key=lambda t: t[1]
        )
        
        summaries.append({
            "n": n, "w": best_w,
            "err_orig": avg_err_orig, "err_a": avg_err_a, "err_b": avg_err_b,
            "best_name": best_for_n[0], "best_err": best_for_n[1]
        })

    # ============================================================
    # Grand Summary
    # ============================================================
    print(f"\n\n{'#'*100}")
    print("FINAL MULTI-SIZE SUMMARY")
    print(f"{'#'*100}")
    
    for s in summaries:
        print(f"\n--- {s['n']}x{s['n']} HMM ---")
        print(f"  Original Error: {s['err_orig']:.2%}")
        print(f"  Guess A Error : {s['err_a']:.2%}  (Optimal w = {s['w']:.2f})")
        print(f"  Guess B Error : {s['err_b']:.2%}")
        print(f"  ✓ Winner      : {s['best_name']} with {s['best_err']:.2%} error")

    return all_results

def save_csv(results: list, filename: str = "viterbi_results.csv"):
    if not results:
        return
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, filename)

    with open(full_path, "w", newline="") as f:
        # Move n_states to the front of the columns for easier reading in Excel
        keys = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nResults saved to {full_path}")

if __name__ == "__main__":
    # We pass [3, 4, 6] to sizes to run all three dimensions automatically
    results = run_experiment(sizes=[3, 4, 6], num_trials=1000, seq_length=50, seed=42)
    save_csv(results)
    print("\nDone! Open outputs/viterbi_results.csv to see all numbers.")