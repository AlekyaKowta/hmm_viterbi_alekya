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
        where x = mean cosine of TRANSITION rows, w = 0.5

    Guess B (product):
        P_a = (1 - y*x) * (n-1)/n + 1/n

This script runs a 4x4 grid:
    emission sharpness in [0.0, 0.3, 0.6, 0.9]
    transition sharpness in [0.0, 0.3, 0.6, 0.9]

For each combination:
    - Builds an HMM with those emission and transition matrices
    - Runs 1000 Viterbi trials
    - Records actual accuracy (AA)
    - Computes all three formula predictions
    - Reports errors for each formula
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
# Helpers (same as original)
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
    """
    sharpness = 0.0 → all rows identical (uniform), cosine ≈ 1
    sharpness = 1.0 → identity-like rows,              cosine ≈ 0

    So HIGH sharpness = rows are VERY DIFFERENT = low cosine similarity
       LOW sharpness  = rows are VERY SIMILAR   = high cosine similarity
    """
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
# HMM builder with custom transition matrix
# ============================================================

@dataclass
class HMM:
    states: List[str]
    observations: List[str]
    start_prob: Dict[str, float]
    transition_prob: Dict[str, Dict[str, float]]
    emission_prob: Dict[str, Dict[str, float]]

def build_hmm(
    n: int,
    emission_matrix: List[List[float]],
    transition_matrix: List[List[float]],
) -> HMM:
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


# ============================================================
# Sequence generation + Viterbi
# ============================================================

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
    dp = [{s: safe_log(hmm.start_prob[s]) + safe_log(hmm.emission_prob[s][observations[0]])
           for s in states}]
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

    # Traceback
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
    """Original paper formula — only uses emission cosine y"""
    return (1 - y) * ((n-1)/n) + (1/n)

def formula_guess_a(y: float, x: float, n: int, w: float = 0.5) -> float:
    """
    Guess A: weighted average of emission (y) and transition (x) cosine
    combined = w*y + (1-w)*x
    When w=0.5, emission and transition are weighted equally.
    """
    combined = w * y + (1 - w) * x
    return (1 - combined) * ((n-1)/n) + (1/n)

def formula_guess_b(y: float, x: float, n: int) -> float:
    """
    Guess B: product of emission (y) and transition (x) cosine
    When both are 0 (totally distinct), product = 0 → accuracy = 1 (perfect)
    When either is 1 (identical rows), product pushes toward 1/n
    """
    return (1 - y * x) * ((n-1)/n) + (1/n)


# ============================================================
# Main experiment grid
# ============================================================

def run_experiment(n: int = 3, num_trials: int = 1000, seq_length: int = 15, seed: int = 42):
    rng = random.Random(seed)
    random.seed(seed)

    # Sharpness levels to test
    # sharpness=0.0 → rows very similar (cosine≈1)
    # sharpness=1.0 → rows very different (cosine≈0)
    sharpness_levels = [0.0, 0.3, 0.6, 0.9]

    results = []

    print(f"\n{'='*100}")
    print(f"Running {n}x{n} HMM experiments: {len(sharpness_levels)**2} combinations × {num_trials} trials each")
    print(f"{'='*100}\n")

    header = (f"{'Em.Sharp':>10} {'Tr.Sharp':>10} {'y(EmCos)':>10} {'x(TrCos)':>10} "
              f"{'AA':>8} {'PA_orig':>8} {'PA_A':>8} {'PA_B':>8} "
              f"{'Err_orig':>10} {'Err_A':>10} {'Err_B':>10}")
    print(header)
    print("-" * len(header))

    for em_sharp in sharpness_levels:
        for tr_sharp in sharpness_levels:
            # Build matrices
            emission_matrix = blended_matrix(n, em_sharp, rng)
            transition_matrix = blended_matrix(n, tr_sharp, rng)

            # Compute cosine similarities
            y = mean_pairwise_cosine(emission_matrix)   # emission cosine
            x = mean_pairwise_cosine(transition_matrix) # transition cosine

            # Build HMM and run trials
            hmm = build_hmm(n, emission_matrix, transition_matrix)
            aa = run_trials(hmm, num_trials=num_trials, seq_length=seq_length)

            # Compute all three formula predictions
            pa_orig = formula_original(y, n)
            pa_a    = formula_guess_a(y, x, n, w=0.5)
            pa_b    = formula_guess_b(y, x, n)

            # Errors
            err_orig = abs(pa_orig - aa)
            err_a    = abs(pa_a - aa)
            err_b    = abs(pa_b - aa)

            row = {
                "em_sharpness": em_sharp,
                "tr_sharpness": tr_sharp,
                "y_emission_cosine": round(y, 4),
                "x_transition_cosine": round(x, 4),
                "actual_accuracy": round(aa, 4),
                "pa_original": round(pa_orig, 4),
                "pa_guess_a": round(pa_a, 4),
                "pa_guess_b": round(pa_b, 4),
                "error_original": round(err_orig, 4),
                "error_guess_a": round(err_a, 4),
                "error_guess_b": round(err_b, 4),
            }
            results.append(row)

            print(
                f"{em_sharp:>10.1f} {tr_sharp:>10.1f} {y:>10.4f} {x:>10.4f} "
                f"{aa:>8.2%} {pa_orig:>8.2%} {pa_a:>8.2%} {pa_b:>8.2%} "
                f"{err_orig:>10.2%} {err_a:>10.2%} {err_b:>10.2%}"
            )

    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")

    avg_err_orig = sum(r["error_original"] for r in results) / len(results)
    avg_err_a    = sum(r["error_guess_a"]  for r in results) / len(results)
    avg_err_b    = sum(r["error_guess_b"]  for r in results) / len(results)

    print(f"Average error — Original formula : {avg_err_orig:.2%}")
    print(f"Average error — Guess A (weighted): {avg_err_a:.2%}")
    print(f"Average error — Guess B (product) : {avg_err_b:.2%}")

    best = min(
        [("Original", avg_err_orig), ("Guess A", avg_err_a), ("Guess B", avg_err_b)],
        key=lambda t: t[1]
    )
    print(f"\n✓ Best formula overall: {best[0]} with average error {best[1]:.2%}")

    return results

def save_csv(results: list, filename: str = "viterbi_results.csv"):
    if not results:
        return
        
    # 1. Find the exact folder where viterbi_extended.py lives
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Define an 'outputs' folder inside runyu_viterbi and create it if it doesn't exist
    output_dir = os.path.join(script_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Combine the folder path with your filename
    full_path = os.path.join(output_dir, filename)

    # 4. Save the file
    with open(full_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nResults saved to {full_path}")

if __name__ == "__main__":
    results = run_experiment(n=3, num_trials=1000, seq_length=15, seed=42)
    save_csv(results)
    print("\nDone! Open viterbi_results.csv to see all numbers.")