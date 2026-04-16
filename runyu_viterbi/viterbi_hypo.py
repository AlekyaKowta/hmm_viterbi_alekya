from __future__ import annotations
import csv
import itertools
import math
import random
from typing import Dict, List, Sequence, Tuple

# ============================================================
# Basic helpers
# ============================================================

def make_labels(prefix: str, n: int) -> List[str]:
    return [f"{prefix}{i+1}" for i in range(n)]

def normalize_row(row: Sequence[float]) -> List[float]:
    total = sum(row)
    return [max(1e-12, x) / total for x in row]

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
    n1  = math.sqrt(sum(a * a for a in v1))
    n2  = math.sqrt(sum(b * b for b in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)

def compute_stationary_distribution(transition_matrix: List[List[float]], n_iter: int = 10000) -> List[float]:
    n = len(transition_matrix)
    dist = [1.0 / n] * n
    for _ in range(n_iter):
        new_dist = [0.0] * n
        for j in range(n):
            for i in range(n):
                new_dist[j] += dist[i] * transition_matrix[i][j]
        dist = new_dist
    return dist

def blended_matrix(n: int, sharpness: float, rng: random.Random) -> List[List[float]]:
    base = 1.0 / n
    matrix = []
    for i in range(n):
        row = [base * (1.0 - sharpness)] * n
        row[i] += sharpness
        noise = [0.02 * rng.random() for _ in range(n)]
        row = [max(1e-9, a + b) for a, b in zip(row, noise)]
        matrix.append(normalize_row(row))
    return matrix

class HMM:
    def __init__(self, states, observations, start_prob, transition_prob, emission_prob):
        self.states = states
        self.observations = observations
        self.start_prob = start_prob
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob

def build_hmm(n, emission_matrix, transition_matrix):
    states = make_labels("S", n)
    observations = make_labels("E", n)
    start_prob = {s: 1.0 / n for s in states}
    transition_prob = {states[i]: {states[j]: transition_matrix[i][j] for j in range(n)} for i in range(n)}
    emission_prob = {states[i]: {observations[j]: emission_matrix[i][j] for j in range(n)} for i in range(n)}
    return HMM(states, observations, start_prob, transition_prob, emission_prob)

def generate_sequence(hmm: HMM, length: int) -> Tuple[List[str], List[str]]:
    state = sample_from_distribution(hmm.start_prob)
    states = [state]
    obs = [sample_from_distribution(hmm.emission_prob[state])]
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

def run_trials(hmm: HMM, num_trials=1000, seq_length=15) -> float:
    correct, total = 0, 0
    for _ in range(num_trials):
        true_states, obs = generate_sequence(hmm, seq_length)
        pred_states = viterbi(hmm, obs)
        correct += sum(a == b for a, b in zip(true_states, pred_states))
        total += seq_length
    return correct / total

# ============================================================
# The Four Formulas
# ============================================================

def formula_original(emission_matrix, n):
    pairs = list(itertools.combinations(range(n), 2))
    mean_y = sum(cosine_similarity(emission_matrix[i], emission_matrix[j]) for i, j in pairs) / len(pairs)
    return (1 - mean_y) * ((n-1)/n) + (1/n)

def formula_hypo1(emission_matrix, transition_matrix, n):
    pairs = list(itertools.combinations(range(n), 2))
    k = len(pairs)
    total = sum(cosine_similarity(emission_matrix[i], emission_matrix[j]) *
                cosine_similarity(transition_matrix[i], transition_matrix[j]) for i, j in pairs)
    return (1 - total / k) * ((n-1)/n) + (1/n)

def formula_hypo1_1(emission_matrix, transition_matrix, n):
    pairs = list(itertools.combinations(range(n), 2))
    k = len(pairs)
    sd = compute_stationary_distribution(transition_matrix)
    total = sum(cosine_similarity(emission_matrix[i], emission_matrix[j]) *
                cosine_similarity(transition_matrix[i], transition_matrix[j]) *
                sd[i] * sd[j] for i, j in pairs)
    return (1 - total / k) * ((n-1)/n) + (1/n)

def formula_hypo2(emission_matrix, transition_matrix, n, w_e=0.8, w_t=0.2):
    pairs = list(itertools.combinations(range(n), 2))
    k = len(pairs)
    weighted_confusion_total = sum((w_e * cosine_similarity(emission_matrix[i], emission_matrix[j]) +
                                    w_t * cosine_similarity(transition_matrix[i], transition_matrix[j]))
                                   for i, j in pairs)
    mean_confusion = weighted_confusion_total / k
    return (1 - mean_confusion) * ((n-1)/n) + (1/n)

# ============================================================
# Main Experiment
# ============================================================

def run_experiment(n=3, num_trials=1000, seq_length=15, seed=42):
    rng = random.Random(seed)
    random.seed(seed)
    sharpness_levels = [0.0, 0.3, 0.6, 0.9]
    results = []

    print(f"\n{'='*150}")
    print(f"  {n}x{n} HMM — Comparing Original, H1, H1.1, and H2")
    print(f"{'='*150}\n")

    header = (f"{'EmSh':>5} {'TrSh':>5} | {'AA':>7} | "
              f"{'Orig':>7} {'H1':>7} {'H1.1':>7} {'H2':>7} | "
              f"{'ErrOrig':>8} {'ErrH1':>8} {'ErrH1.1':>8} {'ErrH2':>8} | {'Best':>8}")
    print(header)
    print("-" * len(header))

    for em_sharp in sharpness_levels:
        for tr_sharp in sharpness_levels:
            em_mat = blended_matrix(n, em_sharp, rng)
            tr_mat = blended_matrix(n, tr_sharp, rng)
            hmm = build_hmm(n, em_mat, tr_mat)
            aa = run_trials(hmm, num_trials, seq_length)

            p_orig = formula_original(em_mat, n)
            p_h1   = formula_hypo1(em_mat, tr_mat, n)
            p_h11  = formula_hypo1_1(em_mat, tr_mat, n)
            p_h2   = formula_hypo2(em_mat, tr_mat, n)

            err_o, err_1, err_11, err_2 = abs(p_orig-aa), abs(p_h1-aa), abs(p_h11-aa), abs(p_h2-aa)
            
            errors = {"Original": err_o, "H1": err_1, "H1.1": err_11, "H2": err_2}
            best_name = min(errors, key=errors.get)

            results.append({
                "em_sharp": em_sharp, "tr_sharp": tr_sharp, "actual": aa,
                "p_orig": p_orig, "p_h1": p_h1, "p_h11": p_h11, "p_h2": p_h2,
                "err_orig": err_o, "err_h1": err_1, "err_h11": err_11, "err_h2": err_2,
                "best": best_name
            })

            print(f"{em_sharp:>5.1f} {tr_sharp:>5.1f} | {aa:>7.2%} | "
                  f"{p_orig:>7.2%} {p_h1:>7.2%} {p_h11:>7.2%} {p_h2:>7.2%} | "
                  f"{err_o:>8.2%} {err_1:>8.2%} {err_11:>8.2%} {err_2:>8.2%} | {best_name:>8}")

    print(f"\nSUMMARY")
    for key in ["err_orig", "err_h1", "err_h11", "err_h2"]:
        avg_err = sum(r[key] for r in results) / len(results)
        print(f"  Average error {key.replace('err_', '').upper():<5}: {avg_err:.4%}")

    return results

def save_csv(results, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {path}")

if __name__ == "__main__":
    results = run_experiment(n=3, num_trials=1000, seq_length=15, seed=42)
    save_csv(results, "viterbi_hypo_results_final.csv")