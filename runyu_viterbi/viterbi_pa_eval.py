"""
viterbi_pa_eval.py

Author: Runyu Ma

Compute the predicted accuracy P_a from Chapter 3.1 and compare
it against observed Viterbi decoding accuracy on synthetic n x n HMMs.

This script follows the paper's experimental scope:
- uniform initial probabilities: 1 / n
- uniform transition matrix: every transition probability is 1 / n
- square emission matrix: n states, n observations
- synthetic 3x3 and 4x4 scenarios

What this script does:
1. Build synthetic emission matrices for 3x3 and 4x4 HMMs.
2. Compute pairwise cosine similarity between emission rows.
3. Compute predicted accuracy P_a from Chapter 3.1:
       P_a = (1 - mean_pairwise_cosine) * (n - 1) / n + 1 / n
4. Simulate hidden-state / observation sequences.
5. Run Viterbi decoding and estimate actual accuracy over repeated trials.
6. Report the prediction-vs-observed error.

Note:
The paper text available here exposes the formula and the general 3x3 / 4x4
setup, but it does not provide the full emission matrix entries for every
published scenario in a machine-readable form. Therefore, this script creates
synthetic scenarios consistent with the paper's setup rather than claiming to
exactly reconstruct every table row from the paper.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


# ============================================================
# Data structures
# ============================================================


@dataclass(frozen=True)
class HMMConfig:
    """An HMM with square emission matrix under the paper's setup."""

    name: str
    states: List[str]
    observations: List[str]
    start_prob: Dict[str, float]
    transition_prob: Dict[str, Dict[str, float]]
    emission_prob: Dict[str, Dict[str, float]]


@dataclass(frozen=True)
class ScenarioResult:
    """Summary for one synthetic scenario."""

    name: str
    n_states: int
    pairwise_cosines: List[float]
    mean_cosine: float
    predicted_accuracy_pa: float
    actual_accuracy_aa: float
    absolute_error: float


# ============================================================
# Basic helpers
# ============================================================


def make_labels(prefix: str, n: int) -> List[str]:
    return [f"{prefix}{i + 1}" for i in range(n)]


def normalize_row(row: Sequence[float]) -> List[float]:
    total = sum(row)
    if total <= 0:
        raise ValueError("Each emission row must have strictly positive sum.")
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
    if last_key is None:
        raise ValueError("Cannot sample from an empty distribution.")
    return last_key


def safe_log(x: float) -> float:
    return float("-inf") if x <= 0.0 else math.log(x)


# ============================================================
# Cosine similarity and Chapter 3.1 P_a
# ============================================================


def cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same dimension.")
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    return dot / (norm1 * norm2)


def pairwise_cosine_similarities(emission_matrix: Sequence[Sequence[float]]) -> List[float]:
    cosines: List[float] = []
    for i, j in itertools.combinations(range(len(emission_matrix)), 2):
        cosines.append(cosine_similarity(emission_matrix[i], emission_matrix[j]))
    return cosines


def compute_pa_from_emission_matrix(emission_matrix: Sequence[Sequence[float]]) -> Tuple[List[float], float, float]:
    """
    Compute Chapter 3.1 predicted accuracy P_a.

    Formula from the paper:
        P_a = (1 - mean_pairwise_cosine) * (n - 1) / n + 1 / n

    Returns:
        pairwise_cosines, mean_cosine, predicted_accuracy_pa
    """
    n = len(emission_matrix)
    if n < 2:
        raise ValueError("Need at least 2 states to compute pairwise cosine similarity.")
    for row in emission_matrix:
        if len(row) != n:
            raise ValueError("This script assumes an n x n emission matrix.")

    cosines = pairwise_cosine_similarities(emission_matrix)
    mean_cos = sum(cosines) / len(cosines)
    pa = (1.0 - mean_cos) * ((n - 1.0) / n) + (1.0 / n)
    return cosines, mean_cos, pa


# ============================================================
# Build HMM from an emission matrix
# ============================================================


def build_uniform_hmm_from_emission_matrix(
    name: str,
    emission_matrix: Sequence[Sequence[float]],
) -> HMMConfig:
    n = len(emission_matrix)
    if n == 0:
        raise ValueError("Emission matrix cannot be empty.")

    states = make_labels("S", n)
    observations = make_labels("E", n)
    start_prob = uniform_distribution(states)
    transition_prob = {state: uniform_distribution(states) for state in states}

    emission_prob: Dict[str, Dict[str, float]] = {}
    for state, row in zip(states, emission_matrix):
        if len(row) != n:
            raise ValueError("Emission matrix must be square (n x n).")
        normalized = normalize_row(row)
        emission_prob[state] = {obs: prob for obs, prob in zip(observations, normalized)}

    return HMMConfig(
        name=name,
        states=states,
        observations=observations,
        start_prob=start_prob,
        transition_prob=transition_prob,
        emission_prob=emission_prob,
    )


def emission_prob_dict_to_matrix(hmm: HMMConfig) -> List[List[float]]:
    matrix: List[List[float]] = []
    for state in hmm.states:
        matrix.append([hmm.emission_prob[state][obs] for obs in hmm.observations])
    return matrix


# ============================================================
# Sequence generation and Viterbi decoding
# ============================================================


def generate_sequence(config: HMMConfig, seq_length: int) -> Tuple[List[str], List[str]]:
    if seq_length <= 0:
        return [], []

    hidden_states: List[str] = []
    observations: List[str] = []

    current_state = sample_from_distribution(config.start_prob)
    hidden_states.append(current_state)
    observations.append(sample_from_distribution(config.emission_prob[current_state]))

    for _ in range(1, seq_length):
        current_state = sample_from_distribution(config.transition_prob[current_state])
        hidden_states.append(current_state)
        observations.append(sample_from_distribution(config.emission_prob[current_state]))

    return hidden_states, observations


def viterbi_decode(hmm: HMMConfig, observation_sequence: Sequence[str]) -> List[str]:
    if not observation_sequence:
        return []

    states = hmm.states
    dp: List[Dict[str, float]] = []
    backpointer: List[Dict[str, str]] = []

    first_obs = observation_sequence[0]
    dp0: Dict[str, float] = {}
    bp0: Dict[str, str] = {}
    for state in states:
        dp0[state] = safe_log(hmm.start_prob[state]) + safe_log(hmm.emission_prob[state][first_obs])
        bp0[state] = ""
    dp.append(dp0)
    backpointer.append(bp0)

    for t in range(1, len(observation_sequence)):
        obs_t = observation_sequence[t]
        dp_t: Dict[str, float] = {}
        bp_t: Dict[str, str] = {}
        for curr_state in states:
            emission_log_prob = safe_log(hmm.emission_prob[curr_state][obs_t])
            best_score = float("-inf")
            best_prev_state = states[0]
            for prev_state in states:
                candidate = (
                    dp[t - 1][prev_state]
                    + safe_log(hmm.transition_prob[prev_state][curr_state])
                    + emission_log_prob
                )
                if candidate > best_score:
                    best_score = candidate
                    best_prev_state = prev_state
            dp_t[curr_state] = best_score
            bp_t[curr_state] = best_prev_state
        dp.append(dp_t)
        backpointer.append(bp_t)

    final_state = max(states, key=lambda s: dp[-1][s])
    best_path = [final_state]
    for t in range(len(observation_sequence) - 1, 0, -1):
        final_state = backpointer[t][final_state]
        best_path.append(final_state)
    best_path.reverse()
    return best_path


def sequence_accuracy(true_states: Sequence[str], pred_states: Sequence[str]) -> float:
    if len(true_states) != len(pred_states):
        raise ValueError("True and predicted sequences must have equal length.")
    if not true_states:
        return 0.0
    correct = sum(1 for a, b in zip(true_states, pred_states) if a == b)
    return correct / len(true_states)


def estimate_actual_accuracy(
    hmm: HMMConfig,
    num_trials: int,
    seq_length: int,
) -> float:
    accuracies: List[float] = []
    for _ in range(num_trials):
        true_states, observations = generate_sequence(hmm, seq_length)
        pred_states = viterbi_decode(hmm, observations)
        accuracies.append(sequence_accuracy(true_states, pred_states))
    return sum(accuracies) / len(accuracies)


# ============================================================
# Synthetic scenario generation
# ============================================================


def identity_emission_matrix(n: int) -> List[List[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def equal_emission_matrix(n: int) -> List[List[float]]:
    value = 1.0 / n
    return [[value for _ in range(n)] for _ in range(n)]


def random_emission_matrix(n: int, rng: random.Random) -> List[List[float]]:
    matrix: List[List[float]] = []
    for _ in range(n):
        raw = [rng.random() + 1e-12 for _ in range(n)]
        matrix.append(normalize_row(raw))
    return matrix


def blended_emission_matrix(n: int, sharpness: float, rng: random.Random) -> List[List[float]]:
    """
    Create a row-stochastic matrix that smoothly interpolates between:
    - equal emission rows when sharpness = 0
    - more state-specific rows as sharpness grows toward 1
    """
    if not 0.0 <= sharpness <= 1.0:
        raise ValueError("sharpness must be in [0, 1].")

    matrix: List[List[float]] = []
    base = 1.0 / n
    for i in range(n):
        row = [base * (1.0 - sharpness) for _ in range(n)]
        row[i] += sharpness

        # Small random perturbation to avoid all intermediate scenarios looking too symmetric.
        noise = [0.05 * rng.random() for _ in range(n)]
        row = [max(0.0, a + b) for a, b in zip(row, noise)]
        matrix.append(normalize_row(row))
    return matrix


def default_scenarios_for_size(n: int, rng: random.Random, random_count: int = 6) -> List[Tuple[str, List[List[float]]]]:
    scenarios: List[Tuple[str, List[List[float]]]] = []

    scenarios.append((f"Scenario {n}x{n} Unique", identity_emission_matrix(n)))

    # A few progressively less-separated structured cases.
    if n == 3:
        sharpness_values = [0.80, 0.60, 0.45]
    else:
        sharpness_values = [0.85, 0.65, 0.50]

    for idx, sharpness in enumerate(sharpness_values, start=1):
        scenarios.append(
            (f"Scenario {n}x{n} Structured-{idx}", blended_emission_matrix(n, sharpness, rng))
        )

    for idx in range(1, random_count + 1):
        scenarios.append((f"Scenario {n}x{n} Random-{idx}", random_emission_matrix(n, rng)))

    scenarios.append((f"Scenario {n}x{n} Equal", equal_emission_matrix(n)))
    return scenarios


# ============================================================
# Evaluation pipeline
# ============================================================


def evaluate_scenario(
    name: str,
    emission_matrix: Sequence[Sequence[float]],
    num_trials: int,
    seq_length: int,
) -> ScenarioResult:
    hmm = build_uniform_hmm_from_emission_matrix(name, emission_matrix)
    normalized_matrix = emission_prob_dict_to_matrix(hmm)
    pairwise_cosines, mean_cos, pa = compute_pa_from_emission_matrix(normalized_matrix)
    aa = estimate_actual_accuracy(hmm, num_trials=num_trials, seq_length=seq_length)
    return ScenarioResult(
        name=name,
        n_states=len(normalized_matrix),
        pairwise_cosines=pairwise_cosines,
        mean_cosine=mean_cos,
        predicted_accuracy_pa=pa,
        actual_accuracy_aa=aa,
        absolute_error=abs(pa - aa),
    )


def format_percent(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def print_matrix(matrix: Sequence[Sequence[float]]) -> None:
    for row in matrix:
        print("  [" + ", ".join(f"{v:.4f}" for v in row) + "]")


def print_result(result: ScenarioResult) -> None:
    print("=" * 90)
    print(result.name)
    print(f"n_states          : {result.n_states}")
    print(
        "pairwise cosines  : "
        + ", ".join(f"{c:.4f}" for c in result.pairwise_cosines)
    )
    print(f"mean cosine       : {result.mean_cosine:.4f}")
    print(f"predicted P_a     : {format_percent(result.predicted_accuracy_pa)}")
    print(f"actual accuracy   : {format_percent(result.actual_accuracy_aa)}")
    print(f"absolute error    : {format_percent(result.absolute_error)}")


def print_summary(title: str, results: Sequence[ScenarioResult]) -> None:
    print("\n" + "#" * 90)
    print(title)
    print("#" * 90)
    header = (
        f"{'Scenario':30} {'MeanCos':>10} {'P_a':>10} {'AA':>10} {'AbsErr':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name[:30]:30} "
            f"{r.mean_cosine:10.4f} "
            f"{100.0 * r.predicted_accuracy_pa:9.2f}% "
            f"{100.0 * r.actual_accuracy_aa:9.2f}% "
            f"{100.0 * r.absolute_error:9.2f}%"
        )


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Chapter 3.1 P_a and compare it with actual Viterbi accuracy."
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[3, 4],
        help="Synthetic HMM sizes to evaluate. Default: 3 4",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1000,
        help="Number of Monte Carlo trials per scenario. Default: 1000",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=15,
        help="Sequence length per trial. Default: 15",
    )
    parser.add_argument(
        "--random-scenarios",
        type=int,
        default=6,
        help="How many additional random scenarios to generate per size. Default: 6",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    parser.add_argument(
        "--show-matrices",
        action="store_true",
        help="Print the emission matrix for each scenario.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    random.seed(args.seed)

    all_results: List[ScenarioResult] = []

    for n in args.sizes:
        if n < 2:
            raise ValueError("Each HMM size must be at least 2.")

        scenario_defs = default_scenarios_for_size(
            n=n,
            rng=rng,
            random_count=args.random_scenarios,
        )

        results_for_n: List[ScenarioResult] = []
        print("\n" + "*" * 90)
        print(f"Evaluating synthetic {n}x{n} HMM scenarios")
        print("*" * 90)

        for scenario_name, emission_matrix in scenario_defs:
            if args.show_matrices:
                print(f"\nEmission matrix for {scenario_name}:")
                print_matrix(emission_matrix)

            result = evaluate_scenario(
                name=scenario_name,
                emission_matrix=emission_matrix,
                num_trials=args.num_trials,
                seq_length=args.seq_length,
            )
            print_result(result)
            results_for_n.append(result)
            all_results.append(result)

        print_summary(f"Summary for synthetic {n}x{n} scenarios", results_for_n)

    avg_abs_error = sum(r.absolute_error for r in all_results) / len(all_results)
    print("\n" + "#" * 90)
    print("Overall summary")
    print("#" * 90)
    print(f"Total scenarios    : {len(all_results)}")
    print(f"Average abs. error : {format_percent(avg_abs_error)}")


if __name__ == "__main__":
    main()
