"""HMM/Viterbi decoding utility for road-segment candidate sequences.

Inputs are one candidate-arc list and corresponding emission probabilities per GPS
observation. Arc endpoints use the trip-local 0-based arc IDs from the .arcs file.
A missing candidate set is represented by the sentinel -1 in the decoded output.
"""
from __future__ import annotations

from typing import Mapping, Sequence
import numpy as np


def viterbi_decode(
    candidate_arc_ids: Sequence[Sequence[int]],
    emission_probabilities: Sequence[Sequence[float]],
    arc_endpoints: Mapping[int, tuple[int, int]],
    *,
    disconnected_transition_probability: float = 0.01,
    long_gap_seconds: float = 30.0,
    time_gaps_seconds: Sequence[float] | None = None,
) -> list[int]:
    """Decode a likely connected arc sequence using log-space Viterbi.

    For directly connected consecutive road arcs, transition mass is split
    uniformly among connected candidates. A small configurable probability
    mass is reserved for disconnected candidates, allowing graceful recovery
    when candidate generation is imperfect. For a long GPS gap or an empty
    candidate set, the transition is reset to a uniform distribution.

    Parameters
    ----------
    candidate_arc_ids:
        Candidate arc IDs for each observation. Empty candidate lists are allowed.
    emission_probabilities:
        Positive classifier probabilities aligned with candidate_arc_ids.
    arc_endpoints:
        Mapping arc_id -> (from_node_id, to_node_id).
    disconnected_transition_probability:
        Probability mass reserved for non-adjacent candidate transitions (0, 1).
    long_gap_seconds:
        Timestamp gap at or above which the transition constraint is relaxed.
    time_gaps_seconds:
        Optional gaps between consecutive observations; length must be T - 1.

    Returns
    -------
    list[int]
        Most likely arc ID per observation. -1 marks an observation with no
        generated road candidates.
    """
    if len(candidate_arc_ids) != len(emission_probabilities):
        raise ValueError("Candidate lists and emission lists must have equal length.")
    if not 0.0 < disconnected_transition_probability < 1.0:
        raise ValueError("disconnected_transition_probability must be in (0, 1).")
    if long_gap_seconds < 0:
        raise ValueError("long_gap_seconds must be non-negative.")

    steps = len(candidate_arc_ids)
    if steps == 0:
        return []
    if time_gaps_seconds is not None and len(time_gaps_seconds) != steps - 1:
        raise ValueError("time_gaps_seconds must have length len(candidate_arc_ids) - 1.")

    states: list[list[int]] = []
    emissions: list[np.ndarray] = []
    for t, (ids, probs) in enumerate(zip(candidate_arc_ids, emission_probabilities)):
        ids = [int(x) for x in ids]
        probs_arr = np.asarray(probs, dtype=float)
        if len(ids) == 0:
            states.append([-1])
            emissions.append(np.array([1.0], dtype=float))
            continue
        if probs_arr.ndim != 1 or len(probs_arr) != len(ids):
            raise ValueError(f"Emission probabilities at step {t} do not align with candidate IDs.")
        if not np.isfinite(probs_arr).all() or (probs_arr < 0).any():
            raise ValueError(f"Emission probabilities at step {t} must be finite and non-negative.")
        probs_arr = np.clip(probs_arr, 1e-12, None)
        probs_arr /= probs_arr.sum()
        states.append(ids)
        emissions.append(probs_arr)

    scores: list[np.ndarray] = [np.log(emissions[0])]
    backpointers: list[np.ndarray] = [np.full(len(states[0]), -1, dtype=int)]

    for t in range(1, steps):
        previous_states, current_states = states[t - 1], states[t]
        previous_scores = scores[-1]
        current_scores = np.full(len(current_states), -np.inf, dtype=float)
        current_back = np.zeros(len(current_states), dtype=int)

        long_gap = (
            time_gaps_seconds is not None
            and float(time_gaps_seconds[t - 1]) >= long_gap_seconds
        )
        reset_transition = long_gap or -1 in previous_states or -1 in current_states

        for j, current_arc in enumerate(current_states):
            best_score = -np.inf
            best_i = 0
            for i, previous_arc in enumerate(previous_states):
                if reset_transition:
                    log_transition = -np.log(len(current_states))
                else:
                    prev_end = arc_endpoints.get(previous_arc, (None, None))[1]
                    curr_start = arc_endpoints.get(current_arc, (None, None))[0]
                    connected = (
                        prev_end is not None and curr_start is not None
                        and prev_end == curr_start
                    )
                    connected_indices = []
                    disconnected_indices = []
                    for k, next_arc in enumerate(current_states):
                        next_start = arc_endpoints.get(next_arc, (None, None))[0]
                        if prev_end is not None and next_start == prev_end:
                            connected_indices.append(k)
                        else:
                            disconnected_indices.append(k)

                    if not connected_indices:
                        log_transition = -np.log(len(current_states))
                    elif connected:
                        log_transition = (
                            np.log(1.0 - disconnected_transition_probability)
                            - np.log(len(connected_indices))
                        )
                    else:
                        log_transition = (
                            np.log(disconnected_transition_probability)
                            - np.log(max(1, len(disconnected_indices)))
                        )

                score = previous_scores[i] + log_transition
                if score > best_score:
                    best_score, best_i = score, i

            current_scores[j] = best_score + np.log(emissions[t][j])
            current_back[j] = best_i

        scores.append(current_scores)
        backpointers.append(current_back)

    last_index = int(np.argmax(scores[-1]))
    path_indices = [last_index]
    for t in range(steps - 1, 0, -1):
        path_indices.append(int(backpointers[t][path_indices[-1]]))
    path_indices.reverse()
    return [states[t][idx] for t, idx in enumerate(path_indices)]
