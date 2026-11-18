# HMM/Viterbi decoder utility

## Hidden state

At each GPS observation, the hidden state is the trip-local road-segment (arc) ID.
The arc ID is the row index in the trip's .arcs file. Its endpoints are node IDs
looked up in that file.

## Emissions

The decoder accepts the classifier's probability for each candidate arc at each
observation. These are normalised per observation and clipped away from zero
before taking logarithms. The classifier/candidate-generation stage is not yet
integrated in this repository.

## Transitions

For a previous arc ending at node v and a current candidate starting at node v,
the transition is treated as topologically connected. The mass 1 - epsilon is
shared uniformly among connected current candidates. A small epsilon mass is
shared among disconnected candidates so that one bad candidate set does not make
the whole sequence impossible.

If no directly connected current candidate exists, the decoder uses a uniform
fallback for that step. For a timestamp gap at or above the configured threshold
(default 30 seconds), the transition constraint is relaxed to a uniform transition.
This is a simple initial policy and must be tuned/validated using the permitted
training and student-testing partitions.

If candidate generation returns no candidates for an observation, the decoder
inserts a missing-state sentinel (-1). The output contains -1 at that observation;
it does not invent a road-segment ID.

## Viterbi recurrence

In log space, for state s at observation t:

V_t(s) = log P(x_t | s) + max_q [V_(t-1)(q) + log P(s | q)]

Backpointers retain the maximizing previous state and are used to reconstruct the
highest-scoring state sequence.

## Code and tests

- Implementation: src/pathsnap/hmm.py
- Unit tests: tests/test_hmm.py
- Run tests from the repository root with: python -m unittest discover -s tests -v

## Current limitation

The decoder utility and its edge-case unit tests are implemented. It has not yet
been connected to a trained candidate classifier or evaluated on the trip dataset.
Do not claim that end-to-end HMM route reconstruction has been completed until
that integration and comparison against classifier-only predictions are run.
