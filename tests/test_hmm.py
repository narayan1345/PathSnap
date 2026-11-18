import unittest

from pathsnap.hmm import viterbi_decode


class TestViterbiDecode(unittest.TestCase):
    def test_prefers_connected_path_over_stronger_local_emission(self):
        endpoints = {10: (0, 1), 11: (1, 2), 12: (5, 6)}
        result = viterbi_decode(
            [[10], [11, 12]],
            [[1.0], [0.1, 0.9]],
            endpoints,
            disconnected_transition_probability=0.01,
        )
        self.assertEqual(result, [10, 11])

    def test_empty_candidates_return_missing_sentinel(self):
        result = viterbi_decode([[10], [], [11]], [[1.0], [], [1.0]], {10: (0, 1), 11: (1, 2)})
        self.assertEqual(result[1], -1)

    def test_long_gap_relaxes_connectivity(self):
        endpoints = {10: (0, 1), 12: (5, 6)}
        result = viterbi_decode(
            [[10], [12]], [[1.0], [1.0]], endpoints,
            time_gaps_seconds=[120.0], long_gap_seconds=30.0,
        )
        self.assertEqual(result, [10, 12])


if __name__ == "__main__":
    unittest.main()
