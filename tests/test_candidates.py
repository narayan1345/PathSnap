import unittest

import pandas as pd

from pathsnap.candidates import FEATURE_COLUMNS, generate_candidates


class TestCandidateGeneration(unittest.TestCase):
    def test_candidate_rows_and_reference_target(self):
        nodes = pd.DataFrame({
            "longitude": [8.0000, 8.0001, 8.0002, 8.0003],
            "latitude": [50.0, 50.0, 50.0, 50.0],
        })
        arcs = pd.DataFrame({
            "from_node_id": [0, 1, 2],
            "to_node_id": [1, 2, 3],
        })
        track = pd.DataFrame({
            "longitude": [8.00005, 8.00015, 8.00025],
            "latitude": [50.0, 50.0, 50.0],
            "timestamp": [1.0, 2.0, 3.0],
        })
        route = pd.DataFrame({"arc_id": [0, 1, 2]})

        candidates, observations = generate_candidates(
            track, nodes, arcs, route,
            trip_id="synthetic", candidate_k=2, observation_stride=1,
        )

        self.assertEqual(len(candidates), 6)
        self.assertEqual(len(observations), 3)
        self.assertTrue(set(FEATURE_COLUMNS).issubset(candidates.columns))
        self.assertEqual(int(candidates.groupby("observation_key")["target"].sum().sum()), 3)
        self.assertFalse(observations["candidate_miss"].any())
        self.assertTrue((candidates["distance_to_segment_m"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
