import unittest

from fastapi.testclient import TestClient

from api.app import app


class TestPathSnapApi(unittest.TestCase):
    def test_health_endpoint(self):
        client = TestClient(app)
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertIn("available_models", response.json())

    def test_predict_rejects_invalid_coordinate(self):
        client = TestClient(app)
        response = client.post("/predict", json={
            "trip_id": "invalid",
            "track": [{"longitude": 250, "latitude": 0, "timestamp": 1}],
            "nodes": [{"longitude": 0, "latitude": 0}, {"longitude": 1, "latitude": 1}],
            "arcs": [{"from_node_id": 0, "to_node_id": 1}],
        })
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
