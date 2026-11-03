
import unittest
import os

from app import app


class FakeNewsAPITest(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_home(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_valid_prediction(self):
        if not os.path.exists("model.pkl"):
            self.fail("model.pkl missing. Run train.py first.")

        data = {
            "source_credibility": 0.8,
            "author_credibility": 0.7,
            "clickbait_score": 0.2,
            "sentiment_score": 0.1,
            "article_length": 500,
            "num_quotes": 3,
            "num_images": 2,
            "num_links": 4,
            "social_shares": 100,
            "verified_source": 1
        }

        response = self.client.post("/predict", json=data)

        self.assertEqual(response.status_code, 200)

        result = response.get_json()
        self.assertIn(result["prediction"], [0, 1])
        self.assertIn(result["news_type"], ["Fake", "Real"])

    def test_missing_features(self):
        response = self.client.post(
            "/predict",
            json={"source_credibility": 0.8}
        )

        self.assertEqual(response.status_code, 400)

    def test_invalid_json(self):
        response = self.client.post(
            "/predict",
            data="invalid",
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
