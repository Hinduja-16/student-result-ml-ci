
import os
import unittest
import json


class TestMLPipeline(unittest.TestCase):

    def test_model_file_exists(self):
        self.assertTrue(os.path.exists("student_result_model.pkl"))

    def test_metrics_file_exists(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_dataset_file_exists(self):
        self.assertTrue(os.path.exists("student_results.csv"))

    def test_metrics_content(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertIn("accuracy", metrics)
        self.assertIn("training_records", metrics)
        self.assertIn("testing_records", metrics)


if __name__ == "__main__":
    unittest.main()
