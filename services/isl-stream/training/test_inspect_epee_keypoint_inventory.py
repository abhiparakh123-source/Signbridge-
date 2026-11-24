from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from training.inspect_epee_keypoint_inventory import (
    EXPECTED_KEYPOINT_FILES, inspect,
)


class FakeApi:
    def __init__(self, count=EXPECTED_KEYPOINT_FILES):
        self.count = count
        self.tree_args = None

    def list_repo_tree(self, **kwargs):
        self.tree_args = kwargs
        return [
            SimpleNamespace(path=f"keypoints/clip_{i:04}.npy", size=1024 + i)
            for i in range(self.count)
        ] + [SimpleNamespace(path="README.md", size=20)]

    def dataset_info(self, repo_id, revision):
        return SimpleNamespace(sha="test-commit-sha")


class KeypointInventoryTests(unittest.TestCase):
    def test_reports_exact_corpus_inventory_without_payload_download(self):
        api = FakeApi()
        report = inspect(api=api, revision="main")
        self.assertEqual(report["keypoint_npy_file_count"], 2400)
        self.assertTrue(report["count_matches_expected"])
        self.assertEqual(report["commit_hash"], "test-commit-sha")
        self.assertEqual(report["downloaded_payload_bytes"], 0)
        self.assertEqual(api.tree_args["revision"], "main")
        self.assertEqual(api.tree_args["path_in_repo"], "keypoints")

    def test_reports_mismatched_count_for_cli_to_reject(self):
        report = inspect(api=FakeApi(count=2399))
        self.assertEqual(report["keypoint_npy_file_count"], 2399)
        self.assertFalse(report["count_matches_expected"])

    def test_empty_inventory_has_safe_empty_statistics(self):
        report = inspect(api=FakeApi(count=0))
        self.assertEqual(report["total_bytes"], 0)
        self.assertIsNone(report["min_file_bytes"])
        self.assertIsNone(report["max_file_bytes"])
        self.assertIsNone(report["average_file_bytes"])


if __name__ == "__main__":
    unittest.main()
