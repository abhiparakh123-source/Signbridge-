from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

# Keep this unit test independent of the optional Hugging Face client package.
hf_stub = types.ModuleType("huggingface_hub")
hf_stub.HfApi = object
hf_stub.snapshot_download = None
with patch.dict(sys.modules, {"huggingface_hub": hf_stub}):
    from training import download_epee_keypoints as downloader


class DownloadEpeeKeypointsTests(unittest.TestCase):
    def permission_file(self, root: Path) -> Path:
        record = {
            "status": "approved",
            "dataset_id": "CLERC-DATA/epee",
            "license_id": "CC-BY-NC-SA-4.0",
            "approved_by": "Family-approved reviewer",
            "reviewed_at": "2026-11-23",
            "written_permission_reference": "https://example.org/permission-record",
            "approved_uses": sorted(downloader.validate_permission_record.__globals__["REQUIRED_PERMISSION_USES"]),
        }
        path = root / "permission.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        return path

    def test_permission_gate_runs_before_any_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = root / "invalid.json"
            invalid.write_text("{}", encoding="utf-8")
            with patch.object(downloader, "snapshot_download") as snapshot:
                with self.assertRaisesRegex(ValueError, "status must be approved"):
                    downloader.download_keypoints(root / "data", invalid)
                snapshot.assert_not_called()

    def test_download_is_limited_to_metadata_annotations_and_keypoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            permission = self.permission_file(root)
            dataset_root = root / "dataset"
            def fake_snapshot_download(**kwargs):
                self.assertEqual(kwargs["repo_id"], "CLERC-DATA/epee")
                self.assertEqual(kwargs["repo_type"], "dataset")
                self.assertEqual(set(kwargs["allow_patterns"]), {
                    "metadata.csv", "annotations/*.json", "keypoints/*.npy"
                })
                dataset_root.mkdir(parents=True, exist_ok=True)
                (dataset_root / "metadata.csv").write_text("clip_id\n", encoding="utf-8")
                (dataset_root / "annotations").mkdir()
                (dataset_root / "keypoints").mkdir()
                for index in range(downloader.EXPECTED_KEYPOINT_FILES):
                    (dataset_root / "annotations" / f"{index}.json").write_text("{}", encoding="utf-8")
                    (dataset_root / "keypoints" / f"{index}.npy").write_bytes(b"x" * 8)
                return str(dataset_root)

            class FakeApi:
                def dataset_info(self, dataset_id, revision):
                    self.assert_dataset = dataset_id
                    return types.SimpleNamespace(sha="test-revision-sha")

            with patch.object(downloader, "snapshot_download", side_effect=fake_snapshot_download) as snapshot, \
                 patch.object(downloader, "HfApi", return_value=FakeApi()):
                report = downloader.download_keypoints(
                    dataset_root, permission, max_bytes=1024, revision="test-revision"
                )

            self.assertEqual(snapshot.call_count, 1)
            self.assertTrue(report["ok"])
            self.assertEqual(report["keypoint_files"], 2400)
            self.assertEqual(report["annotation_files"], 2400)
            self.assertEqual(report["keypoint_bytes"], 2400 * 8)
            self.assertFalse(report["videos_downloaded"])
            self.assertEqual(report["commit_hash"], "test-revision-sha")

    def test_rejects_unexpected_file_count_and_oversized_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            permission = self.permission_file(root)
            dataset_root = root / "dataset"
            def make_dataset(file_count: int, payload_size: int):
                dataset_root.mkdir(parents=True, exist_ok=True)
                (dataset_root / "metadata.csv").write_text("clip_id\n", encoding="utf-8")
                (dataset_root / "annotations").mkdir()
                (dataset_root / "keypoints").mkdir()
                for index in range(file_count):
                    (dataset_root / "annotations" / f"{index}.json").write_text("{}", encoding="utf-8")
                    (dataset_root / "keypoints" / f"{index}.npy").write_bytes(b"x" * payload_size)
                return str(dataset_root)

            with patch.object(downloader, "snapshot_download", side_effect=lambda **kw: make_dataset(1, 8)):
                with self.assertRaisesRegex(ValueError, "expected 2400 keypoint arrays"):
                    downloader.download_keypoints(dataset_root, permission)
            import shutil
            shutil.rmtree(dataset_root)

            with patch.object(downloader, "snapshot_download", side_effect=lambda **kw: make_dataset(2400, 2)):
                with self.assertRaisesRegex(ValueError, "exceeding limit"):
                    downloader.download_keypoints(dataset_root, permission, max_bytes=100)

    def test_rejects_destination_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            permission = self.permission_file(root)
            with patch.object(downloader, "snapshot_download", return_value=str(root / "outside")):
                (root / "outside").mkdir()
                with self.assertRaisesRegex(ValueError, "escaped"):
                    downloader.download_keypoints(root / "inside", permission)


if __name__ == "__main__":
    unittest.main()
