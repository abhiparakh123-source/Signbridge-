from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.audit_epee_metadata import audit, npy_shape_from_header


class AuditEpeeMetadataTests(unittest.TestCase):
    def metadata_bytes(self):
        fields = ["clip_id", "signer_id", "phrase_id", "parallel", "text_en",
                  "fps", "n_frames", "oval_detection", "hand_slots", "gloss_sequence"]
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for clip, signer in [("c1", "A"), ("c2", "B")]:
            writer.writerow({
                "clip_id": clip, "signer_id": signer, "phrase_id": "7",
                "parallel": "yes", "text_en": "Hello", "fps": "30",
                "n_frames": "8", "oval_detection": "0.95",
                "hand_slots": "verified", "gloss_sequence": "HELLO|QUESTION",
            })
        return output.getvalue().encode()

    def test_metadata_counts_and_parallel_phrase_coverage(self):
        report = audit(None, self.metadata_bytes())
        self.assertEqual(report["metadata_rows"], 2)
        self.assertEqual(report["unique_signers"], 2)
        self.assertEqual(report["parallel_phrase_coverage"], {2: 1})
        self.assertEqual(report["errors"], [])
        self.assertFalse(report["video_downloaded"])
        self.assertFalse(report["array_payload_downloaded"])

    def test_local_annotation_identity_timing_and_gloss_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "annotations").mkdir()
            for clip, signer in [("c1", "A"), ("c2", "B")]:
                (root / "annotations" / f"{clip}.json").write_text(json.dumps({
                    "clip_id": clip, "signer_id": signer, "phrase_id": 7,
                    "text_en": "Hello", "fps": 30.0, "n_frames": 8,
                    "parallel": True, "hand_slots": "verified",
                    "segments": [
                        {"gloss": "HELLO", "start": 0.0, "end": 0.1},
                        {"gloss": "QUESTION", "start": 0.2, "end": 0.25},
                    ],
                }))
            report = audit(root, self.metadata_bytes())
            self.assertEqual(report["errors"], [])

    def test_flags_duplicate_ids_and_invalid_metadata(self):
        header = "clip_id,signer_id,phrase_id,parallel,text_en,fps,n_frames,oval_detection,hand_slots,gloss_sequence\n"
        rows = [
            "c1,A,7,yes,Hello,30,8,0.95,verified,HELLO|QUESTION",
            "c1,B,7,yes,Hello,30,8,0.95,verified,HELLO|QUESTION",
            "c3,C,8,no,Thanks,0,1,0.5,unverified,THANKS",
            "c4,D,9,no,Please,30,8,1.5,verified,PLEASE",
        ]
        report = audit(None, (header + "\n".join(rows) + "\n").encode())
        self.assertTrue(any("duplicate" in str(item["error"]) for item in report["errors"]))
        self.assertTrue(any("fps" in str(item["error"]) for item in report["errors"]))
        self.assertTrue(any("oval_detection" in str(item["error"]) for item in report["errors"]))

    def test_reads_npy_header_shape_without_array_payload(self):
        buffer = io.BytesIO()
        np.save(buffer, np.zeros((8, 128, 3), dtype=np.float32))
        self.assertEqual(npy_shape_from_header(buffer.getvalue()[:1024]), (8, 128, 3))


if __name__ == "__main__":
    unittest.main()
