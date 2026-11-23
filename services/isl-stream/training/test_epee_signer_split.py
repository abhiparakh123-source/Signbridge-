"""Regression tests for the deterministic Épée signer split planner."""
import csv
import tempfile
import unittest
from pathlib import Path

from design_epee_signer_split import build_plan


SIGNERS = [
    ("ALPHA", "parallel"),
    ("BRAVO", "parallel"),
    ("CHARLIE", "parallel"),
    ("DELTA", "parallel"),
    ("ECHO", "parallel"),
    ("FOXTROT", "nonparallel"),
    ("GOLF", "nonparallel"),
    ("HOTEL", "nonparallel"),
]


def write_fixture(path: Path, *, duplicate_last=False, unknown_signer=False):
    fields = ["clip_id", "signer_id", "phrase_id", "parallel"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for signer, kind in SIGNERS:
            phrase_ids = (
                [str(i) for i in range(1, 301)]
                if signer != "ECHO"
                else [str(i) for i in range(2, 301) if i != 17] + ["301", "302"]
            ) if kind == "parallel" else [f"{signer}-phrase-{i}" for i in range(1, 301)]
            for index, phrase in enumerate(phrase_ids, start=1):
                actual_signer = "UNKNOWN" if unknown_signer and signer == "HOTEL" else signer
                writer.writerow({
                    "clip_id": f"{signer}-{index}",
                    "signer_id": actual_signer,
                    "phrase_id": phrase,
                    "parallel": "yes" if kind == "parallel" else "no",
                })
        if duplicate_last:
            writer.writerow({
                "clip_id": "ALPHA-1", "signer_id": "ALPHA",
                "phrase_id": "1", "parallel": "yes",
            })


class EpeeSignerSplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.metadata = Path(self.tmp.name) / "metadata.csv"
        write_fixture(self.metadata)

    def tearDown(self):
        self.tmp.cleanup()

    def test_expected_signer_disjoint_counts_and_phrase_groups(self):
        plan = build_plan(self.metadata)
        self.assertEqual(plan["status"], "proposed_split_plan_not_training_authorization")
        summary = plan["split_summary"]
        self.assertEqual([summary[s]["clips"] for s in ("train", "validation", "test")],
                         [1200, 600, 600])
        self.assertEqual(set(summary["train"]["signers"]), {"ALPHA", "BRAVO", "CHARLIE", "FOXTROT"})
        self.assertEqual(set(summary["validation"]["signers"]), {"DELTA", "GOLF"})
        self.assertEqual(set(summary["test"]["signers"]), {"ECHO", "HOTEL"})
        protocol = plan["evaluation_protocol"]
        self.assertEqual(protocol["primary_seen_phrase_test_count"], 298)
        self.assertEqual(protocol["test_only_zero_shot_phrase_count"], 2)
        self.assertEqual(protocol["test_only_zero_shot_phrase_ids"], ["301", "302"])

    def test_duplicate_clip_ids_are_rejected(self):
        write_fixture(self.metadata, duplicate_last=True)
        with self.assertRaisesRegex(ValueError, "duplicate clip_id"):
            build_plan(self.metadata)

    def test_unassigned_signer_is_rejected(self):
        write_fixture(self.metadata, unknown_signer=True)
        with self.assertRaisesRegex(ValueError, "unassigned signer"):
            build_plan(self.metadata)


if __name__ == "__main__":
    unittest.main()
