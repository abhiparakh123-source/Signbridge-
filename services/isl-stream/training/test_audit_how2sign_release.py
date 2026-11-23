import tempfile
import unittest
from pathlib import Path

from training.audit_how2sign_release import audit_release, inspect_table


class How2SignAuditTests(unittest.TestCase):
    def test_translation_only_table_is_not_gloss_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "translations.csv"
            path.write_text("id,split,text\nclip-1,train,HELLO\n", encoding="utf-8")
            report = inspect_table(path)
            self.assertFalse(report["candidate_for_manual_review"])
            self.assertEqual(report["translation_columns_not_usable_as_gloss"], ["text"])
            self.assertFalse(report["training_approved"])

    def test_gloss_table_is_candidate_but_never_auto_approved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviewed.csv"
            path.write_text(
                "id,split,signer,gloss,text\n"
                "clip-1,train,s1,HELLO,Hello there\n",
                encoding="utf-8",
            )
            report = inspect_table(path)
            self.assertTrue(report["candidate_for_manual_review"])
            self.assertTrue(report["has_signer_column"])
            self.assertFalse(report["training_approved"])


    def test_gloss_table_with_duplicate_ids_or_missing_signers_is_not_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete.csv"
            path.write_text(
                "id,split,signer,gloss\\n"
                "clip-1,train,s1,HELLO\\n"
                "clip-1,test,,THANK-YOU\\n",
                encoding="utf-8",
            )
            report = inspect_table(path)
            self.assertFalse(report["candidate_for_manual_review"])
            self.assertEqual(report["duplicate_clip_ids"], 1)
            self.assertEqual(report["missing_signer_rows"], 1)
            self.assertEqual(report["missing_or_invalid_split_rows"], 0)

    def test_release_inventory_counts_videos_and_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "videos").mkdir()
            (root / "videos" / "clip.mp4").write_bytes(b"not a real video")
            (root / "annotations.csv").write_text(
                "id,split,gloss\nclip-1,train,HELLO\n", encoding="utf-8"
            )
            report = audit_release(root)
            self.assertEqual(report["video_file_count"], 1)
            self.assertEqual(report["annotation_table_count"], 1)
            self.assertFalse(report["training_ready"])


if __name__ == "__main__":
    unittest.main()
