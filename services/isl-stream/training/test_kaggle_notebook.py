from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK = ROOT / "notebooks" / "signbridge_free_asl_training.ipynb"
TRAINING = ROOT / "services" / "isl-stream" / "training"


class KaggleNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.cells = cls.notebook["cells"]

    def test_notebook_format_and_training_scripts_exist(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        for filename in (
            "download_epee_keypoints.py",
            "prepare_epee_keypoints.py",
            "train_ctc.py",
            "model.py",
            "feature_validation.py",
        ):
            with self.subTest(filename=filename):
                self.assertTrue((TRAINING / filename).is_file())

    def test_python_cells_are_syntactically_valid(self):
        for index, cell in enumerate(self.cells):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            # IPython magics are valid in Kaggle/Jupyter but not Python's AST.
            normalized = "\n".join(
                "pass" if line.lstrip().startswith("%") else line
                for line in source.splitlines()
            )
            with self.subTest(cell=index):
                ast.parse(normalized, filename=f"kaggle-notebook-cell-{index}")

    def test_notebook_uses_free_gpu_and_does_not_fetch_rgb_videos(self):
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in self.cells
            if cell["cell_type"] == "code"
        )
        self.assertIn("--user-attests-rights", source)
        self.assertIn("--language', 'ASL'", source)
        self.assertIn("'--causal'", source)
        self.assertIn("videos_downloaded", source)
        self.assertIn("production_ready", source)
        self.assertIn("hf_hub_download", source)
        self.assertIn("unseen_heldout_gloss_tokens", source)
        self.assertIn("Preflight stopped before the large keypoint download", source)
        self.assertIn("Expected 2,400 audited metadata rows", source)
        self.assertIn("expected_split_counts = {'train': 1200, 'validation': 600, 'test': 600}", source)
        self.assertIn("Signer leakage detected across splits", source)
        self.assertLess(source.index("unseen_heldout_gloss_tokens"), source.index("download_epee_keypoints.py"))
        self.assertNotIn("HF_TOKEN", source)
        self.assertNotIn("RENDER_API_KEY", source)


if __name__ == "__main__":
    unittest.main()
