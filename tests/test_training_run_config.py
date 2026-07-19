from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.training_run_config import load_training_run_config


DEFAULTS = {
    "formulation": "total",
    "random_seed": 0,
    "show_plots": True,
    "output_root": "legacy",
    "steps": 10,
}


class TrainingRunConfigTest(unittest.TestCase):
    def test_empty_argv_preserves_defaults_without_managing_output(self) -> None:
        config = load_training_run_config(DEFAULTS, formulation="total")
        self.assertEqual(config.values, DEFAULTS)
        self.assertFalse(config.managed_output)

    def test_json_and_cli_overrides_are_hashed_and_archived(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "config.json"
            source.write_text(
                json.dumps({"random_seed": 2, "output_root": "unused", "steps": 20}),
                encoding="utf-8",
            )
            config = load_training_run_config(
                DEFAULTS,
                formulation="total",
                argv=(
                    "--config",
                    str(source),
                    "--seed",
                    "3",
                    "--output-root",
                    "run",
                    "--no-show-plots",
                ),
            )
            self.assertEqual(config.values["random_seed"], 3)
            self.assertEqual(config.values["steps"], 20)
            self.assertFalse(config.values["show_plots"])
            self.assertEqual(len(config.identifier), 64)

            output_root = config.prepare_output_root(root)
            archived = json.loads((output_root / "run_config.json").read_text())
            self.assertEqual(archived["config_id"], config.identifier)
            self.assertEqual(archived["values"], config.values)
            with self.assertRaises(FileExistsError):
                config.prepare_output_root(root)

    def test_unknown_key_and_wrong_formulation_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            unknown = root / "unknown.json"
            unknown.write_text(json.dumps({"typo": 1}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown"):
                load_training_run_config(
                    DEFAULTS,
                    formulation="total",
                    argv=("--config", str(unknown)),
                )

            wrong = root / "wrong.json"
            wrong.write_text(json.dumps({"formulation": "scattered"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "formulation"):
                load_training_run_config(
                    DEFAULTS,
                    formulation="total",
                    argv=("--config", str(wrong)),
                )


if __name__ == "__main__":
    unittest.main()
