import os
import tempfile
import unittest

from seethrough_engine.paths import resolve_model_path, scan_model_dirs


class ResolveModelPathTests(unittest.TestCase):
    def test_local_folder_wins_over_download(self):
        with tempfile.TemporaryDirectory() as models_dir:
            local = os.path.join(models_dir, "seethroughv0.0.2_layerdiff3d")
            os.makedirs(local)
            resolved = resolve_model_path("layerdifforg/seethroughv0.0.2_layerdiff3d", models_dir)
            self.assertEqual(resolved, local)

    def test_missing_repo_falls_back_to_name_without_crashing(self):
        # No local folder, and huggingface_hub is not installed in this
        # environment (or the download fails offline) -- either way this
        # must degrade to returning the model name, not raise.
        with tempfile.TemporaryDirectory() as models_dir:
            resolved = resolve_model_path("org/does-not-exist-xyz", models_dir)
            self.assertEqual(resolved, "org/does-not-exist-xyz")

    def test_plain_name_without_slash_is_returned_as_is(self):
        with tempfile.TemporaryDirectory() as models_dir:
            resolved = resolve_model_path("some_local_only_name", models_dir)
            self.assertEqual(resolved, "some_local_only_name")


class ScanModelDirsTests(unittest.TestCase):
    def test_missing_directory_returns_empty_list(self):
        self.assertEqual(scan_model_dirs("/does/not/exist"), [])

    def test_lists_only_subdirectories_sorted(self):
        with tempfile.TemporaryDirectory() as models_dir:
            os.makedirs(os.path.join(models_dir, "b_model"))
            os.makedirs(os.path.join(models_dir, "a_model"))
            with open(os.path.join(models_dir, "not_a_dir.txt"), "w") as f:
                f.write("x")
            self.assertEqual(scan_model_dirs(models_dir), ["a_model", "b_model"])


if __name__ == "__main__":
    unittest.main()
