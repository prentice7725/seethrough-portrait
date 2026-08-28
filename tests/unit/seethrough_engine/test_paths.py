import os
import tempfile
import unittest
from unittest.mock import patch

from seethrough_engine.paths import resolve_model_path, scan_model_dirs


class ResolveModelPathTests(unittest.TestCase):
    def test_repo_id_calls_snapshot_download_even_when_local_dir_exists(self):
        # A folder existing is not proof the download finished (see the
        # ComfyUI Desktop bug this guards against: an interrupted first
        # download left a folder with only partial files, and the old
        # "if os.path.isdir(local): return local" check meant every later
        # load attempt silently reused that broken folder forever instead
        # of retrying). snapshot_download must always be asked, since it's
        # what actually resumes/completes an interrupted download and is a
        # cheap no-op when everything is already present.
        with tempfile.TemporaryDirectory() as models_dir:
            local = os.path.join(models_dir, "seethroughv0.0.2_layerdiff3d")
            os.makedirs(local)
            with patch("huggingface_hub.snapshot_download") as mock_download:
                resolved = resolve_model_path("layerdifforg/seethroughv0.0.2_layerdiff3d", models_dir)
            self.assertEqual(resolved, local)
            mock_download.assert_called_once_with(
                repo_id="layerdifforg/seethroughv0.0.2_layerdiff3d", local_dir=local,
            )

    def test_download_failure_falls_back_to_existing_local_copy(self):
        with tempfile.TemporaryDirectory() as models_dir:
            local = os.path.join(models_dir, "seethroughv0.0.1_marigold")
            os.makedirs(local)
            with patch("huggingface_hub.snapshot_download", side_effect=OSError("offline")):
                resolved = resolve_model_path("layerdifforg/seethroughv0.0.1_marigold", models_dir)
            self.assertEqual(resolved, local)

    def test_download_failure_without_local_copy_falls_back_to_repo_name(self):
        with tempfile.TemporaryDirectory() as models_dir:
            with patch("huggingface_hub.snapshot_download", side_effect=OSError("offline")):
                resolved = resolve_model_path("org/does-not-exist-xyz", models_dir)
            self.assertEqual(resolved, "org/does-not-exist-xyz")

    def test_missing_huggingface_hub_falls_back_to_existing_local_copy(self):
        with tempfile.TemporaryDirectory() as models_dir:
            local = os.path.join(models_dir, "seethroughv0.0.1_marigold")
            os.makedirs(local)
            with patch.dict("sys.modules", {"huggingface_hub": None}):
                resolved = resolve_model_path("layerdifforg/seethroughv0.0.1_marigold", models_dir)
            self.assertEqual(resolved, local)

    def test_plain_name_without_slash_is_returned_as_is(self):
        with tempfile.TemporaryDirectory() as models_dir:
            resolved = resolve_model_path("some_local_only_name", models_dir)
            self.assertEqual(resolved, "some_local_only_name")

    def test_plain_name_without_slash_uses_local_folder_if_present(self):
        with tempfile.TemporaryDirectory() as models_dir:
            local = os.path.join(models_dir, "already_here")
            os.makedirs(local)
            resolved = resolve_model_path("already_here", models_dir)
            self.assertEqual(resolved, local)


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
