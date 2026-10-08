import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_windows_bundle.py"
SPEC = importlib.util.spec_from_file_location("build_windows_bundle", SCRIPT)
bundle_builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(bundle_builder)


class BundleLayoutTest(unittest.TestCase):
    def test_archive_exposes_one_user_facing_batch_entrypoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "PhysMeta-Windows-1.1.1"
            payload = root / bundle_builder.INTERNAL_DIR
            payload.mkdir(parents=True)
            (root / bundle_builder.UPGRADE_ENTRYPOINT).write_bytes(
                bundle_builder.upgrade_entrypoint()
            )
            (payload / "start-node.bat").write_text("internal", encoding="ascii")
            destination = Path(temporary) / "bundle.zip"

            bundle_builder.write_zip(root, destination)

            with zipfile.ZipFile(destination) as archive:
                names = archive.namelist()
                prefix = root.name + "/"
                root_files = [
                    name.removeprefix(prefix)
                    for name in names
                    if name.startswith(prefix)
                    and "/" not in name.removeprefix(prefix).rstrip("/")
                    and not name.endswith("/")
                ]
                self.assertEqual(root_files, [bundle_builder.UPGRADE_ENTRYPOINT])
                hidden = archive.getinfo(f"{root.name}/{bundle_builder.INTERNAL_DIR}/")
                self.assertEqual(hidden.external_attr & 0x02, 0x02)

    def test_entrypoint_requires_existing_private_configuration(self):
        script = bundle_builder.upgrade_entrypoint().decode("utf-8")
        self.assertIn("inference-node.json", script)
        self.assertIn("不要在公开更新包中新建生产配置", script)

    def test_public_payload_contains_example_but_not_private_configuration(self):
        self.assertIn("node/inference-node.example.json", bundle_builder.NODE_FILES)
        self.assertNotIn("node/inference-node.json", bundle_builder.NODE_FILES)

    def test_start_script_checks_configuration_before_launching_python(self):
        script = (SCRIPT.parents[1] / "node" / "start-node.bat").read_text(
            encoding="utf-8"
        )
        config_check = script.index('if not exist "%~dp0inference-node.json"')
        python_launch = script.index('"%PHYS_META_PYTHON%"')
        self.assertLess(config_check, python_launch)
        self.assertIn("[无法启动]", script)


if __name__ == "__main__":
    unittest.main()
