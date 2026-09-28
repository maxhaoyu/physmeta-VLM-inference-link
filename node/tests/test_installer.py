import hashlib,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import install_update
class InstallerTest(unittest.TestCase):
    def test_verified_update_preserves_private_config_and_backs_up_code(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);bundle=root/'bundle';target=root/'existing';src=root/'ocr/src';bundle.mkdir();target.mkdir();(src/'bubble_ocr').mkdir(parents=True)
            original=b'{"token":"keep-private","bubble_src":'+json.dumps(str(src)).encode()+b'}'
            (target/'inference-node.json').write_bytes(original);(target/'agent.py').write_text('old agent');(src/'bubble_ocr/pipeline.py').write_text('old pipeline')
            for name in ['agent.py','launch_node.py','start-node.bat','vlm_fallback_windows.py']:(bundle/name).write_text('new '+name)
            patchfile=bundle/'bubble-src/bubble_ocr/pipeline.py';patchfile.parent.mkdir(parents=True);patchfile.write_text('new pipeline')
            manifest={str(p.relative_to(bundle)):hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.rglob('*') if p.is_file()}
            (bundle/'bundle-manifest.json').write_text(json.dumps({'files':manifest}))
            with patch.object(install_update,'HERE',bundle),patch.object(install_update,'active_node',return_value=False):install_update.install(target)
            self.assertEqual((target/'inference-node.json').read_bytes(),original)
            self.assertEqual((target/'agent.py').read_text(),'new agent.py')
            self.assertEqual((src/'bubble_ocr/pipeline.py').read_text(),'new pipeline')
            self.assertIn('old agent',[p.read_text() for p in (target/'code-backups').rglob('*') if p.is_file()])
            (bundle/'agent.py').write_text('corrupted')
            with patch.object(install_update,'HERE',bundle),patch.object(install_update,'active_node',return_value=False):
                with self.assertRaisesRegex(ValueError,'verification failed'):install_update.install(target)
            self.assertEqual((target/'agent.py').read_text(),'new agent.py')
