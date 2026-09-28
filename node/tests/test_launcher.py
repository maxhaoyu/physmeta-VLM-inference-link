import sys
from pathlib import Path
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import launch_node

class LauncherTest(unittest.TestCase):
    def test_only_unexpected_crashes_restart(self):
        for codes,expected_calls in [([0],1),([1],1),([2,2,0],3),([2]*11,11)]:
            calls=[];values=iter(codes)
            result=launch_node.supervise(['python','agent.py'],run=lambda c:(calls.append(c),next(values))[1],delay=lambda _:None)
            self.assertEqual(len(calls),expected_calls);self.assertEqual(result,codes[-1])
    def test_reuses_configured_bubble_environment_with_spaces(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'Existing CUDA env';python=root/'.venv/Scripts/python.exe';python.parent.mkdir(parents=True);python.touch()
            self.assertEqual(launch_node.interpreter({'bubble_src':str(root/'src')},Path(td)),python)
