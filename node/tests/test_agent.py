"""节点端辅助函数测试（无第三方依赖，仅测 _to_bool 等纯函数）。

运行：
  cd node && python -m unittest discover -s tests -v
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_NODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_NODE_DIR))

# agent.py 顶层 import 标准库 + 定义函数，但不 import bubble_ocr（在 run_inference 内才 import），
# 故可直接 import 测试纯函数。
import agent  # noqa: E402


class ToBoolTest(unittest.TestCase):
    def test_string_false(self):
        # 关键修复：bool("false") 会得到 True，必须用 _to_bool 正确解析
        self.assertFalse(agent._to_bool("false"))
        self.assertFalse(agent._to_bool("False"))
        self.assertFalse(agent._to_bool("FALSE"))
        self.assertFalse(agent._to_bool("0"))
        self.assertFalse(agent._to_bool("no"))
        self.assertFalse(agent._to_bool("off"))
        self.assertFalse(agent._to_bool(""))

    def test_string_true(self):
        self.assertTrue(agent._to_bool("true"))
        self.assertTrue(agent._to_bool("1"))
        self.assertTrue(agent._to_bool("yes"))
        self.assertTrue(agent._to_bool("anything-else"))

    def test_native_bool(self):
        self.assertTrue(agent._to_bool(True))
        self.assertFalse(agent._to_bool(False))

    def test_none_uses_default(self):
        self.assertFalse(agent._to_bool(None, default=False))
        self.assertTrue(agent._to_bool(None, default=True))

    def test_number(self):
        self.assertFalse(agent._to_bool(0))
        self.assertTrue(agent._to_bool(1))


class RuntimeSafetyTest(unittest.TestCase):
    def test_single_instance_lock_rejects_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "node.lock"
            with agent.SingleInstanceLock(path):
                with self.assertRaisesRegex(agent.AgentError, "node_id"):
                    with agent.SingleInstanceLock(path):
                        self.fail("duplicate lock unexpectedly acquired")

    def test_heartbeat_reports_phase_and_progress(self):
        captured = []

        def request(*args, **kwargs):
            captured.append(args[3])
            return {"ok": True}

        heartbeat = agent.JobHeartbeat({}, "cad-20261008-000000-abcdef", "token")
        heartbeat.update("inference", 0.2)
        with patch.object(agent, "request_json", side_effect=request):
            heartbeat.pulse()
        self.assertEqual(captured, [{"phase": "inference", "progress": 0.2}])


if __name__ == "__main__":
    unittest.main()
