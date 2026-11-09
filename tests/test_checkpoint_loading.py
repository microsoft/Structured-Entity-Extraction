import ast
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


def _write_marker(path):
    Path(path).touch()
    return {}


class _PickledPayload:
    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return _write_marker, (self.marker,)


class CheckpointLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / "experiment_musee.py"
        tree = ast.parse(source.read_text())
        # Exercise the actual load expression without importing training dependencies.
        loads = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "model"
            and node.func.attr == "load_state_dict"
        ]
        if len(loads) != 1:
            raise AssertionError("Expected one model.load_state_dict call")
        cls.load_checkpoint = compile(ast.Expression(body=loads[0]), str(source), "eval")

    def load(self, model, save_path):
        return eval(
            self.load_checkpoint,
            {
                "model": model,
                "torch": torch,
                "save_path": str(save_path),
                "device": torch.device("cpu"),
            },
        )

    def test_loads_state_dict_with_explicit_weights_only(self):
        original = torch.nn.Linear(2, 1)
        restored = torch.nn.Linear(2, 1)
        with tempfile.TemporaryDirectory() as directory:
            save_path = Path(directory) / "best_model"
            torch.save(original.state_dict(), f"{save_path}.pt")
            with mock.patch.object(torch, "load", wraps=torch.load) as load:
                self.load(restored, save_path)
            load.assert_called_once_with(
                f"{save_path}.pt",
                map_location=torch.device("cpu"),
                weights_only=True,
            )
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]))

    def test_rejects_pickled_callable_without_executing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            save_path = Path(directory) / "best_model"
            marker = Path(directory) / "payload_executed"
            torch.save(_PickledPayload(marker), f"{save_path}.pt")
            with self.assertRaises(pickle.UnpicklingError):
                self.load(torch.nn.Linear(2, 1), save_path)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
