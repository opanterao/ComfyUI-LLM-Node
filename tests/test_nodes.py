import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock


def install_dependency_stubs():
    requests_module = types.ModuleType("requests")
    requests_module.exceptions = types.SimpleNamespace(
        RequestException=Exception,
        Timeout=TimeoutError,
    )
    requests_module.get = Mock()
    requests_module.post = Mock()
    sys.modules["requests"] = requests_module

    numpy_module = types.ModuleType("numpy")
    numpy_module.uint8 = "uint8"
    numpy_module.clip = Mock(side_effect=lambda x, a, b: x)
    sys.modules["numpy"] = numpy_module

    torch_module = types.ModuleType("torch")
    torch_module.float32 = "float32"
    torch_module.Tensor = type("Tensor", (), {})
    torch_module.zeros = Mock(return_value="placeholder-image")
    torch_module.from_numpy = Mock(return_value="tensor")
    sys.modules["torch"] = torch_module

    pil = types.ModuleType("PIL")
    pil_img = types.ModuleType("PIL.Image")
    pil_img.open = Mock()
    pil_img.fromarray = Mock()
    pil.Image = pil_img
    sys.modules["PIL"] = pil
    sys.modules["PIL.Image"] = pil_img


install_dependency_stubs()

root = Path(__file__).resolve().parents[1]

pkg = types.ModuleType("comfy_nodes_test")
pkg.__path__ = [str(root)]
sys.modules[pkg.__name__] = pkg

for sub in ("config", "api_client", "node", "nodes"):
    spec = importlib.util.spec_from_file_location(
        f"{pkg.__name__}.{sub}", root / f"{sub}.py", submodule_search_locations=[str(root)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

node_mod = sys.modules[f"{pkg.__name__}.node"]
nodes_mod = sys.modules[f"{pkg.__name__}.nodes"]
api_mod = sys.modules[f"{pkg.__name__}.api_client"]


def _default_params(**overrides):
    params = dict(
        reasoning_effort="auto",
        max_tokens=-1,
        max_context_tokens=-1,
        seed=0,
        top_p=1.0,
        top_k=-1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop="",
    )
    params.update(overrides)
    return params


class RefDescribeTests(unittest.TestCase):
    def setUp(self):
        self.post = Mock()
        api_mod.requests.post = self.post
        api_mod.requests.get = Mock()

    def _success_response(self, content):
        r = Mock()
        r.raise_for_status = Mock()
        r.json.return_value = {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        return r

    def test_describes_each_image_then_combines(self):
        import torch as real_torch
        img = real_torch.zeros((1, 8, 8, 3), dtype=real_torch.float32)
        nodes_mod.image_to_base64 = Mock(return_value="fakebase64data")

        calls = {"n": 0}

        def side_effect(*args, **kwargs):
            payload = kwargs.get("json", {})
            calls["n"] += 1
            content = payload.get("messages", [])[-1].get("content")
            if isinstance(content, list):  # image-describe call
                return self._success_response("described content")
            return self._success_response("FINAL OUTPUT")  # combine call

        self.post.side_effect = side_effect
        out = nodes_mod.CSYIDCRefDescribeNode().describe(
            base_url="https://x/v1", api_key="k",
            system_prompt="combine", image_instruction="describe it",
            model="m", temperature=0.7, request_timeout=120,
            ref_image_1=img, ref_image_2=img,
            **_default_params(),
        )
        self.assertEqual(out[0], "FINAL OUTPUT")
        self.assertEqual(calls["n"], 3)  # 2 describes + 1 combine

    def test_ref_describe_requires_image(self):
        out = nodes_mod.CSYIDCRefDescribeNode().describe(
            base_url="https://x/v1", api_key="k",
            system_prompt="s", image_instruction="d",
            model="m", temperature=0.7, request_timeout=120,
            ref_image_1=None,
            **_default_params(),
        )
        self.assertIn("at least one reference image", out[0])
        self.post.assert_not_called()

    def test_ref_describe_enforces_max_images(self):
        import torch as real_torch
        img = real_torch.zeros((1, 8, 8, 3), dtype=real_torch.float32)
        nodes_mod.image_to_base64 = Mock(return_value="fakebase64data")
        kwargs = {f"ref_image_{i}": img for i in range(2, 17)}
        out = nodes_mod.CSYIDCRefDescribeNode().describe(
            base_url="https://x/v1", api_key="k",
            system_prompt="s", image_instruction="d",
            model="m", temperature=0.7, request_timeout=120,
            ref_image_1=img, **_default_params(), **kwargs,
        )
        self.assertIn("too many images", out[0])
        self.post.assert_not_called()

    def test_registered_only_ref_describe(self):
        self.assertEqual(set(nodes_mod.NODE_CLASS_MAPPINGS), {"CSYIDC-RefDescribe"})


if __name__ == "__main__":
    unittest.main()
