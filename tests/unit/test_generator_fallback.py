import sys, pathlib, json
import types
import pytest
from click.testing import CliRunner

SRC = pathlib.Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(SRC.resolve()))

# We will monkeypatch transformers + peft lightweight stubs

class DummyModel:
    def __init__(self):
        self.device = 'cpu'
    def generate(self, **kwargs):
        import torch
        # Return a tensor shaped like (1, n) with eos token id 0
        return torch.tensor([[0, 1, 2]])
    def parameters(self):
        yield types.SimpleNamespace(device='cpu')

class DummyTokenizer:
    eos_token = '<eos>'
    eos_token_id = 0
    pad_token = None
    def __init__(self):
        self.saved = False
    def __call__(self, text, return_tensors=None, padding=None, truncation=None, max_length=None):
        import torch
        return {'input_ids': torch.tensor([[1,2,3]])}
    def decode(self, ids, skip_special_tokens=True):
        return 'Prompt: Generated content'
    def save_pretrained(self, path):
        self.saved = True
    @staticmethod
    def from_pretrained(path, trust_remote_code=True):
        return DummyTokenizer()

class DummyConfig:
    def to_json_file(self, path):
        pathlib.Path(path).write_text('{"architectures": ["Dummy"]}')
    @staticmethod
    def from_pretrained(path, trust_remote_code=True):
        return DummyConfig()

@pytest.fixture(autouse=True)
def minimal_torch():
    """Skip tests if torch isn't installed; otherwise no-op."""
    try:  # pragma: no cover - environment dependent
        import torch  # noqa: F401
    except Exception:
        pytest.skip('torch not available in test environment')
    yield

@pytest.fixture
def patch_transformers(monkeypatch):
    mod = types.ModuleType('transformers')
    mod.AutoTokenizer = DummyTokenizer
    mod.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyModel())
    mod.AutoConfig = DummyConfig
    monkeypatch.setitem(sys.modules, 'transformers', mod)
    yield

@pytest.fixture
def patch_peft(monkeypatch):
    peft_mod = types.ModuleType('peft')
    class PeftModel:
        @staticmethod
        def from_pretrained(base, path):
            return base  # no-op
    peft_mod.PeftModel = PeftModel
    monkeypatch.setitem(sys.modules, 'peft', peft_mod)
    yield


def test_generator_fallback_load(tmp_path, patch_transformers, patch_peft):
    # Create version directory with only model_info.json
    version_dir = tmp_path / 'v0.0.1'
    version_dir.mkdir()
    info = {
        'version': 'v0.0.1',
        'base_model': 'dummy/base',
        'checkpoint_dir': str(tmp_path / 'ckpt')
    }
    (version_dir / 'model_info.json').write_text(json.dumps(info))
    # no config.json intentionally
    from toaripi_slm.cli.core.generator import ToaripiGenerator
    gen = ToaripiGenerator(version_dir)
    ok = gen.load()
    assert ok, 'Fallback load should succeed with model_info.json'


def test_materialize_command(tmp_path, patch_transformers, patch_peft, monkeypatch):
    # Prepare registry + model version
    models_root = tmp_path / 'models' / 'hf'
    models_root.mkdir(parents=True)
    version = 'v0.0.2'
    version_dir = models_root / version
    version_dir.mkdir()
    info = {
        'version': version,
        'base_model': 'dummy/base',
        'checkpoint_dir': str(tmp_path / 'ckpt')
    }
    (version_dir / 'model_info.json').write_text(json.dumps(info))
    registry = { 'models': [ { 'version': version, 'path': str(version_dir), 'base_model': 'dummy/base', 'created_at': 'now', 'checkpoint_dir': info['checkpoint_dir'] } ] }
    reg_file = models_root / 'registry.json'
    reg_file.write_text(json.dumps(registry))

    # Point CLI to our temp registry by monkeypatching path constant
    from toaripi_slm.cli.commands import models as models_mod
    monkeypatch.setattr(models_mod, 'REGISTRY_FILE', reg_file)

    runner = CliRunner()
    result = runner.invoke(models_mod.models, ['materialize', version])
    assert result.exit_code == 0, result.output
    # After materialization expect config.json to exist
    assert (version_dir / 'config.json').exists()
    # Loading via generator local path should now take materialized branch
    from toaripi_slm.cli.core.generator import ToaripiGenerator
    gen = ToaripiGenerator(version_dir)
    ok = gen.load(allow_fallback=False)
    assert ok, 'Direct load should succeed after materialization'
