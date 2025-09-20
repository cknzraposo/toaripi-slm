import sys, pathlib, json
import pytest
from click.testing import CliRunner

SRC = pathlib.Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(SRC.resolve()))

from toaripi_slm.cli.core.token_weights import SimulatedTokenWeightProvider  # noqa:E402
from toaripi_slm.cli.core.display import BilingualDisplay  # noqa:E402
from toaripi_slm.cli.core.session import InteractiveSession  # noqa:E402
from toaripi_slm.cli.core.versioning import load_registry, resolve_version_dir  # noqa:E402
from toaripi_slm.cli.commands.sessions import sessions  # noqa:E402
from toaripi_slm.cli.core.exporter import generate_model_card, prepare_export  # noqa:E402


def test_token_weight_provider_basic():
    provider = SimulatedTokenWeightProvider()
    weights = provider.weights_for(["ruru/dog", "the", "fishing"], generated=True)
    assert len(weights) == 3
    assert any(w.weight >= 0.8 for w in weights), "bilingual pair should have high weight"


def test_bilingual_display_toggle():
    d = BilingualDisplay()
    initial = d.show_weights
    d.toggle_weights()
    assert d.show_weights != initial


def test_interactive_session_save(tmp_path):
    d = BilingualDisplay()
    session = InteractiveSession(display=d)
    session.add(user="hi", english="Generated story for: hi", toaripi="toaripi text", content_type="story")
    out = tmp_path / 'session_test.json'
    session.save(out)
    data = json.loads(out.read_text())
    assert data['total_exchanges'] == 1


def test_versioning_handles_empty_registry(tmp_path, monkeypatch):
    # Point registry path to a temp file that does not exist
    from toaripi_slm.cli.core import versioning
    monkeypatch.setattr(versioning, 'REGISTRY_PATH', tmp_path / 'registry.json')
    reg = load_registry()
    assert reg.get('models') == []
    assert resolve_version_dir(None) is None


def test_sessions_list_empty(tmp_path, monkeypatch):
    runner = CliRunner()
    from toaripi_slm.cli.commands import sessions as sessions_mod
    monkeypatch.setattr(sessions_mod, 'SESSIONS_DIR', tmp_path)
    result = runner.invoke(sessions, ['list'])
    assert result.exit_code == 0
    assert 'No sessions' in result.output


def test_generate_model_card_content():
    meta = {
        'version': 'v0.0.9',
        'base_model': 'mistralai/Mistral-7B-Instruct-v0.2',
        'created_at': '2025-09-20T00:00:00',
        'checkpoint_dir': './checkpoints'
    }
    card = generate_model_card(meta, quantization='q4_k_m')
    assert 'Toaripi Educational SLM v0.0.9' in card
    assert 'q4_k_m' in card


def test_prepare_export_creates_manifest(tmp_path):
    meta = {
        'version': 'v0.0.10',
        'path': './models/hf/v0.0.10',
        'base_model': 'base',
        'created_at': '2025-09-20T00:00:00',
        'checkpoint_dir': './checkpoints'
    }
    out_dir = prepare_export(meta, export_root=tmp_path, include_card=True, quantization='q4_k_m')
    manifest = (out_dir / 'export_manifest.json')
    assert manifest.exists()
    data = json.loads(manifest.read_text())
    assert data['version'] == 'v0.0.10'
    # New checksum + quantization fields (may be empty if files missing)
    assert 'checksums' in data
    assert 'quantization_placeholder' in data
    qp = data['quantization_placeholder']
    assert qp is None or qp.get('status') == 'placeholder'


def test_prepare_export_checksums(tmp_path):
    # Create fake model dir with minimal files
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    (model_dir / 'config.json').write_text('{"architectures": ["Test"]}')
    (model_dir / 'tokenizer.json').write_text('{}')
    meta = {
        'version': 'v0.0.11',
        'path': str(model_dir),
        'base_model': 'base',
    }
    out_dir = prepare_export(meta, export_root=tmp_path, include_card=False, quantization='q4_k_m')
    data = json.loads((out_dir / 'export_manifest.json').read_text())
    checksums = data['checksums']
    assert 'config.json' in checksums
    assert checksums['config.json'].startswith('sha256:')
    assert data['quantization_placeholder']['status'] == 'placeholder'
