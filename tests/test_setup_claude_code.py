import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "setup_claude_code", Path(__file__).parent.parent / "scripts" / "setup_claude_code.py"
)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def test_apply_and_remove_keep_other_settings():
    mine = {"type": "command", "command": "other-hook"}
    settings = {
        "env": {"FOO": "1", "ANTHROPIC_CUSTOM_HEADERS": "X-Team: a"},
        "hooks": {"UserPromptSubmit": [{"hooks": [mine]}]},
        "model": "opus",
    }
    out = setup.apply(settings, "https://gw.example", "k1", remove=False)
    assert out["env"]["ANTHROPIC_BASE_URL"] == "https://gw.example"
    assert out["env"]["ANTHROPIC_CUSTOM_HEADERS"] == "X-Team: a\nX-EcoRoute-Key: k1"
    assert len(out["hooks"]["UserPromptSubmit"]) == 2
    again = setup.apply(out, "https://gw.example", "k1", remove=False)  # idempotent
    assert len(again["hooks"]["UserPromptSubmit"]) == 2
    assert again["env"]["ANTHROPIC_CUSTOM_HEADERS"].count("X-EcoRoute-Key") == 1
    back = setup.apply(again, None, None, remove=True)
    assert back == {
        "env": {"FOO": "1", "ANTHROPIC_CUSTOM_HEADERS": "X-Team: a"},
        "hooks": {"UserPromptSubmit": [{"hooks": [mine]}]},
        "model": "opus",
    }
