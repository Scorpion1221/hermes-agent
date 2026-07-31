"""Install-source and reinstall-hint tests for Scorpion1221/develop maintenance policy."""

from pathlib import Path


def test_install_script_defaults_to_fork_develop():
    text = Path("scripts/install.sh").read_text()
    assert 'REPO_URL_SSH="git@github.com:Scorpion1221/hermes-agent.git"' in text
    assert 'REPO_URL_HTTPS="https://github.com/Scorpion1221/hermes-agent.git"' in text
    assert 'BRANCH="develop"' in text
    assert "raw.githubusercontent.com/Scorpion1221/hermes-agent/develop/scripts/install.sh" in text
    assert "default: develop" in text


def test_uninstall_reinstall_hint_points_to_fork_develop():
    text = Path("hermes_cli/uninstall.py").read_text()
    assert "raw.githubusercontent.com/Scorpion1221/hermes-agent/develop/scripts/install.sh" in text


def test_update_reinstall_hint_points_to_fork_develop():
    from hermes_cli.main import FORK_INSTALL_URL

    assert (
        FORK_INSTALL_URL
        == "https://raw.githubusercontent.com/Scorpion1221/hermes-agent/"
        "develop/scripts/install.sh"
    )
