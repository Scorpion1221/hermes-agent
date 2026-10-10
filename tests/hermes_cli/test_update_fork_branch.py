"""A GitHub fork checkout follows its own branch: real Git, real ``hermes update`` / ``--check``.

Hosted channel records name the official repository as their authority, so a fork that resolved
the default ``main`` channel either fell back to its own (stale) ``main`` or failed the authority
check and exited 1. The fork follows the branch it is checked out on instead, with no channel read.
"""
import pytest

from hermes_cli import main as cli_main, source_releases
from tests.hermes_cli.test_update_target_identity import git, update_tree  # noqa: F401 -- fixture

FORK_URL = "https://github.com/fork-owner/hermes-agent.git"
OFFICIAL_URL = "https://github.com/NousResearch/hermes-agent.git"


@pytest.fixture
def fork(update_tree, monkeypatch):  # noqa: F811 -- the imported fixture
    """``update_tree`` whose origin carries ``develop`` and ``next`` beside ``main``, each its own tip."""
    t = update_tree
    # The owning-install redirect inspects the TEST RUNNER's venv (outside this fixture's control;
    # a PM test environment sits under the real home), not the fixture checkout under test.
    monkeypatch.setattr("hermes_cli.update_owning_install.owning_install_root", lambda _root: None)
    tips = {"main": t.newer}
    for branch in ("develop", "next"):
        git(t.origin, "checkout", "-qb", branch, t.base)
        (t.origin / f"{branch}.txt").write_text(f"{branch} tip\n", encoding="utf-8")
        git(t.origin, "add", f"{branch}.txt")
        git(t.origin, "-c", "commit.gpgsign=false", "commit", "-qm", f"{branch} tip")
        tips[branch] = git(t.origin, "rev-parse", "HEAD")
    git(t.origin, "checkout", "-q", "main")
    git(t.clone, "fetch", "-q", "origin")
    channel_reads = []
    stub = source_releases._resolve_channel  # conftest's local source-branch stub

    def recording(name, repository):
        channel_reads.append((name, repository))
        return stub(name, repository)

    monkeypatch.setattr(source_releases, "_resolve_channel", recording)
    t.args.channel = None  # no --channel: the install's default subscription (main)
    t.args.gateway = True  # gateway /update spawns exactly `hermes update --gateway`

    def point_origin(url):
        # The GitHub URL is what the updater reads; Git still talks to the local fixture origin.
        git(t.clone, "config", "remote.origin.url", url)
        git(t.clone, "config", f"url.{t.origin}.insteadOf", url)

    def park(branch):
        git(t.clone, "checkout", "-qB", branch, t.base)

    t.tips, t.channel_reads, t.point_origin, t.park = tips, channel_reads, point_origin, park
    return t


@pytest.mark.parametrize("branch", ["develop", "next", "main"])
def test_fork_update_follows_its_checked_out_branch_without_a_channel(fork, monkeypatch, branch):
    fork.point_origin(FORK_URL)
    fork.park(branch)
    if branch == "main":  # a fork on main still gets the fork-upstream sync (no upstream remote here)
        monkeypatch.setattr(cli_main, "_sync_with_upstream_if_needed", lambda *_a, **_k: True)

    cli_main.cmd_update(fork.args)

    request, = fork.requests
    assert request["branch"] == branch
    assert git(fork.clone, "rev-parse", "HEAD") == fork.tips[branch]
    assert git(fork.clone, "branch", "--show-current") == branch
    assert fork.channel_reads == []


def test_fork_check_reports_against_its_checked_out_branch(fork, capsys):
    fork.point_origin(FORK_URL)
    fork.park("next")
    fork.args.check = True

    cli_main.cmd_update(fork.args)

    output = capsys.readouterr().out
    assert "behind origin/next" in output
    assert git(fork.clone, "rev-parse", "HEAD") == fork.base
    assert fork.requests == []
    assert fork.channel_reads == []


@pytest.mark.parametrize("check", [False, True])
def test_fork_on_a_branch_origin_lacks_stops_instead_of_moving_to_main(fork, capsys, check):
    fork.point_origin(FORK_URL)
    fork.park("local-only")
    fork.args.check = check

    with pytest.raises(SystemExit) as stop:
        cli_main.cmd_update(fork.args)

    assert stop.value.code == 1
    assert "fork-owner/hermes-agent" in capsys.readouterr().out
    assert git(fork.clone, "rev-parse", "HEAD") == fork.base
    assert git(fork.clone, "branch", "--show-current") == "local-only"
    assert fork.requests == []
    assert fork.channel_reads == []


def test_explicit_branch_still_wins_on_a_fork(fork):
    fork.point_origin(FORK_URL)
    fork.park("develop")
    fork.args.branch = "next"

    cli_main.cmd_update(fork.args)

    assert git(fork.clone, "rev-parse", "HEAD") == fork.tips["next"]
    assert git(fork.clone, "branch", "--show-current") == "next"
    assert fork.channel_reads == []


def test_official_origin_keeps_resolving_the_channel(fork, monkeypatch):
    """Same checkout state, official origin: the main channel decides and the update targets main."""
    from hermes_cli.update_cmd import _sync_with_upstream_if_needed

    fork.point_origin(OFFICIAL_URL)
    fork.park("develop")
    monkeypatch.setattr(cli_main, "_sync_with_upstream_if_needed", _sync_with_upstream_if_needed)

    cli_main.cmd_update(fork.args)

    request, = fork.requests
    assert request["branch"] == "main"
    assert git(fork.clone, "rev-parse", "HEAD") == fork.tips["main"]
    assert fork.channel_reads == [("main", "NousResearch/hermes-agent")]
