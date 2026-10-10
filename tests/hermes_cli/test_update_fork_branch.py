"""A GitHub fork checkout follows its own branch: real Git, real ``hermes update`` / ``--check``.

Hosted channel records name the official repository as their authority, so a fork that resolved
the default ``main`` channel either fell back to its own (stale) ``main`` or failed the authority
check and exited 1. The fork follows the branch it is checked out on instead, with no channel read.
"""
import pytest

from hermes_cli import main as cli_main, source_releases
from hermes_cli.update_channel import set_install_channel
from hermes_constants import get_hermes_home
from tests.hermes_cli.test_update_target_identity import git, update_tree  # noqa: F401 -- fixture

FORK_URL = "https://github.com/fork-owner/hermes-agent.git"
OFFICIAL_URL = "https://github.com/NousResearch/hermes-agent.git"


@pytest.fixture
def fork(update_tree, monkeypatch):  # noqa: F811 -- the imported fixture
    """``update_tree`` whose origin carries ``develop``, ``next`` and ``preview`` beside ``main``, each its own tip."""
    t = update_tree
    # The owning-install redirect inspects the TEST RUNNER's venv (outside this fixture's control;
    # a PM test environment sits under the real home), not the fixture checkout under test.
    monkeypatch.setattr("hermes_cli.update_owning_install.owning_install_root", lambda _root: None)
    tips = {"main": t.newer}
    for branch in ("develop", "next", "preview"):
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


# GitHub spellings the channel-authority parser reads as the official repository.
FORK_URL_SPELLINGS = [
    "https://x-access-token:TOKEN@github.com/fork-owner/hermes-agent.git",
    "https://fork-owner@github.com/fork-owner/hermes-agent",
    "ssh://git@github.com:22/fork-owner/hermes-agent.git",
    "ssh://git@ssh.github.com:443/fork-owner/hermes-agent.git",
]


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


@pytest.mark.parametrize("url", FORK_URL_SPELLINGS)
def test_fork_with_credentials_or_a_port_in_its_url_still_follows_its_branch(fork, url):
    fork.point_origin(url)
    fork.park("develop")

    cli_main.cmd_update(fork.args)

    request, = fork.requests
    assert request["branch"] == "develop"
    assert git(fork.clone, "rev-parse", "HEAD") == fork.tips["develop"]
    assert git(fork.clone, "branch", "--show-current") == "develop"
    assert fork.channel_reads == []


def test_official_origin_with_credentials_keeps_resolving_the_channel(fork, monkeypatch):
    from hermes_cli.update_cmd import _sync_with_upstream_if_needed

    fork.point_origin("https://x-access-token:TOKEN@github.com/NousResearch/hermes-agent.git")
    fork.park("main")
    monkeypatch.setattr(cli_main, "_sync_with_upstream_if_needed", _sync_with_upstream_if_needed)

    cli_main.cmd_update(fork.args)

    assert fork.channel_reads == [("main", "NousResearch/hermes-agent")]


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
@pytest.mark.parametrize("branch", ["local-only", "next"])
def test_fork_on_a_branch_origin_lacks_stops_instead_of_moving_to_main(fork, monkeypatch, capsys, check, branch):
    """``local-only`` was never pushed; ``next`` was deleted (or renamed) on the fork, so only a
    stale ``refs/remotes/origin/next`` is left: the fetch-time refspec never prunes it."""
    from hermes_cli import update_receipt

    stops = []
    monkeypatch.setattr(update_receipt, "record_stop_reason", stops.append)
    fork.point_origin(FORK_URL)
    fork.park(branch)
    if branch == "next":
        git(fork.origin, "branch", "-D", "next")
    fork.args.check = check

    with pytest.raises(SystemExit) as stop:
        cli_main.cmd_update(fork.args)

    assert stop.value.code == 1
    assert "✗ This checkout tracks the fork fork-owner/hermes-agent" in capsys.readouterr().out
    assert stops == ([] if check else ["branch_missing"])
    assert git(fork.clone, "rev-parse", "HEAD") == fork.base
    assert git(fork.clone, "branch", "--show-current") == branch
    assert fork.requests == []
    assert fork.channel_reads == []


@pytest.mark.parametrize("check", [False, True])
def test_single_branch_fork_follows_a_branch_it_has_no_tracking_ref_for(fork, capsys, check):
    """A ``--depth 1 --branch develop`` installer clone maps only develop; moved to ``next`` via
    ``git fetch origin next && git switch -c next FETCH_HEAD``, it has no refs/remotes/origin/next."""
    fork.point_origin(FORK_URL)
    git(fork.clone, "config", "remote.origin.fetch", "+refs/heads/develop:refs/remotes/origin/develop")
    git(fork.clone, "update-ref", "-d", "refs/remotes/origin/next")
    fork.park("next")
    fork.args.check = check

    cli_main.cmd_update(fork.args)

    if check:
        assert "behind origin/next" in capsys.readouterr().out
        assert fork.requests == []
    else:
        request, = fork.requests
        assert request["branch"] == "next"
        assert git(fork.clone, "rev-parse", "HEAD") == fork.tips["next"]
        assert git(fork.clone, "branch", "--show-current") == "next"
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


def _commit_local_patch(fork):
    (fork.clone / "local-patch.txt").write_text("carried on top of main\n", encoding="utf-8")
    git(fork.clone, "add", "local-patch.txt")
    git(fork.clone, "-c", "commit.gpgsign=false", "commit", "-qm", "local patch")
    return git(fork.clone, "rev-parse", "HEAD")


@pytest.mark.parametrize("pushed", [True, False])
def test_fork_with_update_in_place_merges_main_into_its_branch(fork, monkeypatch, pushed):
    """``updates.parked_branch_strategy: update_in_place`` is the documented way to carry local
    patches on main; a fork that set it keeps getting main merged into its branch."""
    (get_hermes_home() / "config.yaml").write_text(
        "updates:\n  parked_branch_strategy: update_in_place\n", encoding="utf-8")
    fork.point_origin(FORK_URL)
    fork.park("custom")
    patch = _commit_local_patch(fork)
    if pushed:
        git(fork.clone, "push", "-q", "origin", "custom")
    monkeypatch.setattr(cli_main, "_sync_with_upstream_if_needed", lambda *_a, **_k: True)

    cli_main.cmd_update(fork.args)

    request, = fork.requests
    assert request["branch"] == "main"
    assert git(fork.clone, "branch", "--show-current") == "custom"
    for kept in (fork.tips["main"], patch):
        git(fork.clone, "merge-base", "--is-ancestor", kept, "HEAD")  # raises when not merged
    assert fork.channel_reads == [("main", "fork-owner/hermes-agent")]


def test_fork_check_with_update_in_place_compares_against_main(fork, capsys):
    (get_hermes_home() / "config.yaml").write_text(
        "updates:\n  parked_branch_strategy: update_in_place\n", encoding="utf-8")
    fork.point_origin(FORK_URL)
    fork.park("custom")
    _commit_local_patch(fork)
    git(fork.clone, "push", "-q", "origin", "custom")
    fork.args.check = True

    cli_main.cmd_update(fork.args)

    assert "behind origin/main" in capsys.readouterr().out
    assert fork.requests == []


@pytest.mark.parametrize("persisted", [False, True])
def test_fork_keeps_an_explicitly_chosen_channel(fork, persisted):
    """``--channel preview`` and a persisted ``--set-channel preview`` are the user's choice: the
    channel is read and its branch wins over the branch the fork is checked out on."""
    fork.point_origin(FORK_URL)
    fork.park("develop")
    if persisted:
        set_install_channel("preview", fork.clone)
    else:
        fork.args.channel = "preview"

    cli_main.cmd_update(fork.args)

    request, = fork.requests
    assert request["branch"] == "preview"
    assert git(fork.clone, "rev-parse", "HEAD") == fork.tips["preview"]
    assert fork.channel_reads == [("preview", "fork-owner/hermes-agent")]
