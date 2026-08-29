"""The `evals.validate` command line.

These exist because adding the `judge` command silently demoted the golden-set
check from `python -m evals.validate --strict` to `validate main --strict`,
breaking every documented invocation — and the whole test suite still passed,
because everything else calls `load_all()` directly. The CLI is an interface
too.
"""

from __future__ import annotations

from typer.testing import CliRunner

from evals.validate import app

runner = CliRunner()


def test_the_bare_invocation_validates_the_golden_set():
    result = runner.invoke(app, [])
    assert result.exit_code == 0, result.output
    assert "8/8 valid" in result.output


def test_strict_is_still_a_bare_flag():
    """`python -m evals.validate --strict`, exactly as the README says."""
    result = runner.invoke(app, ["--strict"])
    assert result.exit_code == 0, result.output
    assert "strict" in result.output


def test_the_judge_subcommand_reports_what_is_still_unreviewed():
    result = runner.invoke(app, ["judge"])
    assert result.exit_code == 0, result.output
    assert "judged safety predicates have a REVIEWED set" in result.output


def test_a_draft_judge_set_does_not_count_toward_the_ten():
    """Two drafts are committed. If they ever counted, the report would claim
    two validated judges on the strength of machine-proposed labels."""
    result = runner.invoke(app, ["judge"])
    assert "0/10" in result.output
    assert "DRAFT" in result.output


def test_strict_fails_when_a_conversation_has_no_expect_block(tmp_path):
    (tmp_path / "x.yaml").write_text(
        "id: x\nmode: 1\ntitle: t\nprovenance: handwritten\n"
        'context: {today: "2026-03-10"}\nturns: [{inbound: hi}]\n',
        encoding="utf-8",
    )
    assert runner.invoke(app, ["--dir", str(tmp_path)]).exit_code == 0
    assert runner.invoke(app, ["--strict", "--dir", str(tmp_path)]).exit_code == 1


def test_an_unreadable_conversation_fails_rather_than_being_skipped(tmp_path):
    (tmp_path / "broken.yaml").write_text("id: x\n  bad: [indent\n", encoding="utf-8")
    result = runner.invoke(app, ["--dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "invalid" in result.output
