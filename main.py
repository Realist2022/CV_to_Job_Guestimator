import sys

from src.harness import HarnessRunner
from src.services import PDFTextExtractionError

DEFAULT_TASK = "tasks/cv_job_match.yaml"


def _use_utf8_output() -> None:
    """Make stdout/stderr able to carry this report's ✓/✗ and box characters.

    Windows consoles and redirected pipes default to cp1252, which raises
    UnicodeEncodeError on the first check mark. The run's artifact is already
    safely on disk by then (HarnessRunner.run logs it before returning), so
    nothing is lost but the human-readable summary — and the traceback that
    replaces it reads like a pipeline failure when it is only a printing one.
    Reconfigure to UTF-8; fall back to replacing unencodable characters so
    that a terminal which cannot do UTF-8 degrades to "?" instead of dying.
    """
    for stream in (sys.stdout, sys.stderr):
        # getattr rather than a bare call: sys.stdout is only a TextIOWrapper
        # by convention, and something capturing output (a test harness, a
        # notebook) can substitute a plain TextIO with no reconfigure at all.
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            # Stream can't be re-encoded; worst case is the original crash.
            continue


def main():
    _use_utf8_output()
    task_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TASK

    runner = HarnessRunner()
    try:
        report = runner.run(task_path)
    except (FileNotFoundError, PDFTextExtractionError, ValueError) as exc:
        print(f"Error: {exc}")
        return

    pipeline_data = report.result
    # An "ingestion" task returns an IngestionResult: PII redaction only, no
    # evaluation model and no scorecard/skills/experience to report. The
    # scored sections are guarded on that rather than assumed, because
    # reading .engine off an IngestionResult raised AttributeError and took
    # the whole summary down *after* the run had already been logged --
    # `uv run main.py tasks/cv_ingest.yaml` printed a traceback instead of
    # its result, despite the run itself having succeeded.
    scored = hasattr(pipeline_data, "scorecard")
    engine_note = f"Evaluation Engine: {pipeline_data.engine} | " if scored else ""
    print(
        f"Task: {report.task_name} | PII Engine: {pipeline_data.pii_engine} | "
        f"{engine_note}Architecture: Multi-Agent Instructor Harness"
    )

    if scored:
        _print_scorecard(pipeline_data)
    else:
        print("\n" + "=" * 66)
        print("INGESTION OUTPUT")
        print("-" * 66)
        print(f"cv_id:              {pipeline_data.cv_id}")
        print(f"PII spans redacted: {len(pipeline_data.pii_spans)}")

    _print_trace_and_evaluation(pipeline_data, report)


def _print_scorecard(pipeline_data) -> None:
    skills_eval = pipeline_data.skills_eval
    overall_experience = pipeline_data.overall_experience
    scorecard = pipeline_data.scorecard

    # Display results
    print("\n" + "=" * 66)
    print("SCORING ENGINE OUTPUT")
    print("-" * 66)
    print(f"Overall Match: {scorecard.final_relevance}%")
    print(f"Skills Match:  {scorecard.pillar_a.score}% ({scorecard.pillar_a.raw})")
    career_score = (
        f"{scorecard.pillar_b.score}%" if scorecard.pillar_b.applicable else "N/A"
    )
    print(f"Career Match:  {career_score} ({scorecard.pillar_b.raw})")

    print("\nAGENT 1 OUTPUT: PII DETECTOR")
    print("-" * 66)
    print(f"PII spans redacted: {len(pipeline_data.pii_spans)}")

    print("\nAGENT 2 OUTPUT: JOB REQUIREMENTS")
    print("-" * 66)
    print(f"Requirements extracted: {skills_eval.total_job_requirements}")
    for requirement in skills_eval.job_requirements:
        print(f"  - {requirement.skill_name}")

    print("\nAGENT 3 OUTPUT: SKILL MATCHER")
    print("-" * 66)
    print(f"Category:     {skills_eval.requirement_category}")
    print(f"Match Score:  {skills_eval.match_percentage}% ({skills_eval.total_matched_skills}/{skills_eval.total_job_requirements})")
    print(f"Rationale:    {skills_eval.rationale}")
    print(f"Matched Skills ({len(skills_eval.matched_cv_skills)}):")
    for s in skills_eval.matched_cv_skills:
        print(f"  ✓ {s}")
    print(f"\nMissing Skills ({len(skills_eval.missing_cv_skills)}):")
    for s in skills_eval.missing_cv_skills:
        print(f"  ✗ {s}")

    print("\nAGENT 4 OUTPUT: OVERALL EXPERIENCE")
    print("-" * 66)
    print(f"Target role:       {overall_experience.target_job_title}")
    print(f"Required experience: {overall_experience.target_overall_years} years")
    for role in overall_experience.candidate_roles:
        relevance = "Relevant" if role.is_relevant else "Not relevant"
        print(f"  Role:      {role.role_title} ({role.start_date} to {role.end_date})")
        print(f"  Decision:  {relevance}")
        print(f"  Rationale: {role.match_rationale}")
        print()


def _print_trace_and_evaluation(pipeline_data, report) -> None:
    if pipeline_data.trace:
        print("\nPIPELINE TRACE")
        print("-" * 66)
        for span in pipeline_data.trace:
            retry_note = (
                f"  ({span.attempts} attempts)" if span.attempts and span.attempts > 1 else ""
            )
            print(f"  {span.step:<30} {span.duration_seconds:>6.2f}s{retry_note}")
        step_time_sum = round(sum(span.duration_seconds for span in pipeline_data.trace), 2)
        print(f"  {'-' * 38}")
        print(f"  {'Sum of step time':<30} {step_time_sum:>6.2f}s")
        print(f"  {'Actual wall time':<30} {pipeline_data.execution_seconds:>6.2f}s")

    if report.evaluation.checks:
        print("\nHARNESS EVALUATION")
        print("-" * 66)
        for check in report.evaluation.checks:
            status = "PASS" if check.passed else "FAIL"
            print(f"  [{status}] {check.name}: expected {check.expected}, got {check.actual}")
        print(f"  Overall: {'PASS' if report.evaluation.passed else 'FAIL'}")

    print("=" * 66)
    print(f"\nTrace ID: {pipeline_data.trace_id}")
    print(f"Run #{report.run_number:06d} artifact saved to: {report.artifact_path}\n")


if __name__ == "__main__":
    main()