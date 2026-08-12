import sys

from src.harness import HarnessRunner
from src.services import PDFTextExtractionError

DEFAULT_TASK = "tasks/cv_job_match.yaml"


def main():
    task_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TASK

    runner = HarnessRunner()
    try:
        report = runner.run(task_path)
    except (FileNotFoundError, PDFTextExtractionError, ValueError) as exc:
        print(f"Error: {exc}")
        return

    pipeline_data = report.result
    print(
        f"Task: {report.task_name} | PII Engine: {pipeline_data.pii_engine} | "
        f"Evaluation Engine: {pipeline_data.engine} | Architecture: Multi-Agent Instructor Harness"
    )

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

    if report.evaluation.checks:
        print("HARNESS EVALUATION")
        print("-" * 66)
        for check in report.evaluation.checks:
            status = "PASS" if check.passed else "FAIL"
            print(f"  [{status}] {check.name}: expected {check.expected}, got {check.actual}")
        print(f"  Overall: {'PASS' if report.evaluation.passed else 'FAIL'}")

    print("=" * 66)
    print(f"\nRun #{report.run_number:06d} artifact saved to: {report.artifact_path}\n")


if __name__ == "__main__":
    main()