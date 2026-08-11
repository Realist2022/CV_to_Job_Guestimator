import os
from pathlib import Path
from src.services import (
    JobListing,
    CandidateCV,
    InstructorClient,
    ExtractionPipeline,
    PDFTextExtractionError,
)
from src.utils import ArtifactLogger
from src.config import (
    MODEL_NAME,
    MODEL_BASE_URL,
    MODEL_API_KEY,
    PII_MODEL_NAME,
    PII_MODEL_BASE_URL,
    PII_MODEL_API_KEY,
)


def main():
    job_path = _first_existing_path(
        os.path.join("dataSet", "tradeMeJobListing", "Job_listing.txt"),
        os.path.join("dataSet", "tradeMeJobListing", "Job_listing.pdf"),
    )
    cv_path = _first_existing_path(
        os.path.join("dataSet", "tradeMeCV", "Sonny H Tapara CV.txt"),
        os.path.join("dataSet", "tradeMeCV", "Sonny H Tapara CV.pdf"),
    )

    if not job_path or not cv_path:
        print("Error: Target document files not found. Please verify your dataSet paths.")
        return

    print(
        f"PII Engine: {PII_MODEL_NAME} (local) | "
        f"Evaluation Engine: {MODEL_NAME} | Architecture: Multi-Agent Instructor Harness"
    )

    # The PII client sees the raw CV; the evaluation client receives only redacted CV text.
    client = InstructorClient(
        model=MODEL_NAME,
        base_url=MODEL_BASE_URL,
        api_key=MODEL_API_KEY,
    )
    pii_client = InstructorClient(
        model=PII_MODEL_NAME,
        base_url=PII_MODEL_BASE_URL,
        api_key=PII_MODEL_API_KEY,
    )
    pipeline = ExtractionPipeline(client, pii_client=pii_client)

    try:
        listing = _load_job_listing(job_path)
        cv = CandidateCV.from_path(cv_path)
    except (PDFTextExtractionError, ValueError) as exc:
        print(f"Error: {exc}")
        return

    # Execute pipeline
    pipeline_data = pipeline.run(listing, cv)

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
    print("=" * 66)

    # Log artifact trace
    logger = ArtifactLogger(output_dir="artifacts")
    saved_file = logger.log_run(pipeline_data)

    print(f"\nRun #{logger.last_run_number:06d} artifact saved to: {saved_file}\n")


def _first_existing_path(*paths: str) -> str | None:
    for path in paths:
        if Path(path).exists():
            return path
    return None


def _load_job_listing(path: str) -> JobListing:
    if Path(path).suffix.lower() == ".pdf":
        return JobListing.from_pdf(path, cache_text=True)
    return JobListing.from_path(path)


if __name__ == "__main__":
    main()