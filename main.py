# main.py
import os
import time
from src.services.document_parser import DocumentParser
from src.services.agent_pipeline import MultiAgentPipeline
from src.services.scoring_engine import RelevanceScoringEngine
from src.utils import ArtifactLogger
from src.config import MODEL_NAME, SKILLS_MATCH_WEIGHT, SKILL_TENURE_WEIGHT, WORK_EXPERIENCE_WEIGHT

def main():
    start_time = time.time()
    
    job_pdf = os.path.join("dataSet", "tradeMeJobListing", "Job_listing.pdf")
    cv_pdf = os.path.join("dataSet", "tradeMeCV", "Sonny H Tapara CV.pdf")

    if not os.path.exists(job_pdf) or not os.path.exists(cv_pdf):
        print("Error: Target PDF files not found.")
        return

    job_desc = DocumentParser.extract_text_from_pdf(job_pdf)
    cv_text = DocumentParser.extract_text_from_pdf(cv_pdf)

    print(f"Engine: {MODEL_NAME} | Architecture: Multi-Agent Harness")
    
    # Run Multi-Agent Extraction
    pipeline = MultiAgentPipeline()
    pipeline_data = pipeline.run_pipeline(job_desc, cv_text)

    # Calculate Scorecard
    scoring_engine = RelevanceScoringEngine()
    report = scoring_engine.calculate_scorecard(pipeline_data)

    execution_duration = time.time() - start_time

    # Log JSON Artifact Trace
    logger = ArtifactLogger(output_dir="artifacts")
    saved_file = logger.log_run(
        engine=MODEL_NAME,
        execution_time_seconds=execution_duration,
        pipeline_data=pipeline_data,
        report=report
    )

    print("\n" + "=" * 60)
    print("MULTI-AGENT COMPUTED RELEVANCE REPORT")
    print("=" * 60)
    print(f"Overall Match Score: {report['final_relevance']}%")
    print("-" * 60)
    print(f"• Pillar A (Skills Match)   [{int(SKILLS_MATCH_WEIGHT*100)}%]: {report['pillar_a']['raw']} ({report['pillar_a']['score']}%)")
    print(f"• Pillar B (Skill Tenure)   [{int(SKILL_TENURE_WEIGHT*100)}%]: {report['pillar_b']['raw']} ({report['pillar_b']['score']}%)")
    print(f"• Pillar C (Overall Tenure) [{int(WORK_EXPERIENCE_WEIGHT*100)}%]: {report['pillar_c']['raw']} ({report['pillar_c']['score']}%)")
    print("-" * 60)
    print(f"-> Validated Skills: {', '.join(report['validated_skills'])}")
    print(f"-> Relevant Roles: {', '.join(report['counted_roles'])}")
    print("=" * 60)
    print(f"[ARTIFACT SAVED]: {saved_file}")
    print(f"Execution time: {round(execution_duration, 2)} seconds")

if __name__ == "__main__":
    main()