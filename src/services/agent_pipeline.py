from openai import OpenAI

# Import your settings and schemas from the other files we created
from src.config import MODEL_BASE_URL, MODEL_API_KEY, MODEL_NAME, MODEL_TEMPERATURE
from src.schemas.agent_outputs import JobRequirementsOutput, CVSkillMatchOutput, OverallExperienceOutput
from src.prompts.templates import JOB_PARSER_PROMPT, CV_SKILL_MATCHER_PROMPT, CV_EXPERIENCE_PROMPT

class MultiAgentPipeline:
    def __init__(self):
        # Initializes the client using your Ollama local settings from config.py
        self.client = OpenAI(base_url=MODEL_BASE_URL, api_key=MODEL_API_KEY)

    def _call_agent(self, system_prompt: str, user_content: str, response_model):
        """Helper method to make structured calls to the local SLM."""
        response = self.client.beta.chat.completions.parse(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=MODEL_TEMPERATURE,
            response_format=response_model,
        )
        return response.choices[0].message.parsed

    def run_pipeline(self, job_text: str, cv_text: str) -> dict:
        """Runs the 3-step sequential agent extraction pipeline."""
        
        # AGENT 1: Extract Job Requirements
        print(" -> [Agent 1] Parsing Job Listing for explicit requirements...")
        job_data: JobRequirementsOutput = self._call_agent(
            JOB_PARSER_PROMPT, 
            f"Job Description:\n{job_text}", 
            JobRequirementsOutput
        )

        # Build clean target list for Agent 2
        target_skills_list = [req.skill_name for req in job_data.required_skills]
        print(f"    Found {len(target_skills_list)} target requirements: {', '.join(target_skills_list)}")

        # AGENT 2: Audit CV against isolated skill list
        print(" -> [Agent 2] Auditing CV against extracted job requirements...")
        matcher_input = f"Target Required Skills List:\n{target_skills_list}\n\nCandidate CV:\n{cv_text}"
        cv_skills_data: CVSkillMatchOutput = self._call_agent(
            CV_SKILL_MATCHER_PROMPT, 
            matcher_input, 
            CVSkillMatchOutput
        )

        # AGENT 3: Extract Career Roles from CV
        print(" -> [Agent 3] Extracting career employment history...")
        experience_data: OverallExperienceOutput = self._call_agent(
            CV_EXPERIENCE_PROMPT, 
            f"Job Text:\n{job_text}\n\nCV Text:\n{cv_text}", 
            OverallExperienceOutput
        )

        return {
            "job_requirements": job_data,
            "matched_skills": cv_skills_data,
            "overall_experience": experience_data
        }