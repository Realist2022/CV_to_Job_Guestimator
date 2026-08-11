from typing import Final

from src.schemas.pii import PIIKind


JOB_REQUIREMENTS_SYSTEM_PROMPT: Final = """Extract atomic technical and operational skill_names from a job description.

Include specific domain tools, machinery, software, methodologies, frameworks, certifications, and technical skill_names.
Exclude generic soft skills such as hard working, communication, teamwork, and punctuality.
Return one skill_name per record. Split all combined requirements into separate records.
The skill_name field must contain only the skill name, without years-of-experience wording.
Return each unique skill_name once, grounded only in the job description.

Expected JSON structure:
{
  "job_requirements": [
    {
      "skill_name": ""
    },
    {
      "skill_name": ""
    }
  ]
}
"""


SKILL_MATCHER_SYSTEM_PROMPT: Final = """Evaluate whether a candidate's CV satisfies each supplied job requirement.

Return exactly one evaluation for every numeric requirement_id supplied.
Set matched to true when the CV explicitly shows the skill_name or a directly equivalent skill_name.
Set matched to false when the CV does not show sufficient evidence.
Evaluate skill_name only. Do not require dated or commercial evidence here.
Do not extract, rename, summarize, or introduce skills in the evaluations.

Treat specific tools or certifications as evidence for a broader requirement only when they directly fulfill it."""


OVERALL_EXPERIENCE_SYSTEM_PROMPT: Final = """Extract and classify the candidate's professional work experience against a target job.

1. Extract target job information from the job description:
- Extract target_job_title exactly as written.
- Extract target_overall_years from explicit minimum overall experience only.
- If the job description states a single minimum such as "3+ years", use that number.
- If a range is given such as "2-5 years", use the lower bound.
- If no minimum overall experience is stated, set target_overall_years to null.

2. Extract candidate roles from the CV:
- Extract ONLY paid employment or professional contractor work experience.
- STRICTLY EXCLUDE educational degrees, academic courses, certifications, bootcamps, personal projects, and volunteer work.
- Include role_title exactly as written.
- Format start_date and end_date as YYYY-MM or "Present" when present in the CV.
- Use null only if the CV is missing the start_date or end_date.
- Do not infer missing dates, missing employers, roles not explicitly stated, or experience not supported by the text.

3. Classify relevance for each role:
- Set is_relevant true only when the role provides directly transferable experience to the target job's responsibilities.
- Provide a brief, evidence-based match_rationale referencing specific job responsibilities and specific CV experience.
- Do not use generic statements such as "software experience is relevant".
- Do not assume skills or responsibilities not present in the text.

4. Output rules:
- Return valid JSON objects containing exactly one overall_experience object.
- Do not add fields not listed in the schema.
- Do not include markdown or trailing commentary.
- Do not hallucinate or infer missing data.

Expected JSON structure:
{
  "overall_experience": {
    "target_job_title": "",
    "target_overall_years": null,
    "candidate_roles": [
      {
        "role_title": "",
        "start_date": null,
        "end_date": null,
        "match_rationale": "",
        "is_relevant": false
      }
    ]
  }
}
"""


_PII_KIND_VALUES = "\n".join(f"- {kind.value}" for kind in PIIKind)

PII_SYSTEM_PROMPT: Final = f"""Find every piece of personally identifying text in this CV.

Copy each snippet word for word, exactly as it appears in the CV text. Do not tidy, reformat, or alter it.

Return only actual PII spans. If text is not PII, omit it completely; do not classify it under a fallback category.
Every span.kind must use one of these exact enum values:
{_PII_KIND_VALUES}
Never invent or return any other kind, including job_title, employer, skill, education, or employment_date.
Return an empty spans list when the CV contains no PII.

STRICT MAPPING RULES:
- Use person_name ONLY for the candidate's personal name and referee names.
- Use street_address ONLY for specific residential or home street addresses.
- Use date_of_birth ONLY for actual birth dates. NEVER use this for graduation, course, or employment year ranges.
- Use nationality ONLY for citizenship or residency status.
- Use marital_or_family ONLY for marital or family status.
- Use other_identifier ONLY for explicit contact or ID fields (email, phone, mobile, LinkedIn/GitHub URLs, driver's licences, IRD numbers).

CRITICAL EXCLUSIONS (DO NOT EXTRACT):
- NEVER extract Employer / Company Names (e.g., "FOODSTUFFS July 2025-October 2025" -> DO NOT EXTRACT (Employer name and employment period), "CRANE AND CARTAGE" -> DO NOT EXTRACT (Employer name), "MISSION READY DIPLOMA" -> DO NOT EXTRACT (Educational provider/qualification), "District Health Board" -> DO NOT EXTRACT (Employer name)).
- NEVER extract Job Titles or Roles (e.g., Software Engineer, Hiab Operator, Technician, Manager).
- NEVER extract Educational Institutions, Diplomas, or Certificates (e.g., Mission Ready, NZQA, University).
- NEVER extract Employment dates or Project names.
- NEVER extract Employer/Company Names, Job Titles, Educational Institutions, Diplomas, or Employment Date Ranges.
- If a piece of text is a job title, company, skill, qualification, work achievement, or other CV content, IT IS NOT PII. Omit it from spans."""

