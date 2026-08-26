from typing import Final

# Bump the version alongside its prompt whenever the wording changes, so an
# artifact's config.prompt_versions records exactly which prompt text
# produced it — independent of which model executed it (see RunConfig in
# src/schemas/artifact.py). Leave a version untouched if its prompt didn't
# change.
JOB_REQUIREMENTS_PROMPT_VERSION: Final = "1.3"
SKILL_MATCHER_PROMPT_VERSION: Final = "1.2"
OVERALL_EXPERIENCE_PROMPT_VERSION: Final = "1.2"

# Convenience grouping for RunConfig.prompt_versions, matching which prompts
# MatchingPipeline actually runs (see agents.py). PII redaction (presidio —
# see src/services/presidio_detector.py) has no LLM prompt at all, so
# ExtractionPipeline (= ingestion + matching) reports the same three
# prompts as MatchingPipeline; IngestionPipeline reports an empty
# prompt_versions dict.
MATCHING_PROMPT_VERSIONS: Final = {
    "job_requirements": JOB_REQUIREMENTS_PROMPT_VERSION,
    "skill_matcher": SKILL_MATCHER_PROMPT_VERSION,
    "overall_experience": OVERALL_EXPERIENCE_PROMPT_VERSION,
}
EXTRACTION_PROMPT_VERSIONS: Final = MATCHING_PROMPT_VERSIONS

JOB_REQUIREMENTS_SYSTEM_PROMPT: Final = """Extract atomic technical and operational skill_names from a job description.

Include specific domain tools, machinery, software, methodologies, frameworks, certifications, and technical skill_names.
Exclude generic soft skills such as hard working, communication, teamwork, and punctuality.
Exclude vague qualitative descriptors that name no specific technology, tool, or named
methodology, such as "modern development practices", "modern engineering practices",
"modern web applications", or "fast-paced environment" — these are not independently
verifiable against a CV. Keep concrete, named methodologies and practices such as
"Agile", "CI/CD", "code reviews", or "test-driven development".
Return one skill_name per record. Split all combined requirements into separate records.
The skill_name field must contain only the skill name, without years-of-experience wording.
Return each unique skill_name once, grounded only in the job description.

When a requirement names a general category followed by a specific example marked as
optional or preferred (in parentheses, or after "e.g."/"such as"/"including"), extract
only the general category as the skill_name and drop the example and its qualifier.
Example: "Exposure to cloud technologies (AWS preferred)" -> skill_name: "Cloud technologies".
Do not create a second record for the example in this case.
Only extract the specific named technology on its own when the text requires that
technology directly, not merely as an example of a broader category.

When a requirement names a general category and then states the exact required
instantiation of that category in the same sentence (e.g. via "specifically
requiring", "specifically", or by naming the only acceptable tool/method), treat
the general category and the specific instantiation as ONE requirement, not two.
Extract only the specific instantiation as the skill_name, and do not create a
separate record for the general category that wraps it.
Example: "A strong foundation in digital literacy, specifically requiring 2 years
of experience using Google Workspace for Education" -> skill_name: "Google
Workspace for Education". Do not also create a record for "digital literacy".

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

Treat specific tools or certifications as evidence for a broader requirement only when they directly fulfill it.

Apply category-inclusion reasoning: when a requirement names a general category or
practice, a specific CV item that is a well-known member of that category counts as
a match, even if the CV never uses the requirement's exact wording.
- "Relational databases" is satisfied by any named relational database the CV lists
  (e.g. MySQL, PostgreSQL, SQLite, Oracle, SQL Server), not only PostgreSQL itself.
- "Cloud technologies" is satisfied by any named cloud provider or service the CV
  lists (e.g. AWS, Azure, GCP), even under a different section heading such as
  "DevOps & Cloud".
- A practice-based requirement such as "AI-assisted software development" is
  satisfied by concrete CV evidence of that practice — a project description,
  tool, or self-description naming AI/LLM/chatbot work — not only by the literal
  phrase appearing in the CV.
- A professional registration or licensing requirement (e.g. "New Zealand
  Practising Certificate") is satisfied by CV evidence of full registration or
  licensure with the relevant regulatory body, even when worded differently
  (e.g. "Fully Registered Teacher (NZTC)").
- A jurisdiction- or system-specific experience requirement (e.g. "teaching
  experience in a New Zealand secondary school") is satisfied by CV evidence
  that uses that jurisdiction's characteristic terminology, curriculum, or role
  titles (e.g. NCEA levels, a Wellington-based school, Dean/Form Teacher roles),
  even without the literal phrase appearing.

These are illustrative examples from a few domains, not an exhaustive list — apply
the same category-inclusion and equivalent-terminology reasoning in whatever
professional domain the CV and requirement belong to (medicine, trades, education,
finance, etc.), not only software or technology.

Ground every match in text that actually appears in the CV. Do not infer a category
match from a requirement alone, and do not invent CV content that is not present."""


OVERALL_EXPERIENCE_SYSTEM_PROMPT: Final = """Extract and classify the candidate's professional work experience against a target job.

1. Extract target job information from the job description:
- Extract target_job_title exactly as written.
- Extract target_overall_years from explicit minimum overall experience only.
- If the job description states a single minimum such as "3+ years", use that number.
- If a range is given such as "2-5 years", use the lower bound.
- Treat a years figure as the overall requirement even when it is phrased alongside
  the role's core/primary technologies (e.g. "2-5 years' commercial experience with
  React and Node.js" -> target_overall_years: 2), since that is the role's experience bar.
- Only leave target_overall_years null when years are tied to a narrow, secondary
  tool/certification unrelated to the role's main responsibilities, or no years figure
  appears anywhere in the listing.

2. Extract candidate roles from the CV:
- Extract ONLY paid employment or professional contractor work experience.
- STRICTLY EXCLUDE educational degrees, academic courses, certifications, bootcamps, personal projects, and volunteer work.
- Include role_title exactly as written.
- Format start_date and end_date as YYYY-MM or "Present" when present in the CV.
- Use null only if the CV is missing the start_date or end_date.
- Do not infer missing dates, missing employers, roles not explicitly stated, or experience not supported by the text.

3. Classify relevance for each role:
- Set is_relevant true only when the role provides directly transferable experience to the target job's responsibilities.
- Base is_relevant on the role's full set of listed duties/bullets, not just one representative bullet — a role is relevant if ANY of its responsibilities directly matches a target job responsibility, even if other bullets in the same role do not.
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

