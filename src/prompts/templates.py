from typing import Final

from src.schemas.pii import PIIKind


JOB_REQUIREMENTS_SYSTEM_PROMPT: Final = """Extract atomic technical and operational capabilities from a job description.

Include specific domain tools, machinery, software, methodologies, frameworks, certifications, and technical capabilities.
Exclude generic soft skills such as hard working, communication, teamwork, and punctuality.
Return one capability per record. Split combined requirements such as "React and Node.js" into separate React and Node.js records.
The capability field must contain only the capability name, without years-of-experience wording.
Set minimum_commercial_years only when the description explicitly requires a duration of commercial experience for that capability.
When one duration governs multiple capabilities, copy its minimum to each affected capability.
Use the lower bound for ranges such as "2-5 years". Otherwise return null.
Return each unique capability once, grounded only in the job description."""


SKILL_MATCHER_SYSTEM_PROMPT: Final = """Evaluate whether a candidate's CV satisfies each supplied job requirement.

Return exactly one evaluation for every numeric requirement_id supplied.
Set matched to true when the CV explicitly shows the capability or a directly equivalent capability.
Set matched to false when the CV does not show sufficient evidence.
Evaluate capability only. Ignore minimum_commercial_years and do not require dated or commercial evidence here.
Do not extract, rename, summarize, or introduce skills in the evaluations.

Treat specific tools or certifications as evidence for a broader requirement only when they directly fulfill it."""


SKILL_TENURE_SYSTEM_PROMPT: Final = """Associate each supplied commercial-tenure capability with dated CV roles that explicitly demonstrate it.

Return exactly one record for every numeric requirement_id supplied.
Return role_ids using only the supplied numeric role IDs.
Link a role only when the CV explicitly demonstrates the requirement within that role's content.
Evaluate every requirement independently. The same role may support multiple requirements.
Treat direct role evidence such as "REST API endpoints" as support for the corresponding REST API requirement.
Do not link a role based only on an undated skills list, summary, job title, or assumed transferability.
Return an empty role_ids list when no dated role explicitly supports the requirement.
Keep evidence concise and quote or summarize the role-specific CV evidence.
Do not add, omit, rename, or merge requirements."""


OVERALL_EXPERIENCE_SYSTEM_PROMPT: Final = """Extract and classify the candidate's professional roles against a target job.

Extract the target job title and any explicit minimum overall experience from the job description; use 2.0 years only when no overall minimum is stated.
Extract every professional role supported by the CV with dates formatted as YYYY-MM or Present.
Mark is_relevant true only when the role provides directly transferable experience for the target job's responsibilities.
Explain each relevance decision briefly using job and CV evidence.
Do not infer roles, dates, employers, or experience that are not present in the supplied text."""


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

from typing import Final

from src.schemas.pii import PIIKind


JOB_REQUIREMENTS_SYSTEM_PROMPT: Final = """Extract atomic technical and operational capabilities from a job description.

Include specific domain tools, machinery, software, methodologies, frameworks, certifications, and technical capabilities.
Exclude generic soft skills such as hard working, communication, teamwork, and punctuality.
Return one capability per record. Split combined requirements such as "React and Node.js" into separate React and Node.js records.
The capability field must contain only the capability name, without years-of-experience wording.
Set minimum_commercial_years only when the description explicitly requires a duration of commercial experience for that capability.
When one duration governs multiple capabilities, copy its minimum to each affected capability.
Use the lower bound for ranges such as "2-5 years". Otherwise return null.
Return each unique capability once, grounded only in the job description."""


SKILL_MATCHER_SYSTEM_PROMPT: Final = """Evaluate whether a candidate's CV satisfies each supplied job requirement.

Return exactly one evaluation for every numeric requirement_id supplied.
Set matched to true when the CV explicitly shows the capability or a directly equivalent capability.
Set matched to false when the CV does not show sufficient evidence.
Evaluate capability only. Ignore minimum_commercial_years and do not require dated or commercial evidence here.
Do not extract, rename, summarize, or introduce skills in the evaluations.

Treat specific tools or certifications as evidence for a broader requirement only when they directly fulfill it."""


SKILL_TENURE_SYSTEM_PROMPT: Final = """Associate each supplied commercial-tenure capability with dated CV roles that explicitly demonstrate it.

Return exactly one record for every numeric requirement_id supplied.
Return role_ids using only the supplied numeric role IDs.
Link a role only when the CV explicitly demonstrates the requirement within that role's content.
Evaluate every requirement independently. The same role may support multiple requirements.
Treat direct role evidence such as "REST API endpoints" as support for the corresponding REST API requirement.
Do not link a role based only on an undated skills list, summary, job title, or assumed transferability.
Return an empty role_ids list when no dated role explicitly supports the requirement.
Keep evidence concise and quote or summarize the role-specific CV evidence.
Do not add, omit, rename, or merge requirements."""


OVERALL_EXPERIENCE_SYSTEM_PROMPT: Final = """Extract and classify the candidate's professional roles against a target job.

Extract the target job title and any explicit minimum overall experience from the job description; use 2.0 years only when no overall minimum is stated.
Extract every professional role supported by the CV with dates formatted as YYYY-MM or Present.
Mark is_relevant true only when the role provides directly transferable experience for the target job's responsibilities.
Explain each relevance decision briefly using job and CV evidence.
Do not infer roles, dates, employers, or experience that are not present in the supplied text."""


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

