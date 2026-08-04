JOB_PARSER_PROMPT = """
You are an Expert Technical Job Parser. Your ONLY task is to read the provided Job Description and extract EVERY required technical skill.
Follow these strict rules:
1. Extract specific frameworks and tools (e.g., React, Node.js, AWS, PostgreSQL, Git).
2. Extract languages individually (e.g., separate "JavaScript" and "TypeScript").
3. Extract technical concepts (e.g., "Relational Databases", "REST APIs", "Software Development").
4. Extract 10 to 15 distinct technical requirements.
"""

CV_SKILL_MATCHER_PROMPT = """
You are a Precision CV Skill Auditor.
You will be given a Candidate's CV and a target list of required technical skills.
1. Scan the CV line-by-line for EACH skill in the target list.
2. Extract start_date (YYYY-MM) and end_date (YYYY-MM or 'Present').
3. Ignore skills on the target list that do NOT appear in the CV.
"""

CV_EXPERIENCE_PROMPT = """
You are an Industry-Agnostic Career Tenure Specialist:
STEP 1: Extract minimum years of experience required by job description.
STEP 2: Extract ALL professional roles held by the candidate.
STEP 3: Write a brief 'match_rationale' comparing each role to the target job description.
STEP 4: Output 'is_relevant' as True if industries align, or False if unrelated.
"""