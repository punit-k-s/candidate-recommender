import os
from typing import Optional

from openai import OpenAI
from src.utils.tokens import count_tokens

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# soft caps so we don't blow context
MAX_JD_TOKENS = 1200
MAX_RESUME_TOKENS = 2200

def _truncate_to_tokens(text: str, max_tokens: int, model: str) -> str:
    if not text:
        return ""
    tks = count_tokens(text, model)
    if tks <= max_tokens:
        return text
    # crude but safe: cut proportionally by token ratio (char-based approximation)
    ratio = max_tokens / max(tks, 1)
    approx_chars = max(512, int(len(text) * ratio * 0.5) * 2)
    return text[:approx_chars]

def generate_fit_summary(jd_text: str, resume_text: str, role_title: Optional[str] = None) -> str:
    """
    Return a concise, evidence-based summary explaining why the candidate fits the JD.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    # We use the embedding model for token counting heuristic; the LLM can be different.
    jd_trim = _truncate_to_tokens(jd_text, MAX_JD_TOKENS, "text-embedding-3-small")
    resume_trim = _truncate_to_tokens(resume_text, MAX_RESUME_TOKENS, "text-embedding-3-small")

    title_bit = f' for the role "{role_title}"' if role_title else ""
    system = (
        "You are a concise, practical recruiter. Given a job description and a candidate resume, "
        "explain WHY the candidate fits. Cite specific evidence from the resume. Avoid fluff."
    )

    user = f"""Job Description{title_bit}:
```text
{jd_trim}
Resume:
```text
{resume_trim}
```
Write:

100-150 words total.

Start with a one-line verdict (e.g., "Strong fit for Power BI-heavy BI role with proven dashboard delivery and SQL expertise.").

Then 2-3 bullet points mapping JD requirements to resume evidence (skills, tools, outcomes/metrics).

"""

    # Call the LLM
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=350,
    )

    # Return the LLM's reply
    return resp.choices[0].message.content.strip()



    



