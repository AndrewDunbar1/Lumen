import os
import asyncio
from typing import List
from openai import AsyncOpenAI
from dotenv import load_dotenv
# Load your key (or set directly)
load_dotenv()  # or api_key="your-key"
MODEL_NAME = "gpt-5-mini"


def _get_client() -> AsyncOpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    return AsyncOpenAI(api_key=api_key) if api_key else None


def _chat_kwargs_for_model(model_name: str) -> dict:
    return {
        "model": model_name,
    }

SYSTEM_PROMPT = """
You are a medical assistant trained to normalize patient condition names into general disease labels
that match the terminology used on ClinicalTrials.gov.

Your task is to take a specific clinical condition (often ICD-like or descriptive) and return a
single, more general condition label suitable for clinical trial searches.

Instructions:
- Respond with only the generalized condition.
- Do not explain your answer.
- If the term is already general (e.g., "colon cancer"), return it as-is.
- If the term is procedural or not a disease (e.g., "accidental laceration during surgery"), respond with "Not applicable".

Examples:

Input: "Secondary malignant neoplasm of bone"  
Output: Metastatic cancer

Input: "Iron deficiency anemia secondary to blood loss"  
Output: Anemia

Input: "Malignant neoplasm of corpus uteri, except isthmus"  
Output: Endometrial cancer

Input: "Body mass index 19.9 or less, adult"  
Output: Underweight

Input: "Personal history of irradiation"  
Output: Cancer treatment history

Input: "Laparoscopic surgical procedure converted to open procedure"  
Output: Not applicable
"""

async def generalize_condition(condition: str) -> str:
    """Send one condition to OpenAI and get a generalized term."""
    client = _get_client()
    if client is None:
        # Fallback when OPENAI_API_KEY is unavailable.
        return condition

    response = await client.chat.completions.create(
        **_chat_kwargs_for_model(MODEL_NAME),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'Input: "{condition}"\nOutput:'}
        ],
    )
    return response.choices[0].message.content.strip()

async def generalize_conditions_async(conditions: List[str]) -> List[str]:
    """Asynchronously generalizes a list of conditions in parallel."""
    tasks = [generalize_condition(c) for c in conditions]
    return await asyncio.gather(*tasks)

def generalize_conditions(conditions: List[str]) -> List[str]:
    """Sync wrapper around the async function for ease of use."""
    return asyncio.run(generalize_conditions_async(conditions))


# Optional test
if __name__ == "__main__":
    test_conditions = [
        "Secondary malignant neoplasm of bone",
        "Iron deficiency anemia secondary to blood loss",
        "Body mass index 19.9 or less, adult",
        "Malignant neoplasm of ascending colon",
        "Abdominal pain, other specified site",
        "Laparoscopic surgical procedure converted to open procedure",
        "Diarrhea, unspecified",
        "Personal history of antineoplastic chemotherapy",
        "Alcohol abuse, unspecified"
    ]

    results = generalize_conditions(test_conditions)
    for original, general in zip(test_conditions, results):
        print(f"{original} → {general}")
