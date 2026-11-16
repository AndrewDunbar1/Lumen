"""Shape per-patient trial evaluations into the JSON written by the pipeline runner."""


def prepare_frontend_response(evaluations, patient_id="default"):
    """Summarize one patient's evaluated trials.

    `evaluations` is the list of trial evaluation dicts, already sorted by the
    pipeline. Returns the eligible and ineligible trials plus a ranking trace.
    """
    frontend_data = {
        "summary": {
            "total_matching_trials": len([e for e in evaluations if e["eligible"]]),
            "total_evaluated": len(evaluations),
            "patient_id": patient_id,
        }
    }
    frontend_data["matching_trials"] = [e for e in evaluations if e["eligible"]]
    frontend_data["non_matching_trials"] = [e for e in evaluations if not e["eligible"]]
    frontend_data["ranking_trace"] = [
        {
            "trial_nct_id": e.get("nct_id"),
            "title": e.get("title"),
            "retrieval_rank": e.get("retrieval_rank"),
            "rerank_rank": e.get("rerank_rank"),
            "match_recommendation": e.get("match_recommendation"),
            "eligibility_probability": e.get("eligibility_probability"),
            "system_rationale_short": e.get("system_rationale_short"),
        }
        for e in evaluations
    ]
    return frontend_data
