"""
Enhanced post-processing system that combines:
1. Auto-rejection for critical criteria
2. Weighted scoring for quality assessment
3. Normalized scoring to account for trial complexity
4. Confidence-based contribution to overall match quality
5. PRIORITIZATION of trials for rare and serious conditions using authoritative databases
"""

import json
import math
import re
from collections import Counter
import logging
import sys
import os
from pathlib import Path

# Add parent directory to path to access modules properly
sys.path.append(str(Path(__file__).parent.parent))

# rare disease functions are not fully built out yet, only have sample data 
# so are commented out for now
# We need to import these directly from the module path
# rare_conditions_db_path = os.path.join(Path(__file__).parent.parent, "post_processing", "rare_conditions_db.py")
# rare_conditions_module = {
#     '__file__': rare_conditions_db_path  # Add __file__ to the module globals
# }
# with open(rare_conditions_db_path) as f:
#     exec(f.read(), rare_conditions_module)


# get_canonical_condition_name = rare_conditions_module["get_canonical_condition_name"]
# get_condition_data = rare_conditions_module["get_condition_data"]
# calculate_severity_modifier = rare_conditions_module["calculate_severity_modifier"]
# is_rare_condition = rare_conditions_module["is_rare_condition"]
# is_severe_condition = rare_conditions_module["is_severe_condition"]
# calculate_patient_condition_priority = rare_conditions_module["calculate_patient_condition_priority"]



# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatchScorer:
    """
    Post-processes and scores trial-patient matches from LLM evaluation.
    
    This handles Step 5 of the trial-to-patient matching pipeline.
    """
    
    def __init__(self, trial_summary):
        """
        Initialize with trial summary.
        
        Args:
            trial_summary (dict): Trial summary from TrialParser.get_trial_summary()
        """
        self.trial_summary = trial_summary
        self.nct_id = trial_summary.get("nct_id")
        self.title = trial_summary.get("title")
        
    def extract_patient_conditions(self, patient):
        """
        Extract condition names from patient data.
        
        Args:
            patient: Patient data object
            
        Returns:
            list: List of patient's condition names
        """
        try:
            if patient and hasattr(patient, 'conditions') and patient.conditions and hasattr(patient.conditions, 'get_active_conditions'):
                conditions = patient.conditions.get_general_conditions()
                # Filter out None values
                return [c for c in conditions if c is not None]
            return []
        except (AttributeError, TypeError) as e:
            logger.warning(f"Error extracting patient conditions: {e}")
            return []
    
    def evaluate_match(self, llm_result, patient=None):
        """
        Evaluate trial-patient match using an enhanced scoring system for accurate matching.
        Handles OR logic groups for inclusion criteria during eligibility check.

        Args:
            llm_result (dict): Dictionary containing LLM evaluation results
            patient (PatientData, optional): Patient data object for condition cross-referencing

        Returns:
            dict: Evaluation results with normalized score, eligibility status, and detailed metrics
        """
        
        # Initialize metrics
        rejected = False
        rejection_reasons = []
        
        # Score components - separate from rejection logic
        raw_score = 0
        max_possible_score = 0
        
        # Track criteria counts and status together - consolidated
        criteria_stats = {
            'inclusion': {'total': 0, 'met': 0, 'not met': 0, 'unsure': 0},
            'exclusion': {'total': 0, 'met': 0, 'not met': 0, 'unsure': 0}
        }
        criteria_list = llm_result.get('results', [])


        # First pass - check rejections, track OR group status
        for criterion in criteria_list:
            # Skip malformed criteria
            try:

                criterion_type = criterion['type']
                status = criterion['status']
                confidence = criterion['confidence']

                # Track both count and status in consolidated structure
                if criterion_type in criteria_stats:
                    criteria_stats[criterion_type]['total'] += 1
                    if status in criteria_stats[criterion_type]:
                        criteria_stats[criterion_type][status] += 1

                is_high_confidence = confidence > 0.8

                # Standard rejection logic for non-OR group inclusion criteria
                if criterion_type == 'inclusion' and status == 'not met' and is_high_confidence:
                    rejected = True
                    rejection_reasons.append(f"Failed mandatory inclusion criterion: {criterion['criterion']}")

                # Standard rejection logic for exclusion criteria
                elif criterion_type == 'exclusion' and status == 'met' and is_high_confidence:
                    rejected = True
                    rejection_reasons.append(f"Met exclusion criterion: {criterion['criterion']}")

            except (KeyError, TypeError) as e:
                logger.warning(f"Malformed criterion during rejection check in trial {self.nct_id}: {criterion} - Error: {str(e)}")
                continue


        # Second pass - calculate scoring (only if not auto-rejected)
        if not rejected:
            for criterion in criteria_list:
                try:
                    criterion_type = criterion['type']
                    status = criterion['status']
                    confidence = criterion['confidence']
                    
                    # Calculate criterion importance (inclusion criteria weighted higher)
                    if criterion_type == 'inclusion':
                        importance = 1.0
                        # For inclusion, met status is positive
                        if status == 'met':
                            criterion_score = 2.0 * confidence  # Scale by confidence
                            max_possible = 2.0  # Maximum possible for this criterion
                        else:
                            criterion_score = 0 #right now this is 0 
                            max_possible = 2.0
                    else:  # exclusion
                        importance = 0.8
                        # For exclusion, not met status is positive
                        if status == 'not met':
                            criterion_score = 1.0 * confidence
                            max_possible = 1.0
                        else:
                            criterion_score = 0
                            max_possible = 1.0
                    
                    # Apply importance weighting
                    weighted_score = criterion_score * importance
                    weighted_max = max_possible * importance
                    # Add to totals
                    raw_score += weighted_score
                    max_possible_score += weighted_max
                    
                except (KeyError, TypeError):
                    # Skip malformed criteria in scoring
                    continue
        
        # Calculate normalized score (0-100 scale)
        normalized_score = 0
        # Ensure max_possible_score is not zero and patient wasn't rejected
        if max_possible_score > 0 and not rejected:
            # Using sigmoid normalization to better distinguish quality differences
            # This creates a 0-100 scale with better distribution in the middle range
            raw_ratio = raw_score / max_possible_score
            # Bound the ratio to prevent extreme values in exp, e.g., [-1, 1] or similar safe range
            # A simple clip might be sufficient if raw_ratio is expected near [0, 1]
            clipped_ratio = max(0.0, min(1.0, raw_ratio))
            # Apply sigmoid: 1 / (1 + exp(-k * (x - midpoint)))
            # Using k=10, midpoint=0.5 as before
            normalized_score = int(100 * (1 / (1 + math.exp(-10 * (clipped_ratio - 0.5)))))
        elif rejected:
             normalized_score = 0 # Explicitly set score to 0 if rejected
        
        # Add embedding similarity scores from LLM result if available
        
        # Prepare evaluation results
        evaluation = {
            'nct_id': self.nct_id,
            'title': self.title,
            'score': normalized_score,
            'raw_score': round(raw_score, 2),
            'max_possible': round(max_possible_score, 2),
            'eligible': not rejected,
            'rejection_reasons': rejection_reasons if rejected else [],
            'metrics': {
                'criteria_stats': criteria_stats,
            },
            'raw_criteria_results': llm_result.get('results', [])
        }
        
        # Add distance information if available
        # right now these are not doing anything. 
        # need to be fixed. 
        if "distance" in llm_result:
            evaluation["distance"] = llm_result["distance"]
        if "closest_facility" in llm_result:
            evaluation["closest_facility"] = llm_result["closest_facility"]
        
        return evaluation
    
    def process_llm_results(self, llm_results, patient_lookup=None):
        """
        Process LLM evaluation results for all patients and return sorted evaluations.
        
        Args:
            llm_results (list): List of LLM evaluation result dictionaries
            patient_lookup (dict, optional): Dictionary mapping patient_id to PatientData
            
        Returns:
            list: Intelligently sorted list of trial-patient evaluations
        """
        print(f"Post-processing and scoring {len(llm_results)} patient matches for trial {self.nct_id}...")
        evaluations = []
        
        for result in llm_results:
            patient_id = result.get('patient_id')
            patient = patient_lookup.get(patient_id) if patient_lookup else None
            
            # Process each evaluation with patient data if available
            eval_result = self.evaluate_match(result, patient)
            evaluations.append(eval_result)
        
        # Handle no evaluations case
        if not evaluations:
            return []
        
        # Primary sort: eligible patients first
        # Secondary sort: normalized score (highest first)
        # Tertiary sort: distance (lowest first, if available)
        # Quaternary sort: priority-weighted raw score / max_possible (highest ratio first)
        evaluations.sort(key=lambda x: (
            not x['eligible'],  # True (eligible) < False (not eligible)
            -x['score'],
            # Handle distance or None - we use float('inf') for missing distance to sort them last
            float('inf') if ('distance' not in x or x['distance'] is None) else x['distance'],
            -(x['raw_score'] / (x['max_possible'] if x['max_possible'] > 0 else 1))
        ))
        
        return evaluations
    ## this is for one trial to many patients function 
    ## did not look at this function yet 
    def prepare_frontend_response(self, evaluations):
        """
        Prepare a structured frontend response with match information.
        
        Args:
            evaluations (list): List of evaluation results
            
        Returns:
            dict: Structured data for frontend display
        """
        frontend_data = {
            "trial": {
                "nct_id": self.nct_id,
                "title": self.title
            },
            "summary": {
                "total_matching_patients": len([e for e in evaluations if e['eligible']]),
                "total_evaluated": len(evaluations)
            }
        }
        
        # Prepare matching and non-matching patients lists
        matching_patients = []
        non_matching_patients = []
        eligible_patient_count = 0
        
        for eval in evaluations:
            # Build enhanced patient data
            patient_data = {
                "patient_id": eval['patient_id'],
                "score": eval['score'],

                "eligible": eval['eligible'],
                "rejection_reasons": eval['rejection_reasons'],
                "retrieval_rank": eval.get('retrieval_rank'),
                "rerank_rank": eval.get('rerank_rank'),
                "eligibility_probability": eval.get('eligibility_probability'),
                "clinical_fit_score": eval.get('clinical_fit_score'),
                "evidence_completeness_score": eval.get('evidence_completeness_score'),
                "match_recommendation": eval.get('match_recommendation'),
                "age_sex_gate_status": eval.get('age_sex_gate_status'),
                "hard_blockers": eval.get('hard_blockers', []),
                "open_questions": eval.get('open_questions', []),
                "system_rationale_short": eval.get('system_rationale_short', ''),
                "criteria_summary": eval['metrics']['criteria_stats'],
                "condition_analysis": {
                    "priority_conditions": [],  # No longer populated
                    "has_rare_condition": any(c.get('is_rare', False) for c in eval['condition_analysis']['conditions']),
                    "has_severe_condition": any(c.get('is_severe', False) for c in eval['condition_analysis']['conditions']),
                    "has_patient_match": any(c.get('is_patient_condition', False) for c in eval['condition_analysis']['conditions'])
                }
            }
            
            # Add distance information if available
            if "distance" in eval and eval["distance"] is not None:
                patient_data["distance"] = round(eval["distance"], 1)
            else:
                patient_data["distance"] = None
                
            # Add facility information if available
            if "closest_facility" in eval:
                facility = eval["closest_facility"]
                print("FACILITY", facility)
                if facility and len(facility) >= 7:
                    patient_data["closest_facility"] = {
                        "name": facility.get("name"),
                        "city": facility.get("city"),
                        "state": facility.get("state"),
                        "zip": facility.get("zip"),
                        "country": facility.get("country"),
                        "latitude": facility.get("lat"),
                        "longitude": facility.get("lon")
                    }
                    
            # --- Add detailed eligibility for top 3 eligible patients ---
            if eval['eligible'] and eligible_patient_count < 3:
                eligibility_details = {'met': [], 'not_met': [], 'unsure': []}
                raw_criteria = eval.get('raw_criteria_results', [])
                for criterion in raw_criteria:
                    try:
                        detail = {
                            'criterion': criterion.get('criterion'),
                            'type': criterion.get('type'),
                            'status': criterion.get('status'),
                            'confidence': criterion.get('confidence'),
                            'reasoning': criterion.get('evidence', 'No evidence provided'),  # Get 'evidence' field, fallback to message
                            'sources': criterion.get('sources_used', [])  # Also add sources used
                        }
                        status_key = criterion.get('status', '').replace(' ', '_') # 'not met' -> 'not_met'
                        if status_key in eligibility_details:
                            eligibility_details[status_key].append(detail)
                        else:
                            # Handle unexpected status values gracefully if needed
                            logger.warning(f"Unexpected criterion status '{criterion.get('status')}' for patient {eval['patient_id']}")
                            eligibility_details.setdefault('other', []).append(detail)
                            
                    except Exception as e:
                         logger.error(f"Error processing criterion detail for patient {eval['patient_id']}: {criterion} - {e}")
                         
                patient_data['eligibility_details'] = eligibility_details
                eligible_patient_count += 1
            # --- End detailed eligibility --- 
            
            # Add to appropriate list based on eligibility
            if eval['eligible']:
                matching_patients.append(patient_data)
            else:
                non_matching_patients.append(patient_data)
                
        frontend_data["matching_patients"] = matching_patients
        frontend_data["non_matching_patients"] = non_matching_patients
        
        return frontend_data
        
    def process_and_save_results(self, llm_results, patient_lookup=None):
        """
        Process LLM results, create scoring evaluations, and prepare frontend data.
        
        Args:
            llm_results (list): LLM evaluation results from LLMEvaluator
            patient_lookup (dict, optional): Dictionary mapping patient_id to PatientData
            
        Returns:
            tuple: (evaluations, frontend_data)
        """
        print(f"Post-processing and scoring {len(llm_results)} patient matches for trial {self.nct_id}...")
        
        # Process results
        evaluations = self.process_llm_results(llm_results, patient_lookup)
        
        # Prepare frontend data
        frontend_data = self.prepare_frontend_response(evaluations)
        
        # Save evaluations and frontend data
        eval_filename = f"data/evaluations_{self.nct_id}.json"
        frontend_filename = f"data/frontend_{self.nct_id}.json"
        
        with open(eval_filename, 'w') as f:
            json.dump(evaluations, f, indent=2)
            
        with open(frontend_filename, 'w') as f:
            json.dump(frontend_data, f, indent=2)
            
        print(f"Saved evaluation results to {eval_filename}")
        print(f"Saved frontend data to {frontend_filename}")
        
        return evaluations, frontend_data

# ======================================================================
# Compatibility wrapper functions for pipeline integration
# ======================================================================

def process_trials(trials_list, patient_data=None):
    """
    Compatibility wrapper for the enhanced pipeline.
    Processes a list of trial results and returns optimally sorted evaluations.
    
    Args:
        trials_list (list): List of trial result dictionaries from LLM evaluation
        patient_data: Optional patient data for condition cross-referencing
        
    Returns:
        list: Intelligently sorted list of trial evaluations
    """
    evaluations = []
    
    # Process each trial
    for trial in trials_list:
        # Create a minimal trial summary for the MatchScorer
        trial_summary = {
            "nct_id": trial.get("nct_id"),
            "title": trial.get("title"),
        }
        
        # Create MatchScorer instance for this trial
        scorer = MatchScorer(trial_summary)
        
        # Evaluate the trial using the new scoring system
        # The trial dict should contain the LLM results
        eval_result = scorer.evaluate_match(trial, patient_data)
        evaluations.append(eval_result)
    
    # Sort evaluations using the new logic
    # Primary sort: eligible trials first
    # Secondary sort: normalized score (highest first)  
    # Tertiary sort: priority-weighted raw score ratio (highest first)
    evaluations.sort(key=lambda x: (
        x['eligible'], 
        x['score'], 
                 x['raw_score'] / (x['max_possible'] if x['max_possible'] > 0 else 1)
    ), reverse=True)
    
    return evaluations

## this is for one patient to many trials function 
def prepare_frontend_response(evaluations, patient_id="default"):
    """
    Simple frontend response preparation.
    
    Args:
        evaluations (list): List of trial evaluation results
        trial_results (list, optional): Not used, kept for compatibility
        
    Returns:
        dict: Simple structured data for frontend display
    """
    # Simple summary
    frontend_data = {
        "summary": {
            "total_matching_trials": len([e for e in evaluations if e['eligible']]),
            "total_evaluated": len(evaluations),
            "patient_id": patient_id
        }
    }
    
    # Split into matching and non-matching (evaluations should already be sorted)
    matching_trials = [eval for eval in evaluations if eval['eligible']]
    non_matching_trials = [eval for eval in evaluations if not eval['eligible']]
    
    frontend_data["matching_trials"] = matching_trials
    frontend_data["non_matching_trials"] = non_matching_trials
    frontend_data["ranking_trace"] = [
        {
            "trial_nct_id": eval.get("nct_id"),
            "title": eval.get("title"),
            "retrieval_rank": eval.get("retrieval_rank"),
            "rerank_rank": eval.get("rerank_rank"),
            "match_recommendation": eval.get("match_recommendation"),
            "eligibility_probability": eval.get("eligibility_probability"),
            "system_rationale_short": eval.get("system_rationale_short"),
        }
        for eval in evaluations
    ]
    
    return frontend_data
