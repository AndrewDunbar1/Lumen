"""
Enhanced post-processing system that combines:
1. Auto-rejection for critical criteria
2. Weighted scoring for quality assessment
3. Normalized scoring to account for trial complexity
4. Confidence-based contribution to overall match quality
5. PRIORITIZATION of trials for rare and serious conditions using authoritative databases

For each criterion:
1. First check for critical rejection criteria:
   - If inclusion criterion is "not met" with confidence > 0.8 → Critical reject
   - If exclusion criterion is "met" with confidence > 0.8 → Critical reject

2. For eligible trials, advanced scoring:
   - Base scores: inclusion met (+2), inclusion met medium confidence (+1), exclusion not met (+1)
   - Normalized by total possible score to handle varying complexity
   - Weighted by criterion importance
   - Confidence-adjusted contributions
   - MAJOR score boost for trials targeting patient's specific rare/serious conditions
"""

import json
import math
import re
from collections import Counter
import logging

# Import the rare disease database
from post_processing.rare_conditions_db import (
    get_canonical_condition_name,
    get_condition_data,
    calculate_severity_modifier,
    is_rare_condition,
    is_severe_condition,
    calculate_patient_condition_priority
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================================================================
# Condition Extraction and Analysis
# ======================================================================

def extract_conditions_from_text(text):
    """
    Extract condition mentions from text using advanced patterns.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        list: List of extracted condition text
    """
    if not text:
        return []
        
    conditions = []
    
    # Advanced patterns to identify condition mentions
    patterns = [
        # Diagnosis patterns
        r"(?:diagnosed with|diagnosis of|have|has|with)\s+([a-z\s\-]+(?:disease|disorder|syndrome|cancer|tumor|condition))",
        r"(?:history of|confirmed)\s+([a-z\s\-]+(?:disease|disorder|syndrome|cancer|tumor|condition))",
        
        # Direct condition mentions
        r"(?:^|[.\s])([a-z\s\-]+(?:disease|disorder|syndrome|cancer|tumor|condition))",
        
        # Specific disease patterns
        r"(stage [iv\d]+ [a-z\s\-]+(?:cancer|tumor|melanoma|carcinoma|sarcoma))",
        r"((?:metastatic|advanced|recurrent|refractory) [a-z\s\-]+(?:cancer|tumor|melanoma|carcinoma|sarcoma))",
        
        # Common condition patterns without disease/disorder
        r"(?:diagnosed with|diagnosis of|have|has|with)\s+([a-z\s\-]+(?:diabetes|asthma|arthritis|hypertension|fibrosis|sclerosis))"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            for match in matches:
                match = match.strip()
                if len(match) > 3:  # Avoid very short matches
                    conditions.append(match)
    
    # Remove duplicates
    return list(set(conditions))

def extract_conditions_from_trial(trial_data):
    """
    Extract condition mentions from trial data using advanced patterns.
    
    Args:
        trial_data (dict): Dictionary containing trial data
        
    Returns:
        list: List of condition information extracted from criteria
    """
    conditions = []
    
    # Extract from inclusion/exclusion criteria text
    for criterion in trial_data.get("results", []):
        criterion_text = criterion.get("criterion", "")
        if not criterion_text:
            continue
        
        # Filter out non-medical criteria
        if any(x in criterion_text.lower() for x in ["consent", "age", "willing to", "able to", "agree to"]):
            continue
            
        # Extract conditions from criterion text
        extracted = extract_conditions_from_text(criterion_text)
        
        for condition_text in extracted:
            # Get canonical name from database if possible
            canonical_name = get_canonical_condition_name(condition_text)
            
            conditions.append({
                "text": condition_text,
                "canonical_name": canonical_name,
                "context": criterion_text,
                "type": criterion.get("type", "unknown")
            })
    
    # Remove duplicates while preserving context
    unique_conditions = {}
    for cond in conditions:
        text = cond["canonical_name"]  # Use canonical name for deduplication
        if text not in unique_conditions:
            unique_conditions[text] = cond
    
    return list(unique_conditions.values())

def analyze_condition_priority(condition_text, patient_conditions=None):
    """
    Analyzes a condition's priority based on rarity, severity, and patient relevance.
    
    Args:
        condition_text (str): The condition text to analyze
        patient_conditions (list): Optional list of patient's conditions
        
    Returns:
        dict: Priority information including scores and multiplier
    """
    # First check if this condition matches a patient's condition (highest priority)
    patient_priority = calculate_patient_condition_priority(condition_text, patient_conditions)
    if patient_priority["is_patient_condition"]:
        # This is one of the patient's conditions - return the high priority immediately
        # This will ensure trials for the patient's rare/severe conditions always come first
        return {
            "canonical_name": get_canonical_condition_name(condition_text),
            "rarity_score": 0.9 if patient_priority["is_rare"] else 0.5,
            "severity_score": 0.9 if patient_priority["is_severe"] else 0.5,
            "patient_relevance": 1.0,
            "priority_multiplier": patient_priority["priority_multiplier"]
        }
    
    # Get canonical name and condition data
    canonical_name = get_canonical_condition_name(condition_text)
    condition_data = get_condition_data(canonical_name)
    
    # Default values
    rarity_score = 0.0
    severity_score = 0.0
    patient_relevance = 0.0
    
    # If in our database, use those values
    if condition_data:
        rarity_score = condition_data.get("rarity_score", 0.0)
        severity_score = condition_data.get("base_severity", 0.0)
    else:
        # For conditions not in database, estimate based on text
        if "rare" in condition_text.lower():
            rarity_score = 0.7
        if any(term in condition_text.lower() for term in ["cancer", "tumor", "malignant"]):
            severity_score = 0.7
    
    # Apply severity modifiers based on text
    severity_modifier = calculate_severity_modifier(condition_text)
    severity_score = min(max(severity_score + severity_modifier, 0.1), 0.9)
    
    # Calculate priority multiplier:
    # 1. Base value of 1.0
    # 2. Add up to +1.0 for combined rarity/severity (if no patient match)
    base_multiplier = 1.0
    condition_multiplier = base_multiplier + ((rarity_score + severity_score) / 2.0) * 0.5
    
    # Cap at 2.0 (non-patient-matching conditions should always rank lower than patient matches)
    condition_multiplier = min(max(condition_multiplier, 1.0), 2.0)
    
    return {
        "canonical_name": canonical_name,
        "rarity_score": round(rarity_score, 2),
        "severity_score": round(severity_score, 2),
        "patient_relevance": patient_relevance,
        "priority_multiplier": round(condition_multiplier, 2)
    }

def find_condition_priorities(trial_data, patient_conditions=None):
    """
    Analyze trial data to find and prioritize conditions based on rarity, severity,
    and patient relevance.
    
    Args:
        trial_data (dict): Dictionary containing trial data
        patient_conditions (list): Optional list of patient's conditions
        
    Returns:
        dict: Information about identified conditions and their priority scores
    """
    # Extract conditions from trial criteria
    conditions = extract_conditions_from_trial(trial_data)
    
    # Analyze each condition
    priority_info = {
        "highest_multiplier": 1.0,
        "conditions": []
    }
    
    for condition in conditions:
        # Get text and analyze
        text = condition["text"]
        
        # Calculate priority
        priority_data = analyze_condition_priority(text, patient_conditions)
        
        # Store information
        condition_info = {
            "text": text,
            "canonical_name": priority_data["canonical_name"],
            "context": condition["context"][:100] + "..." if len(condition["context"]) > 100 else condition["context"],
            "type": condition["type"],
            "rarity_score": priority_data["rarity_score"],
            "severity_score": priority_data["severity_score"],
            "patient_relevance": priority_data["patient_relevance"],
            "priority_multiplier": priority_data["priority_multiplier"]
        }
        
        priority_info["conditions"].append(condition_info)
        
        # Keep track of highest multiplier
        if priority_data["priority_multiplier"] > priority_info["highest_multiplier"]:
            priority_info["highest_multiplier"] = priority_data["priority_multiplier"]
    
    # Sort conditions by priority
    priority_info["conditions"].sort(key=lambda x: x["priority_multiplier"], reverse=True)
    
    return priority_info

# ======================================================================
# Core Evaluation Logic
# ======================================================================

def evaluate_trial_results(trial_results, patient_conditions=None):
    """
    Evaluates trial eligibility using an enhanced scoring system for accurate matching.
    
    Args:
        trial_results (dict): Dictionary containing trial results with criteria evaluations
        patient_conditions (list): Optional list of patient's conditions for cross-referencing
        
    Returns:
        dict: Evaluation results with normalized score, eligibility status, and detailed metrics
    """
    # Initialize metrics
    rejected = False
    rejection_reasons = []
    criteria_counts = {'inclusion': 0, 'exclusion': 0, 'total': 0}
    
    # Score components - separate from rejection logic
    raw_score = 0
    max_possible_score = 0
    criteria_contributions = []
    
    # Track criteria counts for metrics
    status_counts = Counter()
    
    # Analyze conditions in this trial for priority, with patient condition cross-referencing
    condition_priority_info = find_condition_priorities(trial_results, patient_conditions)
    priority_multiplier = condition_priority_info["highest_multiplier"]
    condition_details = condition_priority_info["conditions"]
    
    # First pass - identify auto-rejections and count criteria
    for criterion in trial_results.get('results', []):
        # Skip malformed criteria
        try:
            criterion_type = criterion['type']
            status = criterion['status']
            confidence = criterion['confidence']
            
            # Count for metrics
            criteria_counts[criterion_type] += 1
            criteria_counts['total'] += 1
            status_counts[status] += 1
            
            # Check for auto-rejections
            if criterion_type == 'inclusion' and status == 'not met' and confidence > 0.8:
                rejected = True
                rejection_reasons.append(f"Failed inclusion criterion: {criterion['criterion']}")
                
            elif criterion_type == 'exclusion' and status == 'met' and confidence > 0.8:
                rejected = True
                rejection_reasons.append(f"Met exclusion criterion: {criterion['criterion']}")
                
        except (KeyError, TypeError) as e:
            logger.warning(f"Malformed criterion in trial {trial_results['nct_id']}: {criterion} - Error: {str(e)}")
            continue
    
    # Second pass - calculate scoring (only if not auto-rejected)
    if not rejected:
        for criterion in trial_results.get('results', []):
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
                        criterion_score = 0
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
                
                # Store contribution for analysis
                criteria_contributions.append({
                    'criterion': criterion['criterion'],
                    'type': criterion_type,
                    'contribution': weighted_score,
                    'possible': weighted_max
                })
                
                # Add to totals
                raw_score += weighted_score
                max_possible_score += weighted_max
                
            except (KeyError, TypeError):
                # Skip malformed criteria in scoring
                continue
    
    # Apply condition priority boost to raw score
    raw_score_with_priority = raw_score * priority_multiplier
    
    # Calculate normalized score (0-100 scale)
    normalized_score = 0
    if max_possible_score > 0 and not rejected:
        # Using sigmoid normalization to better distinguish quality differences
        # This creates a 0-100 scale with better distribution in the middle range
        raw_ratio = raw_score_with_priority / max_possible_score
        normalized_score = int(100 * (1 / (1 + math.exp(-10 * (raw_ratio - 0.5)))))
    
    # Prepare evaluation results
    evaluation = {
        'nct_id': trial_results['nct_id'],
        'title': trial_results['title'],
        'score': normalized_score,
        'raw_score': round(raw_score, 2),
        'priority_multiplier': round(priority_multiplier, 2),
        'raw_score_with_priority': round(raw_score_with_priority, 2),
        'max_possible': round(max_possible_score, 2),
        'eligible': not rejected,
        'rejection_reasons': rejection_reasons if rejected else [],
        'condition_analysis': {
            'conditions': condition_details,
            'count': len(condition_details)
        },
        'metrics': {
            'criteria_counts': criteria_counts,
            'status_distribution': dict(status_counts),
        }
    }
    
    return evaluation

def extract_patient_conditions(patient_data):
    """
    Extract condition names from patient data.
    
    Args:
        patient_data: Patient data object or dictionary
        
    Returns:
        list: List of patient's condition names
    """
    conditions = []
    
    # Handle different formats of patient data
    if hasattr(patient_data, 'conditions') and hasattr(patient_data.conditions, 'get_active_conditions'):
        # PatientData object from patient_parser
        conditions = patient_data.conditions.get_active_conditions()
    elif isinstance(patient_data, dict) and 'conditions' in patient_data:
        # Dictionary format
        conditions = patient_data['conditions']
    elif isinstance(patient_data, str):
        # Text format - extract conditions from text
        conditions = extract_conditions_from_text(patient_data)
    
    return conditions

def process_trials(trials_list, patient_data=None):
    """
    Process a list of trial results and return optimally sorted evaluations.
    
    Args:
        trials_list (list): List of trial result dictionaries
        patient_data: Optional patient data for condition cross-referencing
        
    Returns:
        list: Intelligently sorted list of trial evaluations
    """
    # Extract patient conditions if patient data is provided
    patient_conditions = extract_patient_conditions(patient_data) if patient_data else None
    
    if patient_conditions:
        logger.info(f"Found {len(patient_conditions)} patient conditions for priority matching")
    
    evaluations = []
    
    for trial in trials_list:
        eval_result = evaluate_trial_results(trial, patient_conditions)
        evaluations.append(eval_result)
    
    # Primary sort: eligible trials first
    # Secondary sort: normalized score (highest first)
    # Tertiary sort: priority-weighted raw score / max_possible (highest ratio first)
    evaluations.sort(key=lambda x: (
        x['eligible'], 
        x['score'], 
        x['raw_score_with_priority'] / (x['max_possible'] if x['max_possible'] > 0 else 1)
    ), reverse=True)
    
    return evaluations

def print_evaluations(evaluations):
    """
    Print detailed evaluation results for debugging and analysis.
    """
    print("\n=== Processed Trial Evaluations ===")
    for eval in evaluations:
        print(f"\nTrial: {eval['nct_id']}")
        print(f"Title: {eval['title']}")
        print(f"Score: {eval['score']} (Raw: {eval['raw_score']}, Priority: {eval['priority_multiplier']}x → {eval['raw_score_with_priority']}/{eval['max_possible']})")
        print(f"Eligible: {eval['eligible']}")
        
        if eval['condition_analysis']['count'] > 0:
            print("Conditions:")
            for i, cond in enumerate(eval['condition_analysis']['conditions'][:3]):  # Show top 3
                patient_match = " (PATIENT MATCH)" if cond.get('patient_relevance', 0) > 0 else ""
                print(f"  - {cond['text']} (R:{cond['rarity_score']}, S:{cond['severity_score']}, M:{cond['priority_multiplier']}){patient_match}")
            if eval['condition_analysis']['count'] > 3:
                print(f"  ... and {eval['condition_analysis']['count'] - 3} more")
                
        if not eval['eligible']:
            print("Rejection reasons:")
            for reason in eval['rejection_reasons']:
                print(f"- {reason}")
        print(f"Criteria: {eval['metrics']['criteria_counts']} | Status: {eval['metrics']['status_distribution']}")
        print("-" * 80)

def prepare_frontend_response(evaluations, trial_results):
    """
    Prepares a structured response for the frontend with enhanced matching quality.
    
    Args:
        evaluations (list): List of dictionaries containing enhanced trial evaluation results
        trial_results (list): List of dictionaries containing detailed trial criteria results
    
    Returns:
        dict: Structured data optimized for frontend display with accurate match quality
    """
    frontend_data = {
        "summary": {
            "total_matching_trials": len([e for e in evaluations if e['eligible']]),
            "total_evaluated": len(evaluations),
            "has_patient_matched_trials": any(
                any(c.get('patient_relevance', 0) > 0 for c in e['condition_analysis']['conditions']) 
                for e in evaluations if e['eligible']
            ),
            "has_rare_disease_trials": any(
                any(c.get('rarity_score', 0) > 0.8 for c in e['condition_analysis']['conditions']) 
                for e in evaluations if e['eligible']
            )
        }
    }
    
    # Prepare matching and non-matching trials lists
    matching_trials = []
    non_matching_trials = []
    
    for eval in evaluations:
        # Get the detailed results for this trial
        trial_detail = None
        trial_object = None
        for t in trial_results:
            if t['nct_id'] == eval['nct_id']:
                trial_detail = t['results']
                trial_object = t
                break
                
        if trial_detail is None:
            logger.warning(f"Could not find results for trial {eval['nct_id']}")
            continue
            
        # Build enhanced trial data
        trial_data = {
            "nct_id": eval['nct_id'],
            "title": eval['title'],
            "score": eval['score'],  # This is now the normalized score
            "priority_factor": eval['priority_multiplier'],  # Add priority factor for UI
            "eligible": eval['eligible'],
            "rejection_reasons": eval['rejection_reasons'],
            "criteria_summary": {
                "total": eval['metrics']['criteria_counts']['total'],
                "met": eval['metrics']['status_distribution'].get('met', 0),
                "not_met": eval['metrics']['status_distribution'].get('not met', 0),
                "unsure": eval['metrics']['status_distribution'].get('unsure', 0)
            },
            "condition_analysis": {
                "priority_conditions": [c for c in eval['condition_analysis']['conditions'] 
                                       if c['priority_multiplier'] > 1.5][:3],  # Top high-priority conditions
                "has_rare_condition": any(c['rarity_score'] > 0.7 for c in eval['condition_analysis']['conditions']),
                "has_severe_condition": any(c['severity_score'] > 0.7 for c in eval['condition_analysis']['conditions']),
                "has_patient_match": any(c.get('patient_relevance', 0) > 0 for c in eval['condition_analysis']['conditions'])
            },
            "detailed_criteria": trial_detail
        }
        
        # Add distance information if available in the trial object
        if trial_object and "closest_distance" in trial_object:
            trial_data["distance"] = round(trial_object["closest_distance"], 1)
            
            # Add facility information if available
            if "closest_facility" in trial_object:
                facility = trial_object["closest_facility"]
                if facility and len(facility) >= 7:
                    trial_data["closest_facility"] = {
                        "name": facility[0],
                        "city": facility[1],
                        "state": facility[2],
                        "zip": facility[3],
                        "country": facility[4],
                        "latitude": facility[5],
                        "longitude": facility[6]
                    }
        
        # Add to appropriate list based on eligibility
        if eval['eligible']:
            matching_trials.append(trial_data)
        else:
            non_matching_trials.append(trial_data)
    
    # Sort matching trials by a combination of score and distance
    # Trials closer to the patient get a slight boost
    if matching_trials:
        # First sort by score (descending)
        matching_trials.sort(key=lambda x: x['score'], reverse=True)
        
        # For trials with similar scores (within 10 points), prioritize closer ones
        for i in range(len(matching_trials)):
            for j in range(i+1, len(matching_trials)):
                # If scores are close but trial j is significantly closer
                if (abs(matching_trials[i]['score'] - matching_trials[j]['score']) <= 10 and
                    'distance' in matching_trials[i] and 'distance' in matching_trials[j] and
                    matching_trials[j]['distance'] < matching_trials[i]['distance'] * 0.7):
                    # Swap positions
                    matching_trials[i], matching_trials[j] = matching_trials[j], matching_trials[i]
    
    frontend_data["matching_trials"] = matching_trials
    frontend_data["non_matching_trials"] = non_matching_trials
    
    return frontend_data

if __name__ == "__main__":
    # Load and process trial results from output file
    with open("data/output_Katharina121.json", "r") as f:
        trial_results = json.load(f)
        
    # Process the trials and get sorted evaluations
    evaluations = process_trials(trial_results)

    frontend_data = prepare_frontend_response(evaluations, trial_results)
    
    # Print results
    print_evaluations(evaluations)
    print(json.dumps(frontend_data, indent=4))
