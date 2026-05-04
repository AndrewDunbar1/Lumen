"""
Rare Disease Database and Helper Functions

This module provides standardized data and functions for identifying and evaluating
rare and serious medical conditions based on authoritative sources including:
- Orphanet
- GARD (Genetic and Rare Diseases Information Center)
- NORD (National Organization for Rare Disorders)

It supports:
- Disease synonym/abbreviation matching
- Severity classification
- Automated prioritization based on rarity and severity
- External database integration for comprehensive coverage
"""
import os
import re
import json
import csv
import requests
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory for cached external data
DATA_CACHE_DIR = Path("data/disease_databases")
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Standardized rare disease database with synonyms, abbreviations, and codes
RARE_DISEASE_DATABASE = {
    # Rare genetic disorders (with synonyms, abbreviations, and prevalence data)
    "tuberous sclerosis complex": {
        "synonyms": ["tuberous sclerosis", "TSC", "Bourneville disease", "epiloia"],
        "codes": {"icd10": "Q85.1", "orpha": "805", "snomed": "7199000"},
        "prevalence": "1-9/100,000",
        "rarity_score": 0.95,
        "base_severity": 0.85
    },
    "neurofibromatosis": {
        "synonyms": ["NF1", "von recklinghausen disease", "neurofibromatosis type 1"],
        "codes": {"icd10": "Q85.0", "orpha": "636", "snomed": "92824003"},
        "prevalence": "1-5/10,000",
        "rarity_score": 0.90,
        "base_severity": 0.80
    },
    "cystic fibrosis": {
        "synonyms": ["CF", "mucoviscidosis"],
        "codes": {"icd10": "E84", "orpha": "586", "snomed": "190905008"},
        "prevalence": "1-9/10,000",
        "rarity_score": 0.85,
        "base_severity": 0.85
    },
    "huntington disease": {
        "synonyms": ["huntington's disease", "huntington's chorea", "HD"],
        "codes": {"icd10": "G10", "orpha": "399", "snomed": "58756001"},
        "prevalence": "1-9/100,000",
        "rarity_score": 0.90,
        "base_severity": 0.90
    },
    "gaucher disease": {
        "synonyms": ["gaucher's disease", "glucocerebrosidase deficiency"],
        "codes": {"icd10": "E75.2", "orpha": "355", "snomed": "190794006"},
        "prevalence": "1-9/100,000",
        "rarity_score": 0.90,
        "base_severity": 0.80
    },
    "fabry disease": {
        "synonyms": ["anderson-fabry disease", "angiokeratoma corporis diffusum", "alpha-galactosidase A deficiency"],
        "codes": {"icd10": "E75.2", "orpha": "324", "snomed": "16652001"},
        "prevalence": "1-5/10,000",
        "rarity_score": 0.85,
        "base_severity": 0.75
    },
    "pompe disease": {
        "synonyms": ["glycogen storage disease type II", "acid maltase deficiency", "GSD II"],
        "codes": {"icd10": "E74.0", "orpha": "365", "snomed": "76812004"},
        "prevalence": "<1/1,000,000",
        "rarity_score": 0.95,
        "base_severity": 0.90
    },
    "fragile x syndrome": {
        "synonyms": ["fragile x", "FRAXA", "martin-bell syndrome"],
        "codes": {"icd10": "Q99.2", "orpha": "908", "snomed": "613003"},
        "prevalence": "1-5/10,000",
        "rarity_score": 0.85,
        "base_severity": 0.75
    },
    "duchenne muscular dystrophy": {
        "synonyms": ["DMD", "duchenne", "pseudohypertrophic muscular dystrophy"],
        "codes": {"icd10": "G71.0", "orpha": "98896", "snomed": "76670001"},
        "prevalence": "1-9/100,000",
        "rarity_score": 0.90,
        "base_severity": 0.85
    },
    "rett syndrome": {
        "synonyms": ["rett disorder", "RTT", "cerebroatrophic hyperammonemia"],
        "codes": {"icd10": "F84.2", "orpha": "778", "snomed": "68618008"},
        "prevalence": "1-9/100,000",
        "rarity_score": 0.90,
        "base_severity": 0.80
    },
    "ehlers-danlos syndrome": {
        "synonyms": ["EDS", "elastic skin", "cutis hyperelastica"],
        "codes": {"icd10": "Q79.6", "orpha": "98249", "snomed": "69180006"},
        "prevalence": "1-9/100,000",
        "rarity_score": 0.85,
        "base_severity": 0.70
    },
    "amyotrophic lateral sclerosis": {
        "synonyms": ["ALS", "lou gehrig's disease", "motor neuron disease", "charcot disease"],
        "codes": {"icd10": "G12.2", "orpha": "803", "snomed": "86044005"},
        "prevalence": "1-9/100,000",
        "rarity_score": 0.90,
        "base_severity": 0.95
    },
    "sickle cell disease": {
        "synonyms": ["sickle cell anemia", "SCD", "HbSS", "sickle cell disorder"],
        "codes": {"icd10": "D57", "orpha": "232", "snomed": "127040003"},
        "prevalence": "1-5/10,000",
        "rarity_score": 0.80,
        "base_severity": 0.85
    },
    
    # Rare cancers and tumors (with synonyms, abbreviations, and prevalence data)
    "glioblastoma": {
        "synonyms": ["glioblastoma multiforme", "GBM", "grade IV astrocytoma"],
        "codes": {"icd10": "C71", "orpha": "360090", "snomed": "63634009"},
        "prevalence": "1-9/100,000",
        "rarity_score": 0.90,
        "base_severity": 0.95
    },
    "mesothelioma": {
        "synonyms": ["malignant mesothelioma", "pleural mesothelioma"],
        "codes": {"icd10": "C45", "orpha": "50251", "snomed": "423751000"},
        "prevalence": "<1/100,000",
        "rarity_score": 0.90,
        "base_severity": 0.95
    },
    "chordoma": {
        "synonyms": ["notochordal tumor"],
        "codes": {"icd10": "C41", "orpha": "178", "snomed": "443659008"},
        "prevalence": "<1/1,000,000",
        "rarity_score": 0.95,
        "base_severity": 0.90
    },
    "adrenocortical carcinoma": {
        "synonyms": ["adrenal cortical carcinoma", "adrenal cancer", "ACC"],
        "codes": {"icd10": "C74.0", "orpha": "1501", "snomed": "363478007"},
        "prevalence": "<1/1,000,000",
        "rarity_score": 0.95,
        "base_severity": 0.90
    },
    "retinoblastoma": {
        "synonyms": ["RB"],
        "codes": {"icd10": "C69.2", "orpha": "790", "snomed": "370967009"},
        "prevalence": "1-9/100,000",
        "rarity_score": 0.90,
        "base_severity": 0.85
    },
    
    # Rare autoimmune/inflammatory conditions
    "multiple sclerosis": {
        "synonyms": ["MS", "disseminated sclerosis"],
        "codes": {"icd10": "G35", "orpha": "802", "snomed": "24700007"},
        "prevalence": "5-9/10,000",
        "rarity_score": 0.70,
        "base_severity": 0.80
    },
    "myasthenia gravis": {
        "synonyms": ["MG", "goldflam disease"],
        "codes": {"icd10": "G70.0", "orpha": "589", "snomed": "91637004"},
        "prevalence": "1-9/100,000",
        "rarity_score": 0.85,
        "base_severity": 0.75
    },
    "systemic lupus erythematosus": {
        "synonyms": ["lupus", "SLE", "disseminated lupus erythematosus"],
        "codes": {"icd10": "M32", "orpha": "536", "snomed": "55464009"},
        "prevalence": "1-9/10,000",
        "rarity_score": 0.75,
        "base_severity": 0.80
    },
    "guillain-barre syndrome": {
        "synonyms": ["GBS", "acute inflammatory demyelinating polyneuropathy", "landry's paralysis"],
        "codes": {"icd10": "G61.0", "orpha": "2103", "snomed": "92824003"},
        "prevalence": "1-9/100,000",
        "rarity_score": 0.85,
        "base_severity": 0.80
    }
}

# Higher-severity cancer types that may appear in both common and rare forms
CANCER_TYPES = {
    "thyroid cancer": {
        "base_severity": 0.70,
        "rare_variants": ["anaplastic thyroid cancer", "medullary thyroid cancer"]
    },
    "sarcoma": {
        "base_severity": 0.85,
        "subtypes": ["osteosarcoma", "ewing sarcoma", "rhabdomyosarcoma", "liposarcoma", "leiomyosarcoma"]
    },
    "leukemia": {
        "base_severity": 0.80,
        "rare_variants": ["hairy cell leukemia", "chronic myelomonocytic leukemia"]
    },
    "lymphoma": {
        "base_severity": 0.75,
        "rare_variants": ["mantle cell lymphoma", "waldenstrom macroglobulinemia", "burkitt lymphoma"]
    },
    "brain tumor": {
        "base_severity": 0.85,
        "subtypes": ["glioblastoma", "oligodendroglioma", "ependymoma", "meningioma"]
    },
    "pancreatic cancer": {
        "base_severity": 0.90
    },
    "liver cancer": {
        "base_severity": 0.85
    },
    "esophageal cancer": {
        "base_severity": 0.85
    }
}

# Markers that indicate a condition is likely severe
SEVERITY_MARKERS = {
    "metastatic": 0.50,  # Very high severity addition
    "stage iv": 0.50,
    "stage 4": 0.50,
    "stage iii": 0.40,
    "stage 3": 0.40,
    "advanced": 0.40,  # High severity addition
    "refractory": 0.40,
    "recurrent": 0.30,  # Moderate severity addition
    "resistant": 0.30,
    "progressive": 0.30,
    "severe": 0.30,
    "uncontrolled": 0.20,  # Lower severity addition
    "chronic": 0.10,
    "mild": -0.20,  # Severity reduction
    "moderate": -0.10,
    "early stage": -0.30,
    "stage i": -0.30,
    "stage 1": -0.30,
    "stage ii": -0.20,
    "stage 2": -0.20,
    "localized": -0.20,
    "remission": -0.40,
    "controlled": -0.20
}

# =====================================================================
# External Database Integration
# =====================================================================

def fetch_orphanet_data():
    """
    Fetches rare disease data from Orphanet's API or public datasets.
    Caches the result for future use.
    
    Returns:
        dict: Mapping of disease names to their data
    """
    cache_file = DATA_CACHE_DIR / "orphanet_diseases.json"
    
    # Use cached data if available
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading cached Orphanet data: {e}")
    
    # For demonstration, we'll use a sample of Orphanet data
    # In production, this would fetch from Orphanet's API
    orphanet_data = {}
    
    # Sample entry format
    orphanet_data["alagille syndrome"] = {
        "orphacode": "52",
        "synonyms": ["ALGS", "arteriohepatic dysplasia"],
        "prevalence": "1-9/100,000",
        "rarity_score": 0.9,
        "base_severity": 0.8,
        "icd10": "Q93.2",
        "snomed": "68320005"
    }
    
    # Save to cache
    with open(cache_file, 'w') as f:
        json.dump(orphanet_data, f, indent=2)
    
    return orphanet_data

def fetch_gard_data():
    """
    Fetches rare disease data from GARD (Genetic and Rare Diseases Information Center).
    Caches the result for future use.
    
    Returns:
        dict: Mapping of disease names to their data
    """
    cache_file = DATA_CACHE_DIR / "gard_diseases.json"
    
    # Use cached data if available
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading cached GARD data: {e}")
    
    # For demonstration, we'll create a sample of GARD data
    # In production, this would fetch from GARD's API or parse their downloadable datasets
    gard_data = {}
    
    # Sample entries
    gard_data["3-methylglutaconic aciduria"] = {
        "gard_id": "5578",
        "synonyms": ["3-MGA", "MEGDEL syndrome"],
        "prevalence": "<1/1,000,000",
        "rarity_score": 0.95,
        "base_severity": 0.85
    }
    
    # Save to cache
    with open(cache_file, 'w') as f:
        json.dump(gard_data, f, indent=2)
    
    return gard_data

def fetch_nord_data():
    """
    Fetches rare disease data from NORD (National Organization for Rare Disorders).
    Caches the result for future use.
    
    Returns:
        dict: Mapping of disease names to their data
    """
    cache_file = DATA_CACHE_DIR / "nord_diseases.json"
    
    # Use cached data if available
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading cached NORD data: {e}")
    
    # For demonstration, we'll create a sample of NORD data
    # In production, this would parse NORD's database
    nord_data = {}
    
    # Sample entry
    nord_data["cherubism"] = {
        "nord_id": "1515",
        "synonyms": ["familial fibrous dysplasia of the jaws"],
        "prevalence": "<1/100,000",
        "rarity_score": 0.9,
        "base_severity": 0.7
    }
    
    # Save to cache
    with open(cache_file, 'w') as f:
        json.dump(nord_data, f, indent=2)
    
    return nord_data

def load_comprehensive_database():
    """
    Loads and merges disease data from all sources into a comprehensive database.
    This includes our internal database plus external sources.
    
    Returns:
        dict: Comprehensive disease database
    """
    # Start with our internal database
    comprehensive_db = RARE_DISEASE_DATABASE.copy()
    
    # Add Orphanet data
    try:
        orphanet_data = fetch_orphanet_data()
        for disease, data in orphanet_data.items():
            disease_lower = disease.lower()
            if disease_lower not in comprehensive_db:
                comprehensive_db[disease_lower] = {
                    "synonyms": data.get("synonyms", []),
                    "codes": {
                        "orpha": data.get("orphacode", ""),
                        "icd10": data.get("icd10", ""),
                        "snomed": data.get("snomed", "")
                    },
                    "prevalence": data.get("prevalence", ""),
                    "rarity_score": data.get("rarity_score", 0.85),
                    "base_severity": data.get("base_severity", 0.7),
                    "source": "orphanet"
                }
        logger.info(f"Added {len(orphanet_data)} diseases from Orphanet")
    except Exception as e:
        logger.error(f"Error adding Orphanet data: {e}")
    
    # Add GARD data
    try:
        gard_data = fetch_gard_data()
        for disease, data in gard_data.items():
            disease_lower = disease.lower()
            if disease_lower not in comprehensive_db:
                comprehensive_db[disease_lower] = {
                    "synonyms": data.get("synonyms", []),
                    "codes": {"gard": data.get("gard_id", "")},
                    "prevalence": data.get("prevalence", ""),
                    "rarity_score": data.get("rarity_score", 0.85),
                    "base_severity": data.get("base_severity", 0.7),
                    "source": "gard"
                }
        logger.info(f"Added {len(gard_data)} diseases from GARD")
    except Exception as e:
        logger.error(f"Error adding GARD data: {e}")
    
    # Add NORD data
    try:
        nord_data = fetch_nord_data()
        for disease, data in nord_data.items():
            disease_lower = disease.lower()
            if disease_lower not in comprehensive_db:
                comprehensive_db[disease_lower] = {
                    "synonyms": data.get("synonyms", []),
                    "codes": {"nord": data.get("nord_id", "")},
                    "prevalence": data.get("prevalence", ""),
                    "rarity_score": data.get("rarity_score", 0.85),
                    "base_severity": data.get("base_severity", 0.7),
                    "source": "nord"
                }
        logger.info(f"Added {len(nord_data)} diseases from NORD")
    except Exception as e:
        logger.error(f"Error adding NORD data: {e}")
    
    return comprehensive_db

# Load the comprehensive database
COMPREHENSIVE_DISEASE_DB = load_comprehensive_database()
logger.info(f"Loaded comprehensive disease database with {len(COMPREHENSIVE_DISEASE_DB)} conditions")

def build_lookup_dictionaries():
    """
    Builds comprehensive lookup dictionaries for efficient condition matching.
    Returns:
        tuple: (synonym_to_condition, code_to_condition) mappings
    """
    synonym_to_condition = {}
    code_to_condition = {}
    
    # Process the comprehensive disease database
    for condition, data in COMPREHENSIVE_DISEASE_DB.items():
        # Add the main condition name
        synonym_to_condition[condition.lower()] = condition
        
        # Add all synonyms
        for synonym in data.get("synonyms", []):
            synonym_to_condition[synonym.lower()] = condition
        
        # Add all codes
        for code_type, code in data.get("codes", {}).items():
            if code:  # Only add non-empty codes
                code_to_condition[f"{code_type}:{code}"] = condition
    
    return synonym_to_condition, code_to_condition

# Build the lookup dictionaries when module is imported
SYNONYM_TO_CONDITION, CODE_TO_CONDITION = build_lookup_dictionaries()

def get_canonical_condition_name(text):
    """
    Converts a condition text to its canonical name from our database.
    
    Args:
        text (str): The input condition text
        
    Returns:
        str: The canonical condition name if found, otherwise the original text
    """
    if not text:
        return text
        
    text_lower = text.lower()
    
    # Direct lookup in synonym dictionary
    if text_lower in SYNONYM_TO_CONDITION:
        return SYNONYM_TO_CONDITION[text_lower]
    
    # Check if text contains any of our known conditions or synonyms
    for synonym, condition in SYNONYM_TO_CONDITION.items():
        # Only match whole words
        if re.search(r'\b' + re.escape(synonym) + r'\b', text_lower):
            return condition
    
    # If no match, return the original text
    return text

def get_condition_data(condition_name):
    """
    Retrieves data for a condition from the comprehensive disease database.
    
    Args:
        condition_name (str): The canonical condition name
        
    Returns:
        dict: The condition data if found, otherwise None
    """
    # Try direct lookup in comprehensive database
    if condition_name in COMPREHENSIVE_DISEASE_DB:
        return COMPREHENSIVE_DISEASE_DB[condition_name]
    
    # Try canonical name lookup
    canonical_name = get_canonical_condition_name(condition_name)
    if canonical_name in COMPREHENSIVE_DISEASE_DB:
        return COMPREHENSIVE_DISEASE_DB[canonical_name]
    
    # Check cancer types
    for cancer_type, data in CANCER_TYPES.items():
        if cancer_type.lower() in condition_name.lower():
            # Check for rare variants
            for variant in data.get("rare_variants", []):
                if variant.lower() in condition_name.lower():
                    return {
                        "rarity_score": 0.85,
                        "base_severity": data["base_severity"] + 0.10
                    }
            # Check for subtypes
            for subtype in data.get("subtypes", []):
                if subtype.lower() in condition_name.lower():
                    return {
                        "rarity_score": 0.80,
                        "base_severity": data["base_severity"]
                    }
            # Regular cancer type
            return {
                "rarity_score": 0.60,
                "base_severity": data["base_severity"]
            }
    
    # Not in our database
    return None

def calculate_severity_modifier(condition_text):
    """
    Calculates a severity modifier based on severity markers in the text.
    
    Args:
        condition_text (str): The condition text to analyze
        
    Returns:
        float: The severity modifier value
    """
    if not condition_text:
        return 0.0
        
    text_lower = condition_text.lower()
    
    # Add up all applicable severity modifiers
    severity_modifier = 0.0
    for marker, value in SEVERITY_MARKERS.items():
        if marker in text_lower:
            severity_modifier += value
    
    # Cap the modifier to a reasonable range
    return max(min(severity_modifier, 0.5), -0.5)

def is_rare_condition(condition_text):
    """
    Determines if a condition is classified as rare based on our database.
    
    Args:
        condition_text (str): The condition text to analyze
        
    Returns:
        bool: True if the condition is rare, False otherwise
    """
    canonical_name = get_canonical_condition_name(condition_text)
    condition_data = get_condition_data(canonical_name)
    
    if condition_data and condition_data.get("rarity_score", 0) >= 0.80:
        return True
        
    return False

def is_severe_condition(condition_text):
    """
    Determines if a condition is classified as severe based on database and markers.
    
    Args:
        condition_text (str): The condition text to analyze
        
    Returns:
        bool: True if the condition is severe, False otherwise
    """
    text_lower = condition_text.lower()
    
    # Check for high severity markers
    for marker in ["metastatic", "stage iv", "stage 4", "terminal", "life-threatening"]:
        if marker in text_lower:
            return True
    
    # Check database
    canonical_name = get_canonical_condition_name(condition_text)
    condition_data = get_condition_data(canonical_name)
    
    if condition_data:
        severity = condition_data.get("base_severity", 0)
        severity += calculate_severity_modifier(condition_text)
        
        if severity >= 0.85:
            return True
    
    return False

# Maximum multiplier for patient-matching rare diseases
# This ensures trials for a patient's rare disease will appear first
MAX_PATIENT_RARE_DISEASE_MULTIPLIER = 5.0

def calculate_patient_condition_priority(condition_text, patient_conditions=None):
    """
    Calculates an extremely high priority multiplier for conditions that match
    a patient's rare or serious disease. This ensures that trials for a patient's
    rare disease will always appear first in the match list.
    
    Args:
        condition_text (str): The condition text to analyze
        patient_conditions (list): List of patient's conditions
        
    Returns:
        dict: Priority information for patient-matching conditions
    """
    # Default values
    is_patient_condition = False
    priority_multiplier = 1.0
    
    # Get canonical name for matching
    canonical_name = get_canonical_condition_name(condition_text)
    
    # Check if this is a rare or severe condition
    is_rare = is_rare_condition(canonical_name)
    is_severe = is_severe_condition(canonical_name)
    
    # Check if it matches one of the patient's conditions
    if patient_conditions:
        for patient_condition in patient_conditions:
            patient_canonical = get_canonical_condition_name(patient_condition)
            
            # Check for exact match or if one contains the other
            if (canonical_name == patient_canonical or 
                patient_canonical in canonical_name or
                canonical_name in patient_canonical):
                
                is_patient_condition = True
                
                # If patient has this rare or severe condition, give it maximum priority
                if is_rare or is_severe:
                    priority_multiplier = MAX_PATIENT_RARE_DISEASE_MULTIPLIER
                    break
                else:
                    # For non-rare patient conditions, still give a good boost but not maximum
                    priority_multiplier = 3.0
                    break
    
    return {
        "is_patient_condition": is_patient_condition,
        "is_rare": is_rare,
        "is_severe": is_severe,
        "priority_multiplier": priority_multiplier
    }

import re  # Required for regex operations 