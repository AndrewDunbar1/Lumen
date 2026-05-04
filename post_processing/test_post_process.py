import pickle
import sys
from pathlib import Path
# i was using this in root dir, now i moved it to post_processing dir before uploading

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from post_processing.post_processing import process_trials
from post_processing.post_processing import prepare_frontend_response
import json
from full_pipeline_bm25 import DecimalEncoder

def test_post_processing(patient_name):
    """Test post-processing with pickled data."""
    
    # Load the pickled objects
    with open(f"/Users/nadim/Documents/projects/cureva_llm/cureva/data/pickeled/results_{patient_name}.pkl", "rb") as f:
        results = pickle.load(f)
    
    with open(f"/Users/nadim/Documents/projects/cureva_llm/cureva/data/pickeled/patient_{patient_name}.pkl", "rb") as f:
        patient = pickle.load(f)
    
    print(f"Loaded {len(results)} trial results for patient {patient_name}")
    
    # Test the post-processing
    evaluations = process_trials(results, patient)
    # with open(f"data/post_process_testing/evaluations_{patient_name}.json", "w") as f:
    #     json.dump(evaluations, f, indent=2, default=str)  # default=str handles any non-serializable objects
    # exit()
    # Print summary
    frontend_data = prepare_frontend_response(evaluations, patient.personal_info.last_name)

    # Save data for UI
    frontend_filename = f"data/post_process_testing/frontend_data_{patient.personal_info.last_name}.json"

    with open(frontend_filename, "w") as f:
        json.dump(frontend_data, f, indent=4, cls=DecimalEncoder)

    print(f"Saved frontend data to {frontend_filename}")

if __name__ == "__main__":
    test_post_processing("Patient_12838187")