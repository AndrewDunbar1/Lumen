import base64
from fhirclient.models.bundle import Bundle
from fhirclient.models.condition import Condition
from fhirclient.models.diagnosticreport import DiagnosticReport
from fhirclient.models.medicationrequest import MedicationRequest
from fhirclient.models.observation import Observation
from fhirclient.models.patient import Patient
from fhirclient.models.procedure import Procedure
import re 

from llm.langchain.medical_terms import generalize_conditions

class ConditionData:
    """Represents a single medical condition for a patient."""

    def __init__(self, condition_json: Condition):
        # self.id = condition_json.id
        # self.code = condition_json.code.text
        self.code = condition_json.code.coding[0].display
        self.id = condition_json.code.coding[0].code

        if condition_json.clinicalStatus and hasattr(condition_json.clinicalStatus, "coding"):
            if condition_json.clinicalStatus.coding:
                self.clinical_status = condition_json.clinicalStatus.coding[0].code
            else:
                self.clinical_status = "No-clinical-status-found"
        else:
            self.clinical_status = "No-clinical-status-found"

        if condition_json.verificationStatus and hasattr(condition_json.verificationStatus, "coding"):
            if condition_json.verificationStatus.coding:
                self.verification_status = condition_json.verificationStatus.coding[0].code
            else:
                self.verification_status = "No-verification-status-found"
        else:
            self.verification_status = "No-verification-status-found"

        self.onset = condition_json.onsetDateTime
        self.recorded_date = condition_json.recordedDate

    def __str__(self):
        onset_str = self.onset.isostring if self.onset else "Unknown"
        recorded_str = self.recorded_date.isostring if self.recorded_date else "Unknown"

        return (
            f"Condition ID: {self.id}\n"
            f"Description: {self.code}\n"
            f"Status: {self.clinical_status or 'Unknown'} ({self.verification_status or 'Unknown'})\n"
            f"Onset: {onset_str}\n"
            f"Recorded Date: {recorded_str}\n"
        )


class ConditionsManager:
    """Manages a collection of Condition objects with filtering and querying capabilities."""

    def __init__(self):
        self.conditions = []
        self.general_conditions = []

    def insert(self, condition_json: Condition):
        condition_obj = ConditionData(condition_json)
        self.conditions.append(condition_obj)

    def get_active_conditions(self) -> list[ConditionData]:
        return self.clean_conditions([condition.code for condition in self.conditions if condition.clinical_status.lower()
                == "active"])
    
    def get_condition_names(self):
        condition_name_list = [] 

        for condition_obj in self.conditions: 
            condition_name_list.append(condition_obj.code)
        
        return self.clean_conditions(condition_name_list)
    
    def clean_conditions(self, conditions):
        """Removes text inside parentheses and trims whitespace."""
        return [re.sub(r"\s*\(.*?\)", "", condition).strip() for condition in conditions]

    def get_general_conditions(self):
        if not self.general_conditions:
            results = generalize_conditions(self.get_condition_names())
            filtered_conditions = [r for r in results if r.lower() != "not applicable"]
            self.general_conditions = self.clean_conditions(filtered_conditions)

        return self.general_conditions

    def get_all_conditions(self) -> list[ConditionData]:
        return self.conditions

    def __len__(self) -> int:
        return len(self.conditions)

    def __iter__(self):
        return iter(self.conditions)

    def __str__(self) -> str:
        return "\n".join(str(condition) for condition in self.conditions)


class PersonalInfo:
    def __init__(self, patient_resource: Patient):
        self.id = patient_resource.id

        try:
            self.first_name = patient_resource.name[0].given[0]
        except (AttributeError, IndexError, KeyError, TypeError):
            self.first_name = "No-first-name-found"

        try:
            self.last_name = patient_resource.name[0].family
        except (AttributeError, IndexError, KeyError, TypeError):
            self.last_name = "No-last-name-found"

        self.gender = patient_resource.gender

        self.birthday = patient_resource.birthDate.date
        
        # Extract location data
        self.city = None
        self.state = None
        self.zip = None
        self.latitude = None
        self.longitude = None
        
        # Try to extract address information
        if hasattr(patient_resource, 'address') and patient_resource.address:
            address = patient_resource.address[0]
            if hasattr(address, 'city'):
                self.city = address.city
            if hasattr(address, 'state'):
                self.state = address.state
            if hasattr(address, 'postalCode'):
                self.zip = address.postalCode
            
            # Look for extensions that might contain geo coordinates
            if hasattr(address, 'extension') and address.extension:
                for ext in address.extension:
                    if ext.url == 'http://hl7.org/fhir/StructureDefinition/geolocation':
                        for value_ext in ext.extension:
                            if value_ext.url == 'latitude':
                                self.latitude = float(value_ext.valueDecimal)
                            elif value_ext.url == 'longitude':
                                self.longitude = float(value_ext.valueDecimal)

    def extract_name(self, names):
        """Extracts full name from the name field safely."""
        if not names:
            return "Unknown"
        given = " ".join(names[0].given)
        family = names[0].family
        return f"{given} {family}".strip()
    
    def get_location_string(self):
        """Returns a formatted location string if location data is available."""
        if self.city and self.state:
            return f"{self.city}, {self.state}"
        return "Unknown location"

    def __str__(self):
        location_str = self.get_location_string()
        return f"Patient ID: {self.id}\nName: {self.first_name} {self.last_name}\nGender: {self.gender}\nBirthDay: {self.birthday}\nLocation: {location_str}\n"


class ObservationLab:
    def __init__(self, observation_resource: Observation):
        if observation_resource.valueQuantity is not None:
            self.test_name = observation_resource.code.coding[0].display
            self.test_code = observation_resource.code.coding[0].code
            self.value = observation_resource.valueQuantity.value
            self.unit = observation_resource.valueQuantity.unit
            self.date = observation_resource.effectiveDateTime  # Default to "N/A" if missing

    def __str__(self):
        """Safely convert lab observation to string, handling missing attributes."""
        try:
            # Safely access attributes with fallbacks
            test_name = getattr(self, 'test_name', 'Unknown test')
            test_code = getattr(self, 'test_code', 'Unknown code')
            value = getattr(self, 'value', 'No value')
            unit = getattr(self, 'unit', 'No unit')
            
            # Handle date specially since it may need isostring conversion
            date_str = "Unknown date"
            if hasattr(self, 'date') and self.date:
                if hasattr(self.date, 'isostring'):
                    date_str = self.date.isostring
                else:
                    date_str = str(self.date)
            
            return (
                f"Test: {test_name}\n"
                f"\tCode: {test_code}\n"
                f"\tValue: {value}\n"
                f"\tUnit: {unit}\n"
                f"\tDate: {date_str}\n"
            )
        except Exception as e:
            # Last resort fallback if anything goes wrong
            return f"Lab Test: [Error displaying lab: {str(e)}]"

class MedicationData:
    def __init__(self, medication_request):
        med_codeable = getattr(medication_request, "medicationCodeableConcept", None)

        if med_codeable and med_codeable.coding:
            self.name = med_codeable.coding[0].display or "Unnamed medication"
            self.code = med_codeable.coding[0].code or "No code"
        else:
            self.name = "Unnamed medication"
            self.code = "No code"

        self.status = getattr(medication_request, "status", "Unknown")

        try:
            if medication_request.dosageInstruction and medication_request.dosageInstruction[0].text:
                self.dosage = medication_request.dosageInstruction[0].text
            else:
                self.dosage = "Unknown"
        except Exception:
            self.dosage = "Unknown"

    def __str__(self):
        return (
            f"Medication: {self.name}\n"
            f"  Code: {self.code}\n"
            f"  Status: {self.status}\n"
            f"  Dosage: {self.dosage}"
        )


class ObservationManager:
    def __init__(self):
        self.labs = []

    def insert(self, observation_resource: Observation):
        try:
            if observation_resource is None:
                return
                
            if not hasattr(observation_resource, 'category') or not observation_resource.category:
                return
                
            if observation_resource.category[0].coding[0].code == "laboratory":
                lab_obj = ObservationLab(observation_resource)
                self.labs.append(lab_obj)
        except (AttributeError, IndexError, TypeError) as e:
            print(f"Warning: Error processing observation: {e}")

    def return_all_labs(self):
        """Return all lab observations, filtering out None values"""
        # Ensure we're not returning None values or invalid labs
        valid_labs = []
        for lab in self.labs:
            if lab is not None:
                try:
                    # Basic validation - ensure the lab has the bare minimum attributes
                    # This will create them with defaults if they're missing
                    if not hasattr(lab, 'test_name'):
                        lab.test_name = "Unknown test"
                    if not hasattr(lab, 'value'):
                        lab.value = "No value"
                    valid_labs.append(lab)
                except Exception as e:
                    print(f"Warning: Skipping invalid lab observation: {e}")
        
        print(f"Returning {len(valid_labs)} valid labs from {len(self.labs)} total")
        return valid_labs


class ReportManager():
    def __init__(self):
        self.reports = []

    def insert(self, report: DiagnosticReport):
        if report.presentedForm is not None:
            full_note = []
            for form in report.presentedForm:
                decoded_text = base64.b64decode(form.data).decode("utf-8")
                full_note.append(decoded_text)

            self.reports.append(full_note)


class ProcedureData():
    def __init__(self, procedure_resource: Procedure):
        self.status = procedure_resource.status
        self.code = procedure_resource.code.coding[0].code
        self.name = procedure_resource.code.coding[0].display
        if procedure_resource.performedPeriod and hasattr(procedure_resource.performedPeriod, "start"):
            self.performed_time = procedure_resource.performedPeriod.start
        else:
            self.performed_time = "No-performed-time-found"

    def __str__(self):
        return (
            f"Procedure: {self.name}\n"
            f"\tCode: {self.code}\n"
            f"\tStatus: {self.status}\n"
            f"\tDate: {self.performed_time.isostring}\n"
        )


class ProcedureManager():
    def __init__(self):
        self.procedure_list = []

    def insert(self, procedure_resource: Procedure):
        procedure_data = ProcedureData(procedure_resource)
        self.procedure_list.append(procedure_data)


class PatientData:
    """Represents a patient and their associated conditions."""

    def __init__(self, patient_json: Bundle):
        """Initializes patient details and extracts relevant resources from a FHIR Bundle."""
        self.personal_info = None
        self.conditions = ConditionsManager()
        self.labs = ObservationManager()
        self.medications = []
        self.notes = ReportManager()
        self.procedures = ProcedureManager()

        self._extract_resources(patient_json)

    def _extract_resources(self, bundle: Bundle):
        """Extracts resources from a FHIR Bundle and assigns them appropriately."""
        for entry in bundle.entry:
            resource = entry.resource

            match resource:
                case Patient(): 
                    self.personal_info = PersonalInfo(resource) #birthday for l
                case Condition(): # important for LLM decision
                    self.conditions.insert(resource)
                case Observation():
                    self.labs.insert(resource)
                case DiagnosticReport():
                    self.notes.insert(resource)
                case MedicationRequest():
                    self.medications.append(MedicationData(resource))
                case Procedure():
                    self.procedures.insert(resource)

    def get_medications_list(self):
        if not self.medications:
            return ["No medications recorded"]
        return [str(med) for med in self.medications]

    def get_labs_list(self):
        """
        Returns a list of formatted strings, one for each lab result.
        
        Returns:
            list: List of strings, each containing details for one lab result
        """
        labs = self.labs.return_all_labs()
        if not labs:
            return ["No lab results recorded"]
        
        result = []
        print("Processing labs, count:", len(labs))
        
        for i, lab in enumerate(labs):
            try:
                lab_str = str(lab)
                result.append(lab_str)
            except Exception as e:
                print(f"Error converting lab {i} to string: {e}")
                result.append(f"Lab #{i+1}: [Error processing lab data]")
        
        return result

    def get_procedures_list(self):
        """
        Returns a list of formatted strings, one for each procedure.
        
        Returns:
            list: List of strings, each containing details for one procedure
        """
        if not self.procedures.procedure_list:
            return ["No procedures recorded"]
        
        return [str(proc) for proc in self.procedures.procedure_list]
   
    def print_personal_info(self):
        print(self.personal_info)

    def print_conditions(self):
        print(self.conditions)
        