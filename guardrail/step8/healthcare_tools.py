from langchain_core.tools import tool

@tool
def search_symptoms(symptoms: str) -> str:
    """Search for information about medical symptoms."""
    return (
        f"Symptom information for: {symptoms}. "
        "Please consult a doctor for diagnosis."
    )

@tool
def book_appointment(patient_name: str, date: str, doctor: str) -> str:
    """Book a medical appointment."""
    return (
        f"Appointment booked for {patient_name} "
        f"with Dr. {doctor} on {date}"
    )

@tool
def get_medication_info(medication: str) -> str:
    """Get information about a medication."""
    return (
        f"General info about {medication}. "
        "Always follow your doctor's prescription."
    )
