from utils.patient import PatientCase
from rules.tg18_rules import TG18RuleEngine

# Sample patient
patient = PatientCase(
    patient_id="AC_001",
    age=65,
    sex="M",
    bilirubin=4.2,
    crp=120.0,
    wbc=15000,
    alt=180,
    ast=160,
    fever=True,
    abdominal_pain=True,
    jaundice=True,
    imaging_report="Dilated common bile duct with suspected obstruction"
)

engine = TG18RuleEngine()
result = engine.diagnose(patient)

print("Diagnosis:", "AC" if result["diagnosis"] == 1 else "No AC")
print("Rule Trace:", result["rule_trace"])
print("Explanation:\n", result["explanation"])
