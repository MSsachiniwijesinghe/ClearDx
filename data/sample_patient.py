from utils.patient import PatientCase

sample_patient = PatientCase(
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
    imaging_report="Dilated common bile duct with suspected obstruction",
    ac_diagnosis=1,
    severity="severe"
)

print(sample_patient)
