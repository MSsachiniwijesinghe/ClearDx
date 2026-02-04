from utils.patient import PatientCase
from ml.severity_model import SeverityModel

# Synthetic training cases
patients = [
    PatientCase("P1", 60, "M", 1.8, 20, 9000, 80, 75, False, False, False,
                "Normal biliary tree", severity="mild"),
    PatientCase("P2", 65, "F", 3.2, 80, 12000, 150, 140, True, True, True,
                "Dilated bile duct", severity="moderate"),
    PatientCase("P3", 70, "M", 5.0, 160, 18000, 220, 210, True, True, True,
                "Obstruction in CBD", severity="severe"),
]

model = SeverityModel()
model.train(patients)

# Test prediction
test_patient = patients[2]
result = model.predict(test_patient)

print("Severity:", result["severity_label"])
print("Probabilities:", result["probabilities"])
print("ERCP timing:", result["ercp_timing"])
