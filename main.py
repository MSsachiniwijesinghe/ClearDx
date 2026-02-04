# main.py
# Entry point for Acute Cholangitis CDSS PoC

import importlib # Add this line
from utils.patient import PatientCase
from ml.severity_model import SeverityModel
from explainability.shap_explainer import SeveritySHAPExplainer
from interface.clinical_explainer import ClinicalExplanationGenerator
import numpy as np

# Force reload of utils.patient and interface.clinical_explainer
# This is a workaround for potential module caching issues when files are modified with %%writefile
import utils.patient
importlib.reload(utils.patient)
# Re-import specific classes if needed, though usually the module reload is sufficient for class definitions
from utils.patient import PatientCase

import interface.clinical_explainer
importlib.reload(interface.clinical_explainer)
from interface.clinical_explainer import ClinicalExplanationGenerator


# -------------------------------------------------
# 1. Create synthetic training patients (PoC only)
# -------------------------------------------------
training_patients = [
    PatientCase(
        patient_id="P1",
        age=70,
        sex="M",
        bilirubin=5.0,
        crp=160,
        wbc=18000,
        alt=220,
        ast=210,
        fever=True,
        abdominal_pain=True,
        jaundice=True,
        imaging_report="CBD obstruction",
        severity="severe"
    ),
    PatientCase(
        patient_id="P2",
        age=60,
        sex="F",
        bilirubin=2.5,
        crp=60,
        wbc=12000,
        alt=140,
        ast=130,
        fever=True,
        abdominal_pain=True,
        jaundice=False,
        imaging_report="Dilated bile duct",
        severity="moderate"
    ),
    PatientCase(
        patient_id="P3",
        age=55,
        sex="M",
        bilirubin=1.4,
        crp=20,
        wbc=9000,
        alt=80,
        ast=75,
        fever=False,
        abdominal_pain=False,
        jaundice=False,
        imaging_report="Normal biliary tree",
        severity="mild"
    ),
]

# -------------------------------------------------
# 2. Train ML severity model
# -------------------------------------------------
severity_model = SeverityModel()
severity_model.train(training_patients)

# -------------------------------------------------
# 3. Prepare background data for SHAP
# -------------------------------------------------
X_background, _ = severity_model.prepare_training_data(training_patients)

# -------------------------------------------------
# 4. Patch SHAP explainer to safely handle multi-class output
# -------------------------------------------------
original_explain_patient = SeveritySHAPExplainer.explain_patient

def patched_explain_patient(self, patient: PatientCase) -> dict:
    x = self.model._extract_features(patient).reshape(1, -1)
    shap_values = self.explainer.shap_values(x)

    severity_result = self.model.predict(patient)
    predicted_class = severity_result["severity_class"]

    # Handle different SHAP output formats safely
    if isinstance(shap_values, list):
        if len(shap_values) > predicted_class:
            class_shap = shap_values[predicted_class][0]
        else:
            class_shap = shap_values[0][0]
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            class_shap = shap_values[0, :, predicted_class]
        else:
            class_shap = shap_values[0]
    else:
        raise TypeError("Unexpected SHAP output format")

    return {
        "predicted_severity": severity_result["severity_label"],
        "ercp_timing": severity_result["ercp_timing"],
        "feature_contributions": self._format_contributions(class_shap)
    }

SeveritySHAPExplainer.explain_patient = patched_explain_patient

# -------------------------------------------------
# 5. Initialize SHAP explainer
# -------------------------------------------------
shap_explainer = SeveritySHAPExplainer(
    severity_model=severity_model,
    background_data=X_background
)

# -------------------------------------------------
# 6. Run prediction + explanation for one patient
# -------------------------------------------------
patient = training_patients[0]

severity_result = severity_model.predict(patient)
shap_explanation = shap_explainer.explain_patient(patient)

# -------------------------------------------------
# 7. Generate clinician-facing explanation
# -------------------------------------------------
clinical_explainer = ClinicalExplanationGenerator()

diagnosis_result = {
    "diagnosis": "acute_cholangitis",
    "triggered_rules": [
        "Charcot triad satisfied",
        "Evidence of biliary obstruction"
    ]
}

clinical_text = clinical_explainer.generate(
    patient=patient,
    diagnosis_result=diagnosis_result,
    severity_result=severity_result,
    shap_explanation=shap_explanation
)

# -------------------------------------------------
# 8. Output
# -------------------------------------------------
print("\n================ CLINICAL DECISION SUPPORT OUTPUT ================\n")
print(clinical_text)
print("\n=================================================================\n")
