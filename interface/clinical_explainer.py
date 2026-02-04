from typing import Dict
from utils.patient import PatientCase


class ClinicalExplanationGenerator:
    """
    Generates clinician-facing explanations for CDSS outputs
    """

    def generate(
        self,
        patient: PatientCase,
        diagnosis_result: Dict,
        severity_result: Dict,
        shap_explanation: Dict
    ) -> str:

        explanation = []

        # 1. Clinical summary
        explanation.append(self._clinical_summary(patient))

        # 2. Diagnosis reasoning
        explanation.append(self._diagnosis_reasoning(diagnosis_result))

        # 3. Severity reasoning (SHAP-based)
        explanation.append(self._severity_reasoning(shap_explanation))

        # 4. ERCP timing justification
        explanation.append(self._ercp_reasoning(severity_result))

        # 5. Uncertainty and override note
        explanation.append(self._override_note())

        return "\n\n".join(explanation)

    # -------------------------------------------------
    # 1. Clinical summary
    # -------------------------------------------------
    def _clinical_summary(self, patient: PatientCase) -> str:
        sex_label = "male" if patient.sex == "M" else "female"

        return (
            "Patient Summary:\n"
            f"A {patient.age}-year-old {sex_label} presenting with "
            f"{'fever' if patient.fever else 'no fever'}, "
            f"{'abdominal pain' if patient.abdominal_pain else 'no abdominal pain'}, "
            f"and {'jaundice' if patient.jaundice else 'no jaundice'}. "
            f"Laboratory findings include bilirubin {patient.bilirubin} mg/dL, "
            f"CRP {patient.crp} mg/L, and WBC {patient.wbc}/µL. "
            f"Imaging report notes: {patient.imaging_report}."
        )

    # -------------------------------------------------
    # 2. Diagnosis reasoning
    # -------------------------------------------------
    def _diagnosis_reasoning(self, diagnosis_result: Dict) -> str:
        if diagnosis_result.get("diagnosis") == "acute_cholangitis":
            rules = ", ".join(diagnosis_result.get("triggered_rules", []))
            return (
                "Diagnostic Reasoning:\n"
                "The patient meets diagnostic criteria for Acute Cholangitis. "
                f"This decision is supported by the following findings: {rules}."
            )
        else:
            return (
                "Diagnostic Reasoning:\n"
                "The patient does not meet sufficient diagnostic criteria for "
                "Acute Cholangitis based on the applied rule set."
            )

    # -------------------------------------------------
    # 3. Severity reasoning (XAI / SHAP layer)
    # -------------------------------------------------
    def _severity_reasoning(self, shap_explanation: Dict) -> str:
        top_factors = shap_explanation.get("feature_contributions", [])[:3]

        factor_text = []
        for f in top_factors:
            direction = (
                "contributing to increased severity"
                if f.get("direction") == "increases"
                else "reducing estimated severity"
            )
            factor_text.append(f"{f.get('feature')} ({direction})")

        factors = ", ".join(factor_text) if factor_text else "multiple clinical factors"

        return (
            "Severity Assessment Reasoning:\n"
            f"The predicted severity is **{shap_explanation.get('predicted_severity')}**. "
            f"This assessment is primarily influenced by {factors}."
        )

    # -------------------------------------------------
    # 4. ERCP timing recommendation
    # -------------------------------------------------
    def _ercp_reasoning(self, severity_result: Dict) -> str:
        return (
            "Management Recommendation:\n"
            f"Based on the estimated severity, the system recommends "
            f"{severity_result.get('ercp_timing')}. "
            "This aligns with guideline-based evidence suggesting earlier biliary "
            "intervention in higher severity cases."
        )

    # -------------------------------------------------
    # 5. Clinical override note
    # -------------------------------------------------
    def _override_note(self) -> str:
        return (
            "Clinical Judgment Note:\n"
            "This system is intended to support, not replace, clinical judgment. "
            "Clinicians may override the recommendation, and such decisions should "
            "be documented with appropriate clinical justification."
        )
