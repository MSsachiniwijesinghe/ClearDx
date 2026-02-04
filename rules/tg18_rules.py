from typing import Dict, List
from utils.patient import PatientCase


class TG18RuleEngine:
    """
    Simplified TG18 rule-based diagnostic engine (PoC level)
    """

    def __init__(self):
        # Thresholds (PoC assumptions)
        self.CRP_THRESHOLD = 10.0
        self.WBC_THRESHOLD = 11000
        self.BILIRUBIN_THRESHOLD = 2.0

    def check_systemic_inflammation(self, patient: PatientCase) -> bool:
        return (
            patient.fever or
            patient.crp >= self.CRP_THRESHOLD or
            patient.wbc >= self.WBC_THRESHOLD
        )

    def check_cholestasis(self, patient: PatientCase) -> bool:
        return (
            patient.bilirubin >= self.BILIRUBIN_THRESHOLD or
            patient.jaundice
        )

    def check_imaging(self, patient: PatientCase) -> bool:
        keywords = ["dilated", "obstruction", "bile duct", "cholestasis"]
        report = patient.imaging_report.lower()
        return any(keyword in report for keyword in keywords)

    def diagnose(self, patient: PatientCase) -> Dict:
        """
        Apply TG18 rules and return diagnosis with explanation
        """

        rule_trace: List[str] = []

        inflammation = self.check_systemic_inflammation(patient)
        cholestasis = self.check_cholestasis(patient)
        imaging = self.check_imaging(patient)

        if inflammation:
            rule_trace.append("Systemic inflammation criteria satisfied")
        if cholestasis:
            rule_trace.append("Cholestasis criteria satisfied")
        if imaging:
            rule_trace.append("Imaging evidence of biliary obstruction present")

        diagnosis = int(inflammation and cholestasis and imaging)

        explanation = self.generate_explanation(diagnosis, rule_trace)

        return {
            "diagnosis": diagnosis,
            "rule_trace": rule_trace,
            "explanation": explanation
        }

    def generate_explanation(self, diagnosis: int, rule_trace: List[str]) -> str:
        if diagnosis == 1:
            explanation = (
                "Diagnosis: Acute Cholangitis.\n"
                "Reasoning:\n"
            )
        else:
            explanation = (
                "Diagnosis: Acute Cholangitis not confirmed.\n"
                "Reasoning:\n"
            )

        for rule in rule_trace:
            explanation += f"- {rule}\n"

        if not rule_trace:
            explanation += "- Insufficient TG18 criteria satisfied\n"

        return explanation
