import shap
import numpy as np
from typing import Dict, List

from ml.severity_model import SeverityModel
from utils.patient import PatientCase


FEATURE_NAMES = [
    "bilirubin",
    "crp",
    "wbc",
    "alt",
    "ast",
    "fever",
    "abdominal_pain",
    "jaundice"
]


class SeveritySHAPExplainer:
    """
    SHAP-based explainability for severity ML model
    """

    def __init__(self, severity_model: SeverityModel, background_data: np.ndarray):
        """
        background_data: feature matrix used for SHAP baseline
        """
        self.model = severity_model
        self.background_data = background_data

        self.explainer = shap.LinearExplainer(
            self.model.pipeline.named_steps["model"],
            self.background_data,
            feature_names=FEATURE_NAMES
        )

    def explain_patient(self, patient: PatientCase) -> Dict:
        """
        Local SHAP explanation for one patient
        """
        x = self.model._extract_features(patient).reshape(1, -1)

        shap_values = self.explainer.shap_values(x)

        severity_result = self.model.predict(patient)
        predicted_class = severity_result["severity_class"]

        class_shap = shap_values[predicted_class][0]

        explanation = {
            "predicted_severity": severity_result["severity_label"],
            "ercp_timing": severity_result["ercp_timing"],
            "feature_contributions": self._format_contributions(class_shap)
        }

        return explanation

    def global_feature_importance(self, X: np.ndarray) -> Dict:
        """
        Global SHAP importance across dataset
        """
        shap_values = self.explainer.shap_values(X)

        mean_abs_shap = np.mean(
            np.abs(shap_values), axis=(0, 1)
        )

        return dict(zip(FEATURE_NAMES, mean_abs_shap))

    def _format_contributions(self, shap_values: np.ndarray) -> List[Dict]:
        """
        Convert SHAP vector to readable explanation
        """
        contributions = []

        for feature, value in zip(FEATURE_NAMES, shap_values):
            contributions.append({
                "feature": feature,
                "shap_value": float(value),
                "direction": "increases severity" if value > 0 else "decreases severity"
            })

        return sorted(
            contributions,
            key=lambda x: abs(x["shap_value"]),
            reverse=True
        )
