import numpy as np
from typing import List, Dict, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from utils.patient import PatientCase


class SeverityModel:
    def __init__(self) -> None:
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                solver="lbfgs",
                max_iter=500
            ))
        ])
        self.is_trained = False

    def _extract_features(self, patient: PatientCase) -> np.ndarray:
        return np.array([
            patient.bilirubin,
            patient.crp,
            patient.wbc,
            patient.alt,
            patient.ast,
            int(patient.fever),
            int(patient.abdominal_pain),
            int(patient.jaundice)
        ])

    def prepare_training_data(self, patients: List[PatientCase]) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []

        for p in patients:
            if p.severity is None:
                continue

            X.append(self._extract_features(p))

            if p.severity == "mild":
                y.append(0)
            elif p.severity == "moderate":
                y.append(1)
            elif p.severity == "severe":
                y.append(2)

        return np.array(X), np.array(y)

    def train(self, patients: List[PatientCase]):
        X, y = self.prepare_training_data(patients)
        self.pipeline.fit(X, y)
        self.is_trained = True

    def predict(self, patient: PatientCase) -> Dict:
        if not self.is_trained:
            raise RuntimeError("Severity model not trained")

        X = self._extract_features(patient).reshape(1, -1)

        severity_class = int(self.pipeline.predict(X)[0])
        probabilities = self.pipeline.predict_proba(X)[0]

        return {
            "severity_class": severity_class,
            "severity_label": self._severity_label(severity_class),
            "probabilities": probabilities,
            "ercp_timing": self._ercp_recommendation(severity_class)
        }

    def _severity_label(self, severity_class: int) -> str:
        return ["mild", "moderate", "severe"][severity_class]

    def _ercp_recommendation(self, severity_class: int) -> str:
        if severity_class == 2:
            return "Urgent ERCP (<24h)"
        elif severity_class == 1:
            return "Early ERCP (24–72h)"
        else:
            return "Delayed / Conservative management"
