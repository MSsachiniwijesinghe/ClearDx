from dataclasses import dataclass
from typing import Optional


@dataclass
class PatientCase:
    # Metadata
    patient_id: str
    age: int
    sex: str  # 'M' or 'F'

    # Laboratory values
    bilirubin: float
    crp: float
    wbc: float
    alt: float
    ast: float

    # Clinical symptoms
    fever: bool
    abdominal_pain: bool
    jaundice: bool

    # Imaging report (text-based)
    imaging_report: str

    # Labels (optional, used for evaluation)
    ac_diagnosis: Optional[int] = None  # 1 = AC, 0 = No AC
    severity: Optional[str] = None      # 'mild', 'moderate', 'severe'