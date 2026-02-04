import sys
import os
import importlib
import numpy as np
import streamlit as st

# Optional (nice tables + charts). If you don't have pandas, install it or remove pandas usage.
import pandas as pd

# -----------------------------
# Make sure root is on sys.path
# -----------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# -----------------------------
# Imports from your project
# -----------------------------
import utils.patient
import interface.clinical_explainer

importlib.reload(utils.patient)
importlib.reload(interface.clinical_explainer)

from utils.patient import PatientCase
from ml.severity_model import SeverityModel
from explainability.shap_explainer import SeveritySHAPExplainer
from interface.clinical_explainer import ClinicalExplanationGenerator


# =========================================================
# Page config + small styling
# =========================================================
st.set_page_config(
    page_title="Acute Cholangitis CDSS (PoC)",
    page_icon="🩺",
    layout="wide"
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      .big-title { font-size: 1.6rem; font-weight: 800; margin-bottom: 0.2rem; }
      .subtle { color: rgba(255,255,255,0.72); margin-top: 0; }
      .card {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        padding: 14px 16px;
        background: rgba(255,255,255,0.03);
      }
      .muted { color: rgba(255,255,255,0.68); }
      .pill {
        display:inline-block; padding: 3px 10px; border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.18);
        background: rgba(255,255,255,0.05);
        font-size: 0.85rem;
        margin-right: 6px;
      }
      .hr { height: 1px; background: rgba(255,255,255,0.08); margin: 10px 0 14px 0; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="big-title">🩺 ClearDx(PoC)</div>', unsafe_allow_html=True)
st.markdown('<p class="subtle">Enter a patient profile → get severity prediction, ERCP timing suggestion, SHAP-based feature contributions, and a clinician-facing explanation.</p>', unsafe_allow_html=True)


# =========================================================
# Training data (your synthetic PoC cases)
# =========================================================
def get_training_patients():
    return [
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


# =========================================================
# Patch SHAP explainer (your safe multi-class patch)
# =========================================================
def patch_shap_explainer():
    original_explain_patient = SeveritySHAPExplainer.explain_patient

    def patched_explain_patient(self, patient: PatientCase) -> dict:
        x = self.model._extract_features(patient).reshape(1, -1)
        shap_values = self.explainer.shap_values(x)

        severity_result = self.model.predict(patient)
        predicted_class = severity_result.get("severity_class", 0)

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
            "predicted_severity": severity_result.get("severity_label", "unknown"),
            "ercp_timing": severity_result.get("ercp_timing", "unknown"),
            "feature_contributions": self._format_contributions(class_shap)
        }

    SeveritySHAPExplainer.explain_patient = patched_explain_patient
    return original_explain_patient


# =========================================================
# Cached model + explainer init
# =========================================================
@st.cache_resource(show_spinner=False)
def init_pipeline():
    patch_shap_explainer()

    training_patients = get_training_patients()

    severity_model = SeverityModel()
    severity_model.train(training_patients)

    X_background, _ = severity_model.prepare_training_data(training_patients)

    shap_explainer = SeveritySHAPExplainer(
        severity_model=severity_model,
        background_data=X_background
    )

    clinical_explainer = ClinicalExplanationGenerator()

    return training_patients, severity_model, shap_explainer, clinical_explainer


training_patients, severity_model, shap_explainer, clinical_explainer = init_pipeline()


# =========================================================
# Simple diagnosis rules for PoC (to feed your text generator)
# =========================================================
def build_diagnosis_result(patient: PatientCase) -> dict:
    triggered = []

    # Charcot triad (very simplified)
    if patient.fever and patient.abdominal_pain and patient.jaundice:
        triggered.append("Charcot triad satisfied")

    # Imaging evidence heuristic
    img = (patient.imaging_report or "").lower()
    if any(k in img for k in ["obstruction", "cbd", "dilated", "dilat", "stone", "stricture"]):
        triggered.append("Evidence of biliary obstruction")

    # Inflammation markers heuristic
    if patient.crp >= 100 or patient.wbc >= 15000:
        triggered.append("Marked inflammatory response")

    # If nothing triggered, still keep it explicit
    if not triggered:
        triggered.append("No key PoC rule triggers detected")

    return {"diagnosis": "acute_cholangitis", "triggered_rules": triggered}


# =========================================================
# Sidebar UI
# =========================================================
st.sidebar.header("🧾 Patient Input")

mode = st.sidebar.radio("Input mode", ["Use demo patient", "Enter manually"], horizontal=False)

if mode == "Use demo patient":
    demo_id = st.sidebar.selectbox("Select demo case", [p.patient_id for p in training_patients], index=0)
    base_patient = next(p for p in training_patients if p.patient_id == demo_id)

    # Show a quick summary in sidebar
    st.sidebar.markdown("**Demo summary**")
    st.sidebar.caption(f"Age: {base_patient.age} | Sex: {base_patient.sex}")
    st.sidebar.caption(f"Bilirubin: {base_patient.bilirubin} | CRP: {base_patient.crp} | WBC: {base_patient.wbc}")
    st.sidebar.caption(f"ALT: {base_patient.alt} | AST: {base_patient.ast}")
    st.sidebar.caption(f"Fever: {base_patient.fever} | Pain: {base_patient.abdominal_pain} | Jaundice: {base_patient.jaundice}")
    st.sidebar.caption(f"Imaging: {base_patient.imaging_report}")

    patient = base_patient

else:
    patient_id = st.sidebar.text_input("Patient ID", value="NEW-1")

    age = st.sidebar.slider("Age", 1, 100, 60)
    sex = st.sidebar.selectbox("Sex", ["M", "F"], index=0)

    st.sidebar.markdown("**Labs**")
    bilirubin = st.sidebar.number_input("Bilirubin", min_value=0.0, max_value=50.0, value=2.5, step=0.1)
    crp = st.sidebar.number_input("CRP", min_value=0, max_value=500, value=60, step=5)
    wbc = st.sidebar.number_input("WBC", min_value=0, max_value=50000, value=12000, step=500)
    alt = st.sidebar.number_input("ALT", min_value=0, max_value=2000, value=140, step=10)
    ast = st.sidebar.number_input("AST", min_value=0, max_value=2000, value=130, step=10)

    st.sidebar.markdown("**Symptoms**")
    fever = st.sidebar.checkbox("Fever", value=True)
    abdominal_pain = st.sidebar.checkbox("Abdominal pain", value=True)
    jaundice = st.sidebar.checkbox("Jaundice", value=False)

    imaging_report = st.sidebar.text_area("Imaging report", value="Dilated bile duct", height=80)

    patient = PatientCase(
        patient_id=patient_id,
        age=age,
        sex=sex,
        bilirubin=float(bilirubin),
        crp=int(crp),
        wbc=int(wbc),
        alt=int(alt),
        ast=int(ast),
        fever=bool(fever),
        abdominal_pain=bool(abdominal_pain),
        jaundice=bool(jaundice),
        imaging_report=imaging_report,
        severity=None  # unknown at inference
    )

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🔎 Run CDSS", type="primary", use_container_width=True)


# =========================================================
# Run inference
# =========================================================
def safe_get_prob_table(severity_result: dict):
    probs = severity_result.get("probabilities")
    if probs is None:
        return None

    # If it's already a dict label->prob
    if isinstance(probs, dict):
        rows = [{"Severity": k, "Probability": float(v)} for k, v in probs.items()]
        return pd.DataFrame(rows).sort_values("Probability", ascending=False)

    # If it's a list/np array, try to map to common classes
    if isinstance(probs, (list, tuple, np.ndarray)):
        probs = list(probs)
        labels = ["mild", "moderate", "severe"][: len(probs)]
        rows = [{"Severity": labels[i], "Probability": float(probs[i])} for i in range(len(probs))]
        return pd.DataFrame(rows).sort_values("Probability", ascending=False)

    return None


def normalize_contributions(feature_contributions):
    """
    Tries to convert whatever _format_contributions returns into a DataFrame with:
    Feature | SHAP | Direction
    """
    if feature_contributions is None:
        return pd.DataFrame(columns=["Feature", "SHAP", "Direction"])

    # common patterns:
    # - list[dict(feature=?, contribution=?)]
    # - list[dict(feature=?, shap_value=?)]
    # - list[tuple(feature, value)]
    rows = []

    if isinstance(feature_contributions, list):
        for item in feature_contributions:
            if isinstance(item, dict):
                feat = item.get("feature") or item.get("name") or item.get("Feature") or "unknown"
                val = item.get("contribution")
                if val is None:
                    val = item.get("shap_value")
                if val is None:
                    val = item.get("value")
                direction = item.get("direction")
                if direction is None and val is not None:
                    direction = "↑ increases" if float(val) >= 0 else "↓ decreases"
                rows.append({"Feature": str(feat), "SHAP": float(val) if val is not None else 0.0, "Direction": str(direction or "")})
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                feat, val = item[0], item[1]
                direction = "↑ increases" if float(val) >= 0 else "↓ decreases"
                rows.append({"Feature": str(feat), "SHAP": float(val), "Direction": direction})

    elif isinstance(feature_contributions, dict):
        # dict feature->value
        for feat, val in feature_contributions.items():
            direction = "↑ increases" if float(val) >= 0 else "↓ decreases"
            rows.append({"Feature": str(feat), "SHAP": float(val), "Direction": direction})

    df = pd.DataFrame(rows)
    if not df.empty:
        df["AbsSHAP"] = df["SHAP"].abs()
        df = df.sort_values("AbsSHAP", ascending=False).drop(columns=["AbsSHAP"])
    return df


# Auto-run once on first load for a nice UX
if "has_run_once" not in st.session_state:
    st.session_state.has_run_once = True
    run_btn = True


if run_btn:
    with st.spinner("Running severity prediction + SHAP explanation..."):
        severity_result = severity_model.predict(patient)
        shap_explanation = shap_explainer.explain_patient(patient)
        diagnosis_result = build_diagnosis_result(patient)

        clinical_text = clinical_explainer.generate(
            patient=patient,
            diagnosis_result=diagnosis_result,
            severity_result=severity_result,
            shap_explanation=shap_explanation
        )

    # =========================================================
    # Main layout
    # =========================================================
    left, right = st.columns([1.15, 0.85], gap="large")

    # ----------------------------
    # LEFT: Results + Explanation
    # ----------------------------
    with left:
        st.markdown("### Results")

        sev_label = severity_result.get("severity_label", "unknown")
        ercp_timing = severity_result.get("ercp_timing", "unknown")

        # Nice “card” header
        st.markdown(
            f"""
            <div class="card">
              <div style="display:flex; justify-content:space-between; align-items:center; gap:12px;">
                <div>
                  <div class="muted">Patient</div>
                  <div style="font-size:1.15rem; font-weight:800;">{patient.patient_id}</div>
                </div>
                <div style="text-align:right;">
                  <span class="pill">Age: {patient.age}</span>
                  <span class="pill">Sex: {patient.sex}</span>
                </div>
              </div>
              <div class="hr"></div>
              <div style="display:flex; gap:18px; flex-wrap:wrap;">
                <span class="pill">Severity: <b>{sev_label}</b></span>
                <span class="pill">ERCP timing: <b>{ercp_timing}</b></span>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted severity", sev_label)
        m2.metric("ERCP timing", ercp_timing)
        m3.metric("Triggered rules", str(len(diagnosis_result.get("triggered_rules", []))))

        st.markdown("#### Triggered PoC rules")
        st.write(diagnosis_result.get("triggered_rules", []))

        st.markdown("#### Probability breakdown")
        prob_df = safe_get_prob_table(severity_result)
        if prob_df is not None and not prob_df.empty:
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
        else:
            st.info("No probability vector returned by the model (or unsupported format).")

        st.markdown("#### Clinician-facing explanation")
        st.text_area("Generated explanation", value=clinical_text, height=240)

    # ----------------------------
    # RIGHT: SHAP + Patient summary
    # ----------------------------
    with right:
        st.markdown("### Explainability (SHAP)")

        contrib_df = normalize_contributions(shap_explanation.get("feature_contributions"))
        if contrib_df is None or contrib_df.empty:
            st.warning("No SHAP contributions available to display.")
        else:
            st.caption("Top contributing features for the predicted class (sorted by absolute SHAP).")
            st.dataframe(contrib_df.head(12), use_container_width=True, hide_index=True)

            # Bar chart (top 10) using Streamlit's built-in charting
            top = contrib_df.head(10).copy()
            top = top.set_index("Feature")[["SHAP"]]
            st.bar_chart(top)

        st.markdown("### Patient summary")
        st.markdown(
            f"""
            <div class="card">
              <div class="muted" style="font-weight:700;">Labs</div>
              <div class="hr"></div>
              <div class="pill">Bilirubin: <b>{patient.bilirubin}</b></div>
              <div class="pill">CRP: <b>{patient.crp}</b></div>
              <div class="pill">WBC: <b>{patient.wbc}</b></div>
              <div class="pill">ALT: <b>{patient.alt}</b></div>
              <div class="pill">AST: <b>{patient.ast}</b></div>
              <div class="hr"></div>
              <div class="muted" style="font-weight:700;">Symptoms</div>
              <div class="hr"></div>
              <div class="pill">Fever: <b>{patient.fever}</b></div>
              <div class="pill">Pain: <b>{patient.abdominal_pain}</b></div>
              <div class="pill">Jaundice: <b>{patient.jaundice}</b></div>
              <div class="hr"></div>
              <div class="muted" style="font-weight:700;">Imaging</div>
              <div class="hr"></div>
              <div>{patient.imaging_report}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("Debug (raw outputs)"):
            st.write("severity_result:", severity_result)
            st.write("shap_explanation:", shap_explanation)

else:
    st.info("Use the sidebar to select/enter a patient, then click **Run CDSS**.")


# =========================================================
# Footer
# =========================================================
st.markdown("---")
st.caption("PoC only. Not validated for clinical use. Always follow local clinical guidelines and senior clinician judgement.")
