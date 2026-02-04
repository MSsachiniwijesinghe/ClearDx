# 🩺 ClearDx — Acute Cholangitis XAI based CDSS Prototype

ClearDx is an interactive **Proof-of-Concept Clinical Decision Support System (CDSS)** for predicting **acute cholangitis severity** using machine learning and Explainable AI (SHAP).

The prototype allows users to enter a patient profile and receive:

- Severity classification (mild / moderate / severe)  
- ERCP timing recommendation  
- SHAP-based feature contribution explanations  
- Clinician-facing natural language summary  

> ⚠️ Academic prototype only. Not validated for real clinical use.

---

## 🚀 Running the App

### 1. Install Dependencies

```bash
pip install -r requirements.txt
````

### 2. Start Streamlit

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 📌 Key Modules

* `ml/severity_model.py` — Severity prediction model
* `explainability/shap_explainer.py` — SHAP explainability engine
* `interface/clinical_explainer.py` — Clinician explanation generator
* `utils/patient.py` — Patient case structure

---

## 🧪 Prototype Outputs

* Predicted severity + probability breakdown
* Triggered diagnostic rule indicators
* Ranked SHAP feature contributions
* Generated clinician-friendly explanation text

---

## Disclaimer

This system is intended for **research demonstration only** and must not be used for real clinical decision-making.



