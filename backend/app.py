import matplotlib
matplotlib.use("Agg")   # 🔥 CRITICAL FIX

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from xgboost import XGBClassifier
import io
import os
import shap
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np

app = FastAPI()

# ---------------------------------------
# ENABLE CORS
# ---------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------
# LOAD MODEL & ARTIFACTS
# ---------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = XGBClassifier()
model.load_model(os.path.join(BASE_DIR, "xgboost_disease_model.json"))

label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))

disease_map = {
    0: "SCD",
    1: "CF",
    2: "HD"
}

# ---------------------------------------
# HEALTH CHECK
# ---------------------------------------
@app.get("/")
def home():
    return {"message": "FastAPI backend is running"}

# ---------------------------------------
# PREDICT ROUTE
# ---------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Ensure correct column order
        df = df[feature_columns]

        # Predict labels
        predictions = model.predict(df)

        # Predict probabilities
        probabilities = model.predict_proba(df)

        results = []

        for i, pred in enumerate(predictions):
            predicted_label = disease_map[int(pred)]
            confidence = float(max(probabilities[i])) * 100

            results.append({
                "row": i,
                "predicted_disease": predicted_label,
                "confidence": round(confidence, 2)
            })

        return {"predictions": results}

    except Exception as e:
        return {"error": str(e)}
    # explain route for SHAP visualizations
@app.get("/explain")
def explain():
    try:
        df = pd.read_csv("rf_training_dataset.csv")
        X = df[feature_columns]

        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        # 🔥 IMPORTANT FIX FOR MULTI-CLASS
        # We take mean absolute SHAP across classes
        shap_values_mean = np.abs(shap_values.values).mean(axis=2)

        # -----------------------------
        # 1️⃣ Global Bar Plot
        # -----------------------------
        plt.figure()
        shap.summary_plot(
            shap_values_mean,
            X,
            plot_type="bar",
            show=False
        )

        buf1 = BytesIO()
        plt.savefig(buf1, format="png", bbox_inches="tight")
        plt.close()
        buf1.seek(0)

        global_plot = base64.b64encode(buf1.read()).decode("utf-8")

        # -----------------------------
        # 2️⃣ Summary Beeswarm
        # -----------------------------
        plt.figure()
        shap.summary_plot(
            shap_values_mean,
            X,
            show=False
        )

        buf2 = BytesIO()
        plt.savefig(buf2, format="png", bbox_inches="tight")
        plt.close()
        buf2.seek(0)

        summary_plot = base64.b64encode(buf2.read()).decode("utf-8")

        return {
            "global_plot": global_plot,
            "summary_plot": summary_plot
        }

    except Exception as e:
        return {"error": str(e)}
   
   
@app.post("/explain-local")
async def explain_local(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df = df[feature_columns]

        explainer = shap.Explainer(model, df)
        shap_values = explainer(df)

        # Use first row
        sample_index = 0

        # Get predicted class
        prediction = model.predict(df)[sample_index]
        class_index = int(prediction)

        # Extract correct SHAP values for multi-class
        shap_vals = shap_values.values[sample_index][:, class_index]

        base_value = shap_values.base_values[sample_index][class_index]

        # Create waterfall plot manually
        plt.figure(figsize=(8,6))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_vals,
                base_values=base_value,
                data=df.iloc[sample_index],
                feature_names=feature_columns
            ),
            show=False
        )

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)

        force_plot = base64.b64encode(buf.read()).decode("utf-8")

        return {"force_plot": force_plot}

    except Exception as e:
        print("LOCAL SHAP ERROR:", str(e))   # 👈 very important
        return {"error": str(e)}
    
    # model comparison route
@app.get("/model-comparison")
def model_comparison():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        import base64
        from io import BytesIO

        # Load dataset
        df = pd.read_csv("rf_training_dataset.csv")
        X = df[feature_columns]
        y = label_encoder.transform(df["disease"])

        # ===== XGBOOST =====
        xgb_preds = model.predict(X)
        xgb_cm = confusion_matrix(y, xgb_preds)
        xgb_acc = (xgb_preds == y).mean()

        # Plot XGB CM
        plt.figure()
        sns.heatmap(xgb_cm, annot=True, fmt="d", cmap="Blues")
        plt.title("XGBoost Confusion Matrix")
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        xgb_img = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        # ===== RANDOM FOREST =====
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(random_state=42)
        rf.fit(X, y)
        rf_preds = rf.predict(X)
        rf_cm = confusion_matrix(y, rf_preds)
        rf_acc = (rf_preds == y).mean()

        plt.figure()
        sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Greens")
        plt.title("Random Forest Confusion Matrix")
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        rf_img = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        return {
            "rf_accuracy": round(rf_acc * 100, 2),
            "xgb_accuracy": round(xgb_acc * 100, 2),
            "rf_confusion_matrix": rf_img,
            "xgb_confusion_matrix": xgb_img
        }

    except Exception as e:
        return {"error": str(e)}    