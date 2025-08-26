import io
import json
import textwrap
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
import streamlit as st
import json
from io import BytesIO
from PIL import Image
import os
import random
import joblib
import pandas as pd
import joblib
import os
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer  # ✅ Add this line
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import sklearn
try:
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression

    SKLEARN_AVAILABLE = True
except:
    SKLEARN_AVAILABLE = False

# ---------------------------
# Styling
# ---------------------------
st.set_page_config(
    page_title="Portfolio & ML Demo",
    page_icon="🧭",
    layout="wide",
    menu_items={
        "Get help": "https://docs.streamlit.io",
        "Report a bug": "https://github.com/streamlit/streamlit/issues",
        "About": "Demo app with portfolio, models, and predictions"
    },
)

PRIMARY = "#2563eb"

st.markdown(
    f"""
    <style>
      .big-title {{ font-size: 2rem; font-weight: 800; margin: 0; }}
      .pill {{ display:inline-block; padding: 2px 10px; border-radius:999px; background:{PRIMARY}15; color:{PRIMARY}; font-weight:600; font-size:.85rem; }}
      .card {{ border:1px solid #e5e7eb; border-radius:16px; padding:16px; background:white; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------
# Helper functions
# ---------------------------
def load_pickle(upload):
    if upload is None or not SKLEARN_AVAILABLE:
        return None
    try:
        return joblib.load(upload)
    except:
        try:
            upload.seek(0)
            return joblib.load(io.BytesIO(upload.read()))
        except:
            return None


def safe_predict(model, df: pd.DataFrame):
    if model is None:
        return None
    try:
        preds = model.predict(df)
        return preds
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


# ---------------------------
# Demo models
# ---------------------------
@st.cache_resource
def build_demo_salary_pipe():
    if not SKLEARN_AVAILABLE:
        return None
    rng = np.random.default_rng(42)
    n = 600
    df = pd.DataFrame({
        "years_experience": rng.integers(0, 16, size=n),
        "education": rng.choice(["High School", "Bachelor", "Master", "PhD"], size=n, p=[.2, .45, .3, .05]),
        "job_title": rng.choice(["Data Scientist", "ML Engineer", "Software Dev", "Analyst"], size=n),
        "location": rng.choice(["Bangalore", "Delhi", "Mumbai", "Remote"], size=n),
        "company_size": rng.choice(["Startup", "SME", "MNC"], size=n, p=[.3, .4, .3]),
    })
    base = 3.5 + 0.25 * df["years_experience"]
    edu_map = {"High School": -0.3, "Bachelor": 0.0, "Master": 0.3, "PhD": 0.6}
    job_map = {"Data Scientist": 0.6, "ML Engineer": 0.5, "Software Dev": 0.35, "Analyst": 0.2}
    loc_map = {"Bangalore": 0.25, "Delhi": 0.15, "Mumbai": 0.3, "Remote": 0.0}
    size_map = {"Startup": -0.05, "SME": 0.0, "MNC": 0.2}
    y = base + df["education"].map(edu_map) + df["job_title"].map(job_map) + df["location"].map(loc_map) + df[
        "company_size"].map(size_map) + rng.normal(0, 0.25, size=n)
    y = np.maximum(2.5, y)
    num = ["years_experience"]
    cat = ["education", "job_title", "location", "company_size"]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    pipe = Pipeline([("pre", pre), ("model", RandomForestRegressor(n_estimators=220, random_state=42))])
    pipe.fit(df[num + cat], y)
    return pipe


@st.cache_resource
def build_demo_water_pipe():
    if not SKLEARN_AVAILABLE:
        return None
    rng = np.random.default_rng(7)
    n = 1200
    df = pd.DataFrame({
        "ph": rng.normal(7.0, 1.2, size=n).clip(0.0, 14.0),
        "Hardness": rng.normal(200, 60, size=n).clip(10, 400),
        "Solids": rng.normal(20000, 9000, size=n).clip(100, 55000),
        "Chloramines": rng.normal(7.0, 2.0, size=n).clip(0.1, 13),
        "Sulfate": rng.normal(330, 120, size=n).clip(10, 600),
        "Conductivity": rng.normal(425, 120, size=n).clip(80, 800),
        "Organic_carbon": rng.normal(10, 5, size=n).clip(1, 30),
        "Trihalomethanes": rng.normal(70, 25, size=n).clip(5, 130),
        "Turbidity": rng.normal(3.0, 1.5, size=n).clip(0.1, 9.0),
    })

    def potable_rule(r):
        score = 0
        score += 1 if 6.5 <= r.ph <= 8.5 else 0
        score += 1 if r.Hardness <= 300 else 0
        score += 1 if r.Solids <= 30000 else 0
        score += 1 if 2 <= r.Chloramines <= 10 else 0
        score += 1 if 150 <= r.Sulfate <= 400 else 0
        score += 1 if 100 <= r.Conductivity <= 700 else 0
        score += 1 if r.Organic_carbon <= 20 else 0
        score += 1 if r.Trihalomethanes <= 100 else 0
        score += 1 if r.Turbidity <= 5 else 0
        return 1 if score >= 6 else 0

    y = df.apply(potable_rule, axis=1)
    num = df.columns.tolist()
    pre = ColumnTransformer([("num", StandardScaler(), num)])
    pipe = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=200))])
    pipe.fit(df, y)
    return pipe


# Load demo models
DEMO_SALARY = build_demo_salary_pipe()
DEMO_WATER = build_demo_water_pipe()

# ---------------------------
# Sidebar Navigation (navbar style)
# ---------------------------
st.sidebar.title("VIRTUAL DEMO")
menu_options = ["Home", "Portfolio", "Salary Prediction", "Water Quality", "Team"]
choice = st.sidebar.selectbox("MODEL VIEW", menu_options)


# ---------------------------
# Main content based on selection
# ---------------------------
if choice == "Home":
    # Home.py — Streamlit Home Page for Two ML Dashboards
    # Place this file at the root of your Streamlit app.
    # If you use Streamlit's multipage structure, also create two pages:
    #   pages/1_💼_Salary_Prediction.py
    #   pages/2_💧_Water_Quality_Prediction.py
    # The buttons below use st.page_link to jump to those pages.

    import streamlit as st
    import pandas as pd
    from datetime import datetime

    st.set_page_config(
        page_title="AI Dashboards Hub",
        page_icon="🧭",
        layout="wide",
    )

    # --------------- Styling ---------------
    st.markdown(
        """
        <style>
          .hero {
            background: linear-gradient(135deg, #0ea5e9 0%, #22c55e 100%);
            color: white; padding: 28px; border-radius: 24px;
            box-shadow: 0 10px 30px rgba(0,0,0,.15);
          }
          .badge {display:inline-block; padding:6px 10px; border-radius:999px;
                  background:#eef2ff; color:#3730a3; font-weight:600; margin-right:6px;}
          .card {background: white; border-radius: 20px; padding: 18px; border:1px solid #eef2f7;
                 box-shadow: 0 2px 12px rgba(16,24,40,.06);}    
          .muted {color:#475467}
          .step {display:flex; gap:12px; margin:10px 0;}
          .stepnum {min-width:28px; height:28px; border-radius:999px; background:#111827; color:white;
                    font-weight:700; display:flex; align-items:center; justify-content:center;}
          .pill {display:inline-block; border:1px solid #e5e7eb; padding:6px 10px; border-radius:999px; margin:4px 6px 0 0;}
          .small {font-size: 13px; color:#6b7280}
          .metricbox {border:1px dashed #e5e7eb; border-radius:16px; padding:16px;}
          .tablewrap {border:1px solid #eef2f7; border-radius:12px; overflow:hidden}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --------------- Hero ---------------
    st.markdown(
        f"""
        <div class="hero">
          <h1 style="margin:0;">🧭 Unified ML Hub</h1>
          <p style="margin-top:6px; font-size:18px; opacity:.95;">
            One home page for your machine learning dashboards. Explore <b>Salary Prediction</b> and
            <b>Water Quality Prediction</b> with clear, step‑by‑step details.
          </p>
          <div class="small">Last updated: {datetime.now().strftime('%b %d, %Y %I:%M %p')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # --------------- Overview Badges ---------------
    colA, colB, colC, colD = st.columns([1, 1, 1, 1])
    with colA:
        st.markdown('<span class="badge">Framework: Streamlit</span>', unsafe_allow_html=True)
    with colB:
        st.markdown('<span class="badge">Models: RandomForest, XGBoost*</span>', unsafe_allow_html=True)
    with colC:
        st.markdown('<span class="badge">Encoders: One‑Hot, StandardScaler</span>', unsafe_allow_html=True)
    with colD:
        st.markdown('<span class="badge">Artifacts: .pkl</span>', unsafe_allow_html=True)

    st.markdown("\n")
    st.markdown("""
    <h1 style='text-align: center;
               font-size: 42px;
               font-weight: bold;
               background: -webkit-linear-gradient(45deg, #00c6ff, #0072ff);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               text-shadow: 0px 0px 12px rgba(0,114,255,0.5);'>
    💧 Water Quality Prediction System
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h1 style='text-align: center;
               font-size: 42px;
               font-weight: bold;
               background: -webkit-linear-gradient(45deg, #ff6a00, #ee0979);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               text-shadow: 0px 0px 12px rgba(238,9,121,0.5);'>
    💰 Salary Prediction System
    </h1>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>
    .typing {
      width: 20ch;
      white-space: nowrap;
      overflow: hidden;
      border-right: 3px solid #0072ff;
      font-size: 36px;
      font-weight: bold;
      animation: typing 8s steps(29) infinite alternate, blink .9s step-end infinite;
    }
    @keyframes typing {
      from { width: 0 }
      to { width: 30ch }
    }
    @keyframes blink {
      50% { border-color: transparent }
    }
    </style>
    <div style='text-align:center;'>
      <div class='typing'>💧 Salary Prediction System </div>
    </div>
    """, unsafe_allow_html=True)

    # --------------- SALARY PREDICTION CARD ---------------
    with st.container():
        st.markdown("### 💼 Salary Prediction — Details (Step by Step)")
        st.markdown('<div class="card">', unsafe_allow_html=True)

        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown("**Problem**: Predict annual salary (₹ / $) from candidate/job features.")
            st.markdown("**Task Type**: Regression (or Income Bracket Classification → then map to salary).")
            st.markdown("**Primary Model**: RandomForestRegressor (trained offline, loaded via `joblib.load`).")

            # Steps one by one
            st.markdown("#### Pipeline — One by One")
            steps = [
                ("Input", "Collect form inputs: age, education, years_experience, job_role, country, remote, skills."),
                ("Validation", "Basic range checks (e.g., age 18–70) + required fields."),
                ("Feature Eng.",
                 "One‑hot encode categoricals (job_role, country, education). Scale numeric fields if needed."),
                ("Prediction", "Load model.pkl, align columns to training schema, call model.predict(X_aligned)."),
                ("Post‑process", "Clamp to sensible bounds; format currency in INR/USD based on user selection."),
                ("Explain", "Show feature importances, partial dependence or SHAP (optional)."),
            ]
            for i, (title, desc) in enumerate(steps, start=1):
                st.markdown(
                    f"<div class='step'><div class='stepnum'>{i}</div><div><b>{title}</b><br><span class='muted'>{desc}</span></div></div>",
                    unsafe_allow_html=True)

            # Expected schema table
            st.markdown("#### Expected Input Schema")
            salary_schema = pd.DataFrame([
                {"field": "age", "type": "int", "example": 28},
                {"field": "education", "type": "category", "example": "Bachelor"},
                {"field": "years_experience", "type": "float", "example": 3.5},
                {"field": "job_role", "type": "category", "example": "Data Analyst"},
                {"field": "country", "type": "category", "example": "India"},
                {"field": "remote", "type": "bool", "example": True},
                {"field": "skills", "type": "multilabel", "example": "Python, SQL"},
            ])
            st.dataframe(salary_schema, use_container_width=True, hide_index=True)

            with st.expander("Preprocessing Components"):
                st.markdown(
                    """
                    - **One‑Hot Encoder** fit on training categories (store as `encoder.pkl`).  
                    - **Column Aligner** to match training columns (missing → 0; unseen → 'Other').  
                    - **Scaler** (optional) for numeric columns.
                    """
                )

            with st.expander("Evaluation & Monitoring"):
                st.markdown("- Offline metrics: R², MAE, RMSE on hold‑out set.")
                st.markdown("- Online checks: input drift, out‑of‑range guardrails, logging predictions.")

        with c2:
            st.markdown("#### Quick Metrics (example)")
            st.markdown('<div class="metricbox">', unsafe_allow_html=True)
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.metric(label="R²", value="0.82")
            with mcol2:
                st.metric(label="MAE (₹)", value="₹ 58,000")
            mcol3, mcol4 = st.columns(2)
            with mcol3:
                st.metric(label="RMSE (₹)", value="₹ 96,000")
            with mcol4:
                st.metric(label="Train Rows", value="48,210")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("#### Open the App")
            if st.button("💼 Go to Salary Prediction"):
                st.session_state["page"] = "salary"

            if st.button("💧 Go to Water Quality Prediction"):
                st.session_state["page"] = "water"

            page = st.session_state.get("page", "home")

            if page == "salary":
                st.title("Salary Prediction")
                st.write("Show salary prediction form here.")
            elif page == "water":
                st.title("Water Quality Prediction")
                st.write("Show water quality prediction form here.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("\n")

    # --------------- WATER QUALITY CARD ---------------
    st.markdown("""
    <style>
    .typing {
      width: 20ch;
      white-space: nowrap;
      overflow: hidden;
      border-right: 3px solid #0072ff;
      font-size: 36px;
      font-weight: bold;
      animation: typing 4s steps(20) infinite alternate, blink .7s step-end infinite;
    }
    @keyframes typing {
      from { width: 0 }
      to { width: 20ch }
    }
    @keyframes blink {
      50% { border-color: transparent }
    }
    </style>
    <div style='text-align:center;'>
      <div class='typing'>💧 Water Quality Prediction</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("### 💧 Water Quality Prediction — Details (Step by Step)")
        st.markdown('<div class="card">', unsafe_allow_html=True)

        w1, w2 = st.columns([1.2, 1])
        with w1:
            st.markdown("**Problem**: Predict water potability / quality index from physico‑chemical parameters.")
            st.markdown("**Task Type**: Classification (Potable / Not Potable) or Regression (WQI score).")
            st.markdown("**Primary Model**: RandomForestClassifier or XGBoostClassifier (loaded via `joblib`).")

            st.markdown("#### Pipeline — One by One")
            wsteps = [
                ("Input",
                 "Collect: pH, Hardness, Solids (TDS), Chloramines, Sulfate, Conductivity, Organic Carbon, Turbidity, Trihalomethanes, etc."),
                ("Validation", "Range checks per parameter (e.g., pH 0–14; no negatives)."),
                ("Imputation", "Median/mode impute missing numeric/categorical values."),
                ("Scaling", "Standardize numeric features if model benefits."),
                ("Prediction", "Load model.pkl → predict class and probability or continuous WQI."),
                ("Explain", "Show top features; threshold slider for classification decisions."),
            ]
            for i, (title, desc) in enumerate(wsteps, start=1):
                st.markdown(
                    f"<div class='step'><div class='stepnum'>{i}</div><div><b>{title}</b><br><span class='muted'>{desc}</span></div></div>",
                    unsafe_allow_html=True)

            st.markdown("#### Expected Input Schema")
            water_schema = pd.DataFrame([
                {"field": "pH", "type": "float", "example": 7.2},
                {"field": "Hardness", "type": "float", "example": 204.9},
                {"field": "Solids(TDS)", "type": "float", "example": 18630.0},
                {"field": "Chloramines", "type": "float", "example": 7.1},
                {"field": "Sulfate", "type": "float", "example": 350.0},
                {"field": "Conductivity", "type": "float", "example": 420.0},
                {"field": "OrganicCarbon", "type": "float", "example": 10.5},
                {"field": "Trihalomethanes", "type": "float", "example": 56.0},
                {"field": "Turbidity", "type": "float", "example": 3.6},
            ])
            st.dataframe(water_schema, use_container_width=True, hide_index=True)

            with st.expander("Preprocessing Components"):
                st.markdown(
                    """
                    - **Imputer** for numeric columns (median) and optional categorical (most frequent).  
                    - **Scaler** (StandardScaler/MinMaxScaler) as needed.  
                    - **Column Aligner** to ensure inference columns match training schema.
                    """
                )
            with st.expander("Evaluation & Monitoring"):
                st.markdown("- Offline metrics: Accuracy, F1, ROC‑AUC (classification) or R²/MAE (regression).")
                st.markdown("- Calibration plot and threshold tuning (optional).")

        with w2:
            st.markdown("#### Quick Metrics (example)")
            st.markdown('<div class="metricbox">', unsafe_allow_html=True)
            a1, a2 = st.columns(2)
            with a1:
                st.metric(label="Accuracy", value="0.91")
            with a2:
                st.metric(label="ROC‑AUC", value="0.95")
            a3, a4 = st.columns(2)
            with a3:
                st.metric(label="F1", value="0.90")
            with a4:
                st.metric(label="# Samples", value="32,500")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("#### Open the App")
            if page == "🏠 Home":
                st.header("Welcome!")
                st.write("Choose a project from the sidebar.")

            elif page == "💼 Salary Prediction":
                st.header("💼 Salary Prediction")
                st.write("Here goes your salary prediction model UI.")

            elif page == "💧 Water Quality":
                st.header("💧 Water Quality Prediction")
                st.write("Here goes your water quality model UI.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("\n")

    # --------------- Developer Notes / Integration ---------------
    with st.expander("📦 How to integrate your trained models (.pkl)"):
        st.markdown(
            """
            **Folder structure (suggested):**
            ```
            /app
            ├─ Home.py
            ├─ models/
            │   ├─ salary/
            │   │   ├─ model.pkl
            │   │   ├─ encoder.pkl
            │   │   └─ train_columns.json
            │   └─ water/
            │       ├─ model.pkl
            │       └─ train_columns.json
            └─ pages/
                ├─ 1_💼_Salary_Prediction.py
                └─ 2_💧_Water_Quality_Prediction.py
            ```

            **On each page:**
            - Load artifacts with `import joblib` → `joblib.load('models/.../model.pkl')`.  
            - Keep a list of `train_columns` and reindex incoming features to this list (fill missing with 0).  
            - For classification → if you map classes to salary (e.g., `{'<=50K': 40000, '>50K': 80000}`), do the mapping **after** `predict`.
            """
        )

    with st.expander("🧪 Minimal inference helpers (copy into the respective pages)"):
        st.code(
            """
    import json, joblib, numpy as np, pandas as pd

    def load_artifacts(model_dir):
        model = joblib.load(f"{model_dir}/model.pkl")
        cols = json.load(open(f"{model_dir}/train_columns.json"))
        enc = None
        try:
            enc = joblib.load(f"{model_dir}/encoder.pkl")
        except Exception:
            pass
        return model, cols, enc

    # Align columns to training schema
    # X is a pandas DataFrame with raw (already encoded if enc is None) features

    def align_columns(X: pd.DataFrame, train_columns: list[str]) -> pd.DataFrame:
        X_aligned = X.reindex(columns=train_columns, fill_value=0)
        missing = set(train_columns) - set(X.columns)
        if missing:
            print("Filled missing columns with 0:", sorted(missing)[:5], "...")
        return X_aligned
            """,
            language="python",
        )
        st.empty()  # adds a blank space
        st.write("")
        st.write("")
    st.markdown("""
    <div style='text-align:center;
                padding:20px;
                border-radius:20px;
                background:rgba(255,255,255,0.05);
                border:1px solid rgba(255,255,255,0.2);
                backdrop-filter:blur(10px);
                box-shadow:0 8px 25px rgba(0,0,0,0.2);'>
        <h1 style='font-size:38px; color:#00bfff;'>💧 Water Quality Prediction System</h1>
    </div>
    """, unsafe_allow_html=True)
    st.empty()  # adds a blank space
    st.write("")
    st.write("")
    st.markdown("""
    <div style='text-align:center;
                padding:20px;
                border-radius:20px;
                background:rgba(255,255,255,0.05);
                border:1px solid rgba(255,255,255,0.2);
                backdrop-filter:blur(10px);
                box-shadow:0 8px 25px rgba(0,0,0,0.2);'>
        <h1 style='font-size:38px; color:#ff6347;'>💰 Salary Prediction System</h1>
    </div>
    """, unsafe_allow_html=True)
    st.empty()  # adds a blank space
    st.write("")
    st.write("")
    with st.expander("🧭 Navigation without pages (single‑file fallback)"):
        st.markdown(
            "If you don't want to create separate files, you can use a selectbox on this Home page to switch between mini‑UIs.")
        choice = st.selectbox("Quick preview (single‑file demo)", ["— Select —", "Salary mini‑form", "Water mini‑form"])
        if choice == "Salary mini‑form":
            age = st.number_input("Age", 18, 70, 28)
            edu = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
            exp = st.number_input("Years of Experience", 0.0, 50.0, 3.5, step=0.5)
            role = st.text_input("Job Role", "Data Analyst")
            country = st.text_input("Country", "India")
            remote = st.checkbox("Remote", True)
            if st.button("Predict (demo)"):
                st.info("This is a UI demo only. Hook up your trained model on the dedicated page.")
        elif choice == "Water mini‑form":
            pH = st.number_input("pH", 0.0, 14.0, 7.2)
            hard = st.number_input("Hardness", 0.0, 1000.0, 204.9)
            tds = st.number_input("Solids (TDS)", 0.0, 100000.0, 18630.0)
            chl = st.number_input("Chloramines", 0.0, 50.0, 7.1)
            sul = st.number_input("Sulfate", 0.0, 1000.0, 350.0)
            cond = st.number_input("Conductivity", 0.0, 2000.0, 420.0)
            org = st.number_input("Organic Carbon", 0.0, 50.0, 10.5)
            thm = st.number_input("Trihalomethanes", 0.0, 200.0, 56.0)
            turb = st.number_input("Turbidity", 0.0, 100.0, 3.6)
            if st.button("Predict (demo)"):
                st.info("This is a UI demo only. Hook up your trained model on the dedicated page.")

    st.markdown("---")
    import streamlit as st
    import pandas as pd
    import plotly.express as px

    # Sample glacier location data (latitude, longitude, name)
    data = {
        "Name": ["Glacier A", "Glacier B", "Glacier C"],
        "Latitude": [61.5, 46.8, 78.9],
        "Longitude": [-149.9, 11.2, 16.0],
        "Size_km2": [120, 80, 200]
    }
    df = pd.DataFrame(data)

    st.title("World Glacier Map")

    fig = px.scatter_geo(df,
                         lat="Latitude",
                         lon="Longitude",
                         hover_name="Name",
                         size="Size_km2",
                         projection="natural earth",
                         title="Sample Glacier Locations")
    st.plotly_chart(fig)
    st.caption("Tip: Put your links (GitHub, LinkedIn, Email) in the sidebar of each page for quick team contacts.")


elif choice == "Portfolio":
    st.markdown(f"<div class='big-title'>See <span style='color:{PRIMARY}'>Details</span></div>",
                unsafe_allow_html=True)
    st.markdown("""
    <h2 style='text-align: center;
               font-size: 38px;
               color: #20c997;
               text-shadow: 0px 0px 8px #20c997, 0px 0px 16px #0dcaf0; 
               transition: 0.3s;'>
    💼 Our Portfolio
    </h2>
    """, unsafe_allow_html=True)

    import streamlit as st

    # Page settings
    st.set_page_config(page_title="🌟 Portfolio", page_icon="✨", layout="wide")

    # ---------- HERO / INTRO ----------
    col1, col2 = st.columns([1, 3])
    import streamlit as st

    col1, col2 = st.columns([1, 3])

    with col1:
            url = "https://img.freepik.com/premium-photo/illustration-business-man-american-cartoon-art-style-images-with-ai-generated_545052-2744.jpg"
            st.image(url,  )
    with col2:
        st.markdown("""
            ## 👋 Hi, I'm Vishesh Kumar Prajapati
            **Machine Learning Enthusiast | Data Scientist | Full Stack Web Developer**

            Passionate about building intelligent systems and stunning dashboards.  
            Connect with me 👇
        """)
        st.markdown(
            """
            <a href="https://www.linkedin.com/in/vishesh-kumar-prajapati-45111829a?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app" target="_blank">
                <img src="https://img.shields.io/badge/LinkedIn-blue?logo=linkedin&logoColor=white" height="30">
            </a>
            <a href="https://github.com/vishes-i" target="_blank">
                <img src="https://img.shields.io/badge/GitHub-black?logo=github&logoColor=white" height="30">
            </a>
            <a href="mailto:visheshprajapati7920@gmail.com" target="_blank">
                <img src="https://img.shields.io/badge/Email-red?logo=gmail&logoColor=white" height="30">
            </a>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    import streamlit as st

    # --- SKILLS / BADGES SECTION ---
    st.markdown("## 🛠️ Skills & Technologies")

    st.markdown(
        """
        <div style='display: flex; flex-wrap: wrap; gap: 10px;'>
            <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikitlearn&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/Numpy-013243?logo=numpy&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/Matplotlib-00457C?logo=matplotlib&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=black&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/MySQL-4479A1?logo=mysql&logoColor=white&style=for-the-badge" height="30">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
    # ---------- PROJECTS ----------
    st.markdown("## 💼 Projects Showcase")

    projects = [
        {
            "title": "Salary Prediction App",
            "desc": "ML model with Random Forest that predicts salaries based on education, job role, and experience.",
            "link": "https://github.com/yourusername/salary-prediction"
        },
        {
            "title": "Water Quality Prediction",
            "desc": "Classifies water as drinkable or not based on chemical parameters using ML.",
            "link": "https://github.com/yourusername/water-quality"
        },
        {
            "title": "Personal Portfolio Website",
            "desc": "Modern responsive portfolio website built with HTML, CSS, and JS.",
            "link": "https://your-portfolio-link.com"
        }
    ]

    # Custom CSS for cards
    st.markdown("""
        <style>
            .card {
                background: linear-gradient(135deg, #f6f9fc, #e9f0ff);
                border-radius: 20px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0px 6px 15px rgba(0,0,0,0.1);
                transition: transform 0.3s ease-in-out;
            }
            .card:hover {
                transform: translateY(-6px);
                box-shadow: 0px 12px 25px rgba(0,0,0,0.2);
            }
            .card-title {
                font-size: 20px;
                font-weight: 700;
                color: #2c3e50;
                margin-bottom: 8px;
            }
            .card-desc {
                font-size: 15px;
                color: #555;
            }
        </style>
    """, unsafe_allow_html=True)

    for project in projects:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">{project['title']}</div>
                <div class="card-desc">{project['desc']}</div>
                <br>
                <a href="{project['link']}" target="_blank">🔗 View Project</a>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ---------- CONTACT FORM ----------
    st.markdown("## 📩 Contact Me")

    with st.form("contact_form", clear_on_submit=True):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        query = st.text_area("Your Message")
        uploaded_file = st.file_uploader("Upload a file/photo (optional)", type=["png", "jpg", "jpeg", "pdf"])

        submitted = st.form_submit_button("Send Message ✉️")
        if submitted:
            if name and email and query:
                st.success(f"✅ Thanks {name}! Your message has been sent successfully.")
                if uploaded_file:
                    st.info(f"📎 You uploaded: {uploaded_file.name}")
            else:
                st.error("⚠️ Please fill in all required fields (Name, Email, Message).")

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import streamlit as st

    st.title("❄️ Glacier Melting Tracker")

    # Upload images
    before_file = st.file_uploader("Upload BEFORE Glacier Image", type=["jpg", "png"])
    after_file = st.file_uploader("Upload AFTER Glacier Image", type=["jpg", "png"])

    if before_file and after_file:
        # Read images in grayscale
        before = cv2.imdecode(np.frombuffer(before_file.read(), np.uint8), 0)
        after = cv2.imdecode(np.frombuffer(after_file.read(), np.uint8), 0)

        # Threshold to highlight glacier regions
        _, before_mask = cv2.threshold(before, 127, 255, cv2.THRESH_BINARY)
        _, after_mask = cv2.threshold(after, 127, 255, cv2.THRESH_BINARY)

        # Calculate glacier area (white pixels)
        before_area = np.sum(before_mask == 255)
        after_area = np.sum(after_mask == 255)
        melted = (before_area - after_area) / before_area * 100

        st.success(f"❄️ Glacier Melted: {melted:.2f}%")

        # --- Plot Graphs ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Show images
        axes[0].imshow(before, cmap="gray")
        axes[0].set_title("Before Glacier")
        axes[0].axis("off")

        axes[1].imshow(after, cmap="gray")
        axes[1].set_title("After Glacier")
        axes[1].axis("off")

        # Bar chart of glacier area
        axes[2].bar(["Before", "After"], [before_area, after_area], color=["blue", "red"])
        axes[2].set_title("Glacier Area Comparison")
        axes[2].set_ylabel("Pixel Count (Area)")

        st.pyplot(fig)

    import streamlit as st
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    st.title("💧 Water Quality Prediction System")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload Water Quality Dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📊 Dataset Preview:", df.head())

        if "Potability" in df.columns:
            # Features and labels
            X = df.drop("Potability", axis=1)
            y = df["Potability"]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            st.success("✅ Model trained successfully!")

            # User input for prediction
            st.subheader("🔎 Test a Water Sample")
            ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0)
            hardness = st.number_input("Hardness", min_value=0.0, value=150.0)
            solids = st.number_input("Solids", min_value=0.0, value=200.0)
            chloramines = st.number_input("Chloramines", min_value=0.0, value=7.0)
            sulfate = st.number_input("Sulfate", min_value=0.0, value=400.0)
            conductivity = st.number_input("Conductivity", min_value=0.0, value=250.0)
            organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=3.0)
            trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=60.0)
            turbidity = st.number_input("Turbidity", min_value=0.0, value=3.0)

            # Arrange in same order as dataset
            sample = [[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes,
                       turbidity]]

            if st.button("🔮 Predict Potability"):
                pred = model.predict(sample)
                st.success("💧 Safe to Drink" if pred[0] == 1 else "⚠️ Not Safe to Drink")
        else:
            st.error("❌ The dataset must contain a 'Potability' column.")
    import pandas as pd
    import plotly.express as px
    import streamlit as st

    st.title("💧 Water Quality Prediction by pH Values")

    # Upload CSV dataset
    uploaded_file = st.file_uploader("Upload your water potability CSV", type=["csv"])

    if uploaded_file is not None:
        # Load dataset
        df = pd.read_csv(uploaded_file)

        # Ensure dataset has required columns
        if "ph" in df.columns and "Potability" in df.columns:
            # Map Potability: 1 = Safe, 0 = Not Safe
            df["Potability"] = df["Potability"].map({1: "✅ Safe to Drink", 0: "⚠️ Not Safe"})

            # Scatter plot
            fig = px.scatter(
                df,
                x="ph",
                y="Hardness",  # You can change to another parameter (like Sulfate, Solids, etc.)
                color="Potability",
                title="pH vs Water Quality Potability",
                labels={"ph": "pH Value", "Hardness": "Hardness Level"},
                opacity=0.7
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("CSV file must contain 'ph' and 'Potability' columns.")
    else:
        st.info("Please upload a CSV file to visualize water quality data.")

    import streamlit as st
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    st.title("💼 Salary Prediction System")

    # Upload CSV file
    uploaded_file = st.file_uploader("📂 Upload Salary Dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.write("📊 Dataset Preview", df.head())

        if "income" in df.columns:
            # Features & Labels
            X = pd.get_dummies(df.drop("income", axis=1))
            y = df["income"]

            # Train/Test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            st.success("✅ Model trained successfully!")

            # Show accuracy
            accuracy = model.score(X_test, y_test)
            st.info(f"📈 Model Accuracy: {accuracy * 100:.2f}%")

            # Predict from test sample
            idx = st.number_input("Choose row index from test data", min_value=0, max_value=len(X_test) - 1, value=0)
            if st.button("🔮 Predict Salary"):
                sample = X_test.iloc[[idx]]
                pred = model.predict(sample)
                st.success(f"💰 Predicted Salary Category: {pred[0]}")
        else:
            st.error("❌ Dataset must contain an 'income' column.")
    else:
        st.warning("⬆️ Please upload a CSV file to continue.")
    import pandas as pd
    import plotly.express as px
    import streamlit as st

    st.title("💼 Salary Prediction by Sector")

    # Example dataset (you can replace with your salary_data.csv or model output)
    data = {
        "Sector": ["Student", "Intern", "Junior Engineer", "Senior Engineer", "Manager", "Data Scientist",
                   "Software Engineer"],
        "Predicted_Salary_USD": [5000, 12000, 35000, 70000, 95000, 110000, 105000]
    }

    df = pd.DataFrame(data)

    # Plot
    fig = px.bar(
        df,
        x="Sector",
        y="Predicted_Salary_USD",
        color="Predicted_Salary_USD",
        text="Predicted_Salary_USD",
        title="💰 Salary Prediction from Student to Engineer & Beyond",
        color_continuous_scale="Blues"
    )

    fig.update_traces(texttemplate='$%{text:,.0f}', textposition="outside")
    fig.update_layout(
        xaxis_title="Career Sector",
        yaxis_title="Predicted Salary (USD)",
        uniformtext_minsize=8,
        uniformtext_mode="hide"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Featured Projects")
    st.markdown("<div class='grid-3'>", unsafe_allow_html=True)
    projects = [
        {
            "title": "Salary Prediction Dashboard",
            "desc": "RandomForest pipeline with categorical encoding and CSV batch scoring.",
            "tags": "Python • scikit‑learn • Streamlit"
        },
        {
            "title": "Water Potability Classifier",
            "desc": "Logistic Regression baseline with StandardScaler; supports form & CSV input.",
            "tags": "Python • scikit‑learn • Streamlit"
        },
        {
            "title": "Glacier Change Visualizer",
            "desc": "Image differencing + portfolio UI (separate app).",
            "tags": "OpenCV • Streamlit"
        }
    ]
    for proj in projects:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"**{proj['title']}**")
        st.write(proj['desc'])
        st.caption(proj['tags'])
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    import streamlit as st
    import pandas as pd
    import pydeck as pdk

    st.set_page_config(page_title="World Map", layout="wide")
    st.title("🌍 World Map (PyDeck)")

    # Countries with coordinates (approx lat/lon)
    data = pd.DataFrame([

    ])

    # Initial view centered around South Asia
    initial_view = pdk.ViewState(latitude=30, longitude=80, zoom=2)

    # Scatterplot layer
    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position='[lon, lat]',
        get_radius=600000,  # in meters
        get_fill_color=[255, 0, 0],  # red circles
        pickable=True,
        auto_highlight=True,
    )

    # Tooltip
    tooltip = {"text": "{country}\nLat: {lat}\nLon: {lon}"}

    # Deck
    r = pdk.Deck(
        layers=[scatter],
        initial_view_state=initial_view,
        tooltip=tooltip,
    )

    st.pydeck_chart(r)
    st.caption("Tip: Zoom, pan, and hover over the red circles to see country details.")

    st.download_button(
        label="Download README.md",
        data=textwrap.dedent(
            """
            # Portfolio — Salary & Water Quality App
            Features:
            - Salary regression (RandomForest demo)
            - Water potability classification (Logistic demo)
            - Batch CSV scoring & model uploads.
            """
        ).strip(),

        file_name="README.md"
    )

elif choice == "Salary Prediction":


    st.markdown(f"<div class='big-title'>Our  <span style='color:{PRIMARY}'>Prediction Model</span></div>",
                unsafe_allow_html=True)
    st.markdown("""
    <hr style='border: 2px solid #20c997; border-radius: 5px;'>
    <h2 style='text-align: center; font-size: 34px; color:#333;'>
    🚀 📊 Employee Salary Prediction Dashboard
    </h2>
    <hr style='border: 2px solid #20c997; border-radius: 5px;'>
    """, unsafe_allow_html=True)

    # ===============================
    # Upload Options
    # ===============================
    uploaded_model = st.file_uploader("📂 Upload Trained Model (.pkl)", type=["pkl", "joblib"], key="salary_model")
    uploaded_encoder = st.file_uploader("📂 Upload Label Encoders (.pkl)", type=["pkl", "joblib"], key="salary_encoder")
    uploaded_csv = st.file_uploader("📂 Upload Dataset (CSV)", type=["csv"], key="salary_csv")

    # ----------------- Load CSV ------------------
    df = None
    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv)
            st.success("✅ Dataset loaded successfully")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"❌ Error reading CSV: {e}")

    # ===============================
    # Salary Model Loader / Demo Fallback
    # ===============================
    def load_pickle(file):
        try:
            return joblib.load(file)
        except Exception as e:
            st.error(f"Error loading pickle: {e}")
            return None

    def safe_predict(model, X):
        try:
            return model.predict(X)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return None

    def get_demo_salary_model():
        df_demo = pd.DataFrame({
            "age": np.random.randint(18, 65, 200),
            "workclass": np.random.choice(["Private", "Self-emp", "Government"], 200),
            "fnlwgt": np.random.randint(10000, 500000, 200),
            "education": np.random.choice(["Bachelors", "HS-grad", "Masters"], 200),
            "educational-num": np.random.randint(1, 16, 200),
            "marital-status": np.random.choice(["Married", "Single", "Divorced"], 200),
            "occupation": np.random.choice(["Tech-support", "Craft-repair", "Exec-managerial"], 200),
            "relationship": np.random.choice(["Husband", "Wife", "Not-in-family"], 200),
            "race": np.random.choice(["White", "Black", "Asian"], 200),
            "gender": np.random.choice(["Male", "Female"], 200),
            "capital-gain": np.random.randint(0, 10000, 200),
            "capital-loss": np.random.randint(0, 2000, 200),
            "hours-per-week": np.random.randint(20, 60, 200),
            "native-country": np.random.choice(["United-States", "India", "Mexico"], 200),
        })
        y_demo = np.random.choice(["<=50K", ">50K"], 200)

        num_features = ["age", "fnlwgt", "educational-num", "capital-gain", "capital-loss", "hours-per-week"]
        cat_features = ["workclass", "education", "marital-status", "occupation",
                        "relationship", "race", "gender", "native-country"]

        preprocessor = ColumnTransformer([
            ("num", SimpleImputer(strategy="mean"), num_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_features)
        ])

        pipe = Pipeline([
            ("pre", preprocessor),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        pipe.fit(df_demo, y_demo)
        return pipe

    DEMO_SALARY = get_demo_salary_model()

    # ===============================
    # Model Source
    # ===============================
    with st.expander("Model Source", expanded=True):
        model_type = st.radio("Select Model", ["Demo (built-in)", "Upload .pkl"],
                              horizontal=True, key="salary_model_source")
        user_salary_model = None
        if model_type == "Upload .pkl" and uploaded_model:
            user_salary_model = load_pickle(uploaded_model)
        salary_model = user_salary_model if user_salary_model else DEMO_SALARY

    # ===============================
    # Single Sample Prediction
    # ===============================
    st.subheader("🔹 Single Prediction")

    col1, col2 = st.columns(2)
    age = col1.number_input("Age", 18, 100, 30)
    workclass = col1.selectbox("Workclass", ["Private", "Self-emp", "Government"])
    fnlwgt = col1.number_input("Fnlwgt", 10000, 1000000, 200000)
    education = col1.selectbox("Education", ["Bachelors", "HS-grad", "Masters"])
    educational_num = col1.number_input("Educational-num", 1, 16, 10)

    marital_status = col2.selectbox("Marital-status", ["Married", "Single", "Divorced"])
    occupation = col2.selectbox("Occupation", ["Tech-support", "Craft-repair", "Exec-managerial"])
    relationship = col2.selectbox("Relationship", ["Husband", "Wife", "Not-in-family"])
    race = col2.selectbox("Race", ["White", "Black", "Asian"])
    gender = col2.selectbox("Gender", ["Male", "Female"])
    capital_gain = col2.number_input("Capital-gain", 0, 100000, 0)
    capital_loss = col2.number_input("Capital-loss", 0, 5000, 0)
    hours_per_week = col2.number_input("Hours-per-week", 1, 100, 40)
    native_country = col2.selectbox("Native-country", ["United-States", "India", "Mexico"])

    sample_salary_df = pd.DataFrame([{
        "age": age, "workclass": workclass, "fnlwgt": fnlwgt, "education": education,
        "educational-num": educational_num, "marital-status": marital_status,
        "occupation": occupation, "relationship": relationship, "race": race, "gender": gender,
        "capital-gain": capital_gain, "capital-loss": capital_loss,
        "hours-per-week": hours_per_week, "native-country": native_country,
    }])

    if st.button("Predict Salary"):
        preds = safe_predict(salary_model, sample_salary_df)
        if preds is not None:
            st.success(f"💰 Predicted Income: **{preds[0]}**")

    # ===============================
    # Batch CSV Predictions
    # ===============================
    st.subheader("📂 Batch Predictions from CSV")

    if uploaded_csv is not None:
        if st.button("Run Batch Predictions"):
            try:
                input_data = df.drop(columns=["income"], errors="ignore")
                preds = safe_predict(salary_model, input_data)
                if preds is not None:
                    df["income_pred"] = preds
                    st.dataframe(df.head(50))
                    st.download_button("Download Predictions CSV",
                                       df.to_csv(index=False).encode("utf-8"),
                                       "salary_predictions.csv")
            except Exception as e:
                st.error(f"Error: {e}")

    # ===============================
    # Dynamic Graphs
    # ===============================
    if df is not None and "income" in df.columns:
        st.subheader("📊 Data Insights")

        # Income distribution
        fig1 = px.histogram(df, x="income", color="income", title="Income Distribution")
        st.plotly_chart(fig1, use_container_width=True)

        # Age vs Income
        fig2 = px.box(df, x="income", y="age", title="Age vs Income")
        st.plotly_chart(fig2, use_container_width=True)

        # Hours per week vs Income
        fig3 = px.violin(df, x="income", y="hours-per-week", box=True,
                         title="Work Hours vs Income")
        st.plotly_chart(fig3, use_container_width=True)

        # Education vs Income
        fig4 = px.histogram(df, x="education", color="income",
                            barmode="group", title="Education vs Income")
        st.plotly_chart(fig4, use_container_width=True)
 #---------------------------------------------
    import streamlit as st
    import pandas as pd
    import joblib
    import random
    import sklearn
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, mean_squared_error

    # ----------------- Page Config ------------------
    st.set_page_config(
        page_title="💰 Salary Prediction System",
        layout="wide",  # Full-screen width
        initial_sidebar_state="expanded"
    )

    # ----------------- Constants ------------------
    FEATURE_COLUMNS = [
        'age', 'education', 'occupation', 'hours-per-week', 'gender',
        'capital-gain', 'capital-loss', 'marital-status'
    ]

    # ----------------- Title ------------------
    st.markdown(
        "====================================================================================================================================================================================================================================================================================")
    st.title("💼 Salary Prediction System ")
    st.markdown("Predict salary category and estimated salary using ML.")

    # ----------------- File Uploaders ------------------
    # ----------------- File Uploaders ------------------
    st.header("📂 Upload Files")

    uploaded_model = st.file_uploader("Upload Model (.pkl)", type=["pkl", "joblib"], key="salary_model_file")
    uploaded_encoder = st.file_uploader("Upload Encoder (.pkl)", type=["pkl", "joblib"], key="salary_encoder_file")
    uploaded_csv = st.file_uploader("Upload Dataset (.csv)", type=["csv"], key="salary_csv_file")

    # ----------------- Initialize ------------------
    model, encoders, df = None, None, None
    trained = False

    # ----------------- Load CSV ------------------
    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv)
            st.success("✅ Dataset loaded successfully")
            st.subheader("📊 Dataset Preview")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"❌ Error reading CSV: {e}")


    # ----------------- Train Model ------------------
    def train_model():
        global df
        if df is None:
            st.error("❌ Please upload a dataset first.")
            return None, None, 0, 0

        X = df[FEATURE_COLUMNS].copy()
        y = df["income"]

        enc = {}
        for col in X.select_dtypes(include=["object"]).columns:
            enc[col] = LabelEncoder()
            X[col] = enc[col].fit_transform(X[col])

        enc["income"] = LabelEncoder()
        y = enc["income"].fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

        mse = mean_squared_error(y_test, clf.predict(X_test))
        rmse = mse ** 0.5

        return clf, enc, accuracy_score(y_test, clf.predict(X_test)), rmse


    # ----------------- Safe Load or Train ------------------
    def safe_load_or_train():
        global model, encoders, trained
        try:
            retrain = False

            if uploaded_model and uploaded_encoder:
                model = joblib.load(uploaded_model)
                encoders = joblib.load(uploaded_encoder)

                model_version = getattr(model, '_sklearn_version', None)
                current_version = sklearn.__version__
                if model_version and model_version != current_version:
                    st.warning(f"Incompatible model version: trained on {model_version}, current is {current_version}")
                    retrain = True

                for col in FEATURE_COLUMNS + ['income']:
                    if df is not None and df[col].dtype == 'object' and col not in encoders:
                        st.warning(f"Encoder for '{col}' missing. Retraining...")
                        retrain = True
                        break
            else:
                retrain = True

            if retrain:
                model, encoders, acc, rmse = train_model()
                if model is not None:
                    st.success(f"🔄 Model retrained | Accuracy: {acc:.2f} | RMSE: {rmse:.2f}")
                    trained = True
            else:
                trained = True

        except Exception as e:
            st.warning(f"⚠️ Model loading failed: {e} - Re-training now...")
            model, encoders, acc, rmse = train_model()
            trained = True


    safe_load_or_train()


    # ----------------- User Input ------------------
    def user_input():
        st.subheader("📝 Enter Your Information")

        if not encoders:
            st.error("❌ Encoders not available. Please retrain the model.")
            return None, None

        age = st.slider("Age", 18, 90, 30)
        education = st.selectbox("Education", encoders["education"].classes_)
        occupation = st.selectbox("Occupation", encoders["occupation"].classes_)
        hours_per_week = st.slider("Hours per Week", 1, 99, 40)
        gender = st.selectbox("Gender", encoders["gender"].classes_)
        capital_gain = st.number_input("Capital Gain", min_value=0, max_value=99999, value=0)
        capital_loss = st.number_input("Capital Loss", min_value=0, max_value=99999, value=0)
        marital_status = st.selectbox("Marital Status", encoders["marital-status"].classes_)

        input_dict = {
            "age": age,
            "education": encoders["education"].transform([education])[0],
            "occupation": encoders["occupation"].transform([occupation])[0],
            "hours-per-week": hours_per_week,
            "gender": encoders["gender"].transform([gender])[0],
            "capital-gain": capital_gain,
            "capital-loss": capital_loss,
            "marital-status": encoders["marital-status"].transform([marital_status])[0],
        }

        readable_input = {
            "Age": age,
            "Education": education,
            "Occupation": occupation,
            "Hours per Week": hours_per_week,
            "Gender": gender,
            "Capital Gain": capital_gain,
            "Capital Loss": capital_loss,
            "Marital Status": marital_status
        }

        return pd.DataFrame([input_dict]), readable_input


    # ----------------- Prediction ------------------
    if trained:
        input_df, readable_input = user_input()
        if input_df is not None:
            try:
                input_df = input_df[FEATURE_COLUMNS]
                pred = model.predict(input_df)[0]
                prob = model.predict_proba(input_df)[0]

                label = encoders["income"].inverse_transform([pred])[0]
                confidence = prob[pred] * 100

                # Salary mapping with random range
                income_map_random_range = {
                    '<=50K': (25000, 49000),
                    '>50K': (51000, 150000)
                }
                salary_range = income_map_random_range.get(label, (0, 0))
                predicted_salary = random.randint(*salary_range)

                st.success(f"💰 Predicted Income Category: `{label}`")
                st.success(f"💰 Predicted Salary: ₹{predicted_salary:,.2f}")
                st.info(f"🔐 Confidence: `{confidence:.2f}%`")

                # Show inputs
                st.markdown("### 🧾 Your Inputs")
                st.dataframe(pd.DataFrame([readable_input]))

                # Show probabilities
                st.markdown("### 📊 Prediction Probabilities")
                prob_df = pd.DataFrame({
                    "Category": encoders["income"].classes_,
                    "Probability": prob
                })
                prob_df["Probability"] = (prob_df["Probability"] * 100).round(2)
                prob_df = prob_df.sort_values(by="Probability", ascending=False)
                st.bar_chart(prob_df.set_index("Category"))

            except Exception as e:
                st.error("🚨 Unexpected error during prediction.")
                st.exception(e)
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    # ---------------- Page Config -----------------
    st.set_page_config(page_title="💼 Employee Salary Prediction", layout="wide")
    st.title("💼 Employee Salary Prediction System")

    # ---------------- Upload or Load Data -----------------
    st.sidebar.header("📂 Data Options")
    data_file = st.sidebar.file_uploader("Upload Employee Dataset (CSV)", type=["csv"])

    if data_file is not None:
        df = pd.read_csv(data_file)
    else:
        # Demo dataset
        df = pd.DataFrame({
            "age": np.random.randint(22, 60, 200),
            "education_level": np.random.choice(["High School", "Bachelor", "Master", "PhD"], 200),
            "experience_years": np.random.randint(0, 30, 200),
            "job_role": np.random.choice(["Engineer", "Manager", "Analyst", "Clerk"], 200),
            "hours_per_week": np.random.randint(20, 60, 200),
            "salary": np.random.randint(20000, 150000, 200)
        })

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # ---------------- Feature Selection -----------------
    target = st.sidebar.selectbox("Target Column (Salary)", options=df.columns, index=list(df.columns).index("salary"))
    features = st.sidebar.multiselect("Feature Columns", options=[c for c in df.columns if c != target],
                                      default=[c for c in df.columns if c != target])

    X = pd.get_dummies(df[features], drop_first=True)
    y = df[target]

    # ---------------- Train/Test Split -----------------
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
    random_state = st.sidebar.number_input("Random State", 0, 999, 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # ---------------- Model Training -----------------
    model = RandomForestRegressor(n_estimators=200, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ---------------- Metrics -----------------
    try:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("📊 Model Performance")
    st.metric("RMSE", f"{rmse:,.2f}")
    st.metric("MAE", f"{mae:,.2f}")
    st.metric("R² Score", f"{r2:.3f}")

    # ---------------- Graphs -----------------
    st.subheader("📈 Graphs & Insights")

    col1, col2 = st.columns(2)

    with col1:
        # Actual vs Predicted
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.set_xlabel("Actual Salary")
        ax.set_ylabel("Predicted Salary")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

    with col2:
        # Feature Importance
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig, ax = plt.subplots()
        importances.plot(kind="bar", ax=ax)
        ax.set_title("Feature Importance")
        st.pyplot(fig)

    # Distribution of Salaries
    st.subheader("💵 Salary Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df[target], bins=20, kde=True, ax=ax)
    ax.set_title("Salary Distribution")
    st.pyplot(fig)

    # ---------------- Single Prediction Sandbox -----------------
    st.subheader("🧪 Try a Single Prediction")
    sample_vals = {}
    for col in features:
        if df[col].dtype in [np.int64, np.float64]:
            val = st.number_input(f"{col}", value=float(df[col].mean()))
        else:
            choices = df[col].unique().tolist()
            val = st.selectbox(f"{col}", options=choices)
        sample_vals[col] = val

    if st.button("Predict Salary", type="primary"):
        sample_df = pd.DataFrame([sample_vals])
        sample_df = pd.get_dummies(sample_df)
        sample_df = sample_df.reindex(columns=X.columns, fill_value=0)

        pred_salary = model.predict(sample_df)[0]
        st.success(f"Predicted Salary: ₹{pred_salary:,.2f}")





elif choice == "Water Quality":
    st.markdown(f"<div class='big-title'>Our <span style='color:{PRIMARY}'>Prediction Model</span></div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; 
                background-color: #f8f9fa; 
                padding: 15px; 
                border-radius: 12px; 
                box-shadow: 2px 2px 8px rgba(0,0,0,0.1);'>
        <h2 style='color: #0d6efd; font-size: 34px;'>
            👥 Water Quality <span style='color:#20c997;'>Pridiction Model</span>
        </h2>
    </div>
    """, unsafe_allow_html=True)

    import streamlit as st
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier


    # =========================
    # Utility
    # =========================
    def load_pickle(file):
        try:
            return joblib.load(file)
        except Exception as e:
            st.error(f"Error loading pickle: {e}")
            return None


    def safe_predict(model, X):
        try:
            return model.predict(X)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return None


    # =========================
    # Demo Model (built-in)
    # =========================
    # Train a very simple demo LogisticRegression pipeline
    def get_demo_water_model():
        # fake small dataset for demo
        X_demo = pd.DataFrame({
            "ph": np.random.uniform(6, 8, 100),
            "Hardness": np.random.uniform(100, 300, 100),
            "Solids": np.random.uniform(5000, 20000, 100),
            "Chloramines": np.random.uniform(5, 10, 100),
            "Sulfate": np.random.uniform(200, 400, 100),
            "Conductivity": np.random.uniform(300, 500, 100),
            "Organic_carbon": np.random.uniform(5, 15, 100),
            "Trihalomethanes": np.random.uniform(40, 80, 100),
            "Turbidity": np.random.uniform(2, 5, 100),
        })
        y_demo = np.random.randint(0, 2, 100)

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("model", LogisticRegression(max_iter=500))
        ])
        pipe.fit(X_demo, y_demo)
        return pipe


    DEMO_WATER = get_demo_water_model()

    # =========================
    # Streamlit UI
    # =========================
    st.markdown("<h2>💧 Water Quality Prediction</h2>", unsafe_allow_html=True)
    st.write("Use demo model or upload your own sklearn pipeline (.pkl).")

    with st.expander("Model Source", expanded=True):
        model_type2 = st.radio(
            "Select Model", ["Demo (built-in)", "Upload .pkl"],
            horizontal=True, key="water_model_source"
        )
        user_water_model = None
        if model_type2 == "Upload .pkl":
            uploaded_water = st.file_uploader(
                "Upload scikit-learn Pipeline (.pkl)", type=["pkl", "joblib"], key="water_pkl"
            )
            if uploaded_water:
                user_water_model = load_pickle(uploaded_water)
        water_model = user_water_model if user_water_model else DEMO_WATER

    # =========================
    # Single sample assessment
    # =========================
    st.subheader("🔹 Single Sample Test")
    col1, col2, col3 = st.columns(3)
    ph = col1.number_input("pH", 0.0, 14.0, 7.0, step=0.1)
    hardness = col1.number_input("Hardness", 0.0, 1000.0, 200.0)
    solids = col1.number_input("Solids", 0.0, 100000.0, 20000.0)
    chloramines = col2.number_input("Chloramines", 0.0, 20.0, 7.0, step=0.1)
    sulfate = col2.number_input("Sulfate", 0.0, 800.0, 330.0)
    conductivity = col2.number_input("Conductivity", 0.0, 2000.0, 425.0)
    organic_carbon = col3.number_input("Organic Carbon", 0.0, 50.0, 10.0)
    trihalomethanes = col3.number_input("Trihalomethanes", 0.0, 200.0, 70.0)
    turbidity = col3.number_input("Turbidity", 0.0, 20.0, 3.0)

    sample_df = pd.DataFrame([{
        "ph": ph,
        "Hardness": hardness,
        "Solids": solids,
        "Chloramines": chloramines,
        "Sulfate": sulfate,
        "Conductivity": conductivity,
        "Organic_carbon": organic_carbon,
        "Trihalomethanes": trihalomethanes,
        "Turbidity": turbidity,
    }])

    if st.button("Assess Potability"):
        sample_df = sample_df.fillna(sample_df.mean())  # handle NaN
        preds = safe_predict(water_model, sample_df)
        if preds is not None:
            try:
                proba = water_model.predict_proba(sample_df)[:, 1][0]
            except:
                proba = None
            label = int(np.rint(preds[0]))
            verdict = "Potable ✅" if label == 1 else "Not Potable ❌"
            if proba is not None:
                st.success(f"Prediction: **{verdict}** — Confidence: {proba:.2%}")
            else:
                st.success(f"Prediction: **{verdict}**")

    # =========================
    # Batch CSV predictions
    # =========================
    st.divider()
    st.subheader("📂 Batch CSV Predictions")
    csv_water = st.file_uploader("Upload water samples CSV", type=["csv"], key="water_csv")
    if csv_water:
        try:
            df_water_csv = pd.read_csv(csv_water)
            st.dataframe(df_water_csv.head(20))

            if st.button("Run Batch Water Predictions"):
                df_water_csv = df_water_csv.fillna(df_water_csv.mean())  # handle NaN
                preds = safe_predict(water_model, df_water_csv.drop(columns=["Potability"], errors="ignore"))
                if preds is not None:
                    out_df = df_water_csv.copy()
                    out_df["potable_pred"] = np.rint(preds).astype(int)
                    try:
                        probs = water_model.predict_proba(
                            df_water_csv.drop(columns=["Potability"], errors="ignore")
                        )[:, 1]
                        out_df["potable_confidence"] = probs
                    except:
                        pass
                    st.success("Predictions completed ✅")
                    st.dataframe(out_df.head(50))
                    st.download_button(
                        "Download Predictions CSV",
                        out_df.to_csv(index=False).encode("utf-8"),
                        "water_predictions.csv"
                    )
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    import streamlit as st
    import pandas as pd
    import numpy as np
    from typing import List, Tuple, Optional
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score, roc_curve,
        confusion_matrix, precision_score, recall_score,
        r2_score, mean_squared_error, mean_absolute_error
    )
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.inspection import partial_dependence
    import altair as alt

    # ----------------- Page Config ------------------
    st.set_page_config(page_title="💧 Water Quality Prediction", layout="wide")
    st.title("💧 Water Quality Prediction System — Interactive Graphs")
    st.caption(
        "Upload data, train a model, and explore interactive charts for feature importance, ROC, confusion matrix, and more.")


    # --------------- Helper Functions ---------------
    @st.cache_data(show_spinner=False)
    def read_csv(file) -> pd.DataFrame:
        return pd.read_csv(file)


    def split_features_target(df: pd.DataFrame, target_col: Optional[str]) -> Tuple[pd.DataFrame, pd.Series]:
        if target_col is None:
            # Heuristics
            for guess in ["Potability", "potability", "WQI", "wqi", "Target", "target"]:
                if guess in df.columns:
                    target_col = guess
                    break
        if target_col is None or target_col not in df.columns:
            raise ValueError("Target column not found. Choose it in the sidebar.")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return X, y


    def infer_task(y: pd.Series) -> str:
        # classification if binary / few integers; regression otherwise
        if y.dtype.kind in "biu" and y.nunique() <= 10:
            return "classification"
        if y.dtype.kind in "f" and y.dropna().nunique() <= 5:
            # floats but few unique values (likely 0/1 as float)
            return "classification"
        return "regression"


    def build_pipeline(task: str, numeric: List[str], categorical: List[str]):
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        pre = ColumnTransformer([
            ("num", num_pipe, numeric),
            ("cat", cat_pipe, categorical),
        ])
        if task == "classification":
            model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
        else:
            model = RandomForestRegressor(n_estimators=300, random_state=42)
        pipe = Pipeline([
            ("pre", pre),
            ("model", model),
        ])
        return pipe


    def feature_importance_chart(feature_names: List[str], importances: np.ndarray, top_n: int = 15):
        df_imp = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance",
                                                                                                 ascending=False).head(
            top_n)
        chart = alt.Chart(df_imp).mark_bar().encode(
            x=alt.X("importance:Q", title="Importance"),
            y=alt.Y("feature:N", sort='-x', title="Feature"),
            tooltip=["feature", alt.Tooltip("importance", format=".4f")]
        ).properties(height=28 * min(top_n, len(df_imp)), title="Top Feature Importances")
        return chart


    def roc_curve_chart(y_true: np.ndarray, y_prob: np.ndarray):
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr, "Threshold": thr})
        auc = roc_auc_score(y_true, y_prob)
        line = alt.Chart(roc_df).mark_line().encode(x="FPR", y="TPR", tooltip=["FPR", "TPR", "Threshold"]).properties(
            title=f"ROC Curve (AUC = {auc:.3f})")
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4]).encode(x="x", y="y")
        return line + diag


    def confusion_matrix_chart(y_true: np.ndarray, y_pred: np.ndarray):
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, columns=["Pred 0", "Pred 1"]) if cm.shape == (2, 2) else pd.DataFrame(cm)
        cm_melt = cm_df.reset_index().melt(id_vars='index', var_name='Pred', value_name='Count')
        cm_melt.rename(columns={'index': 'True'}, inplace=True)
        chart = alt.Chart(cm_melt).mark_rect().encode(
            x=alt.X('Pred:N', title='Predicted'),
            y=alt.Y('True:N', title='Actual'),
            color=alt.Color('Count:Q'),
            tooltip=['True', 'Pred', 'Count']
        ).properties(title="Confusion Matrix")
        text = alt.Chart(cm_melt).mark_text(baseline='middle').encode(
            x='Pred:N', y='True:N', text='Count:Q'
        )
        return chart + text


    def pred_vs_actual_chart(y_true: np.ndarray, y_pred: np.ndarray):
        dfp = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
        pts = alt.Chart(dfp).mark_circle(size=60, opacity=0.6).encode(
            x="Actual:Q", y="Predicted:Q", tooltip=["Actual", "Predicted"]
        ).properties(title="Predicted vs Actual")
        line = alt.Chart(pd.DataFrame({"x": [dfp["Actual"].min(), dfp["Actual"].max()]})).transform_calculate(
            y='datum.x').mark_line().encode(x='x:Q', y='y:Q')
        return pts + line


    def correlation_heatmap(df: pd.DataFrame):
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return None
        corr = numeric_df.corr(numeric_only=True).stack().reset_index()
        corr.columns = ['Feature X', 'Feature Y', 'Correlation']
        chart = alt.Chart(corr).mark_rect().encode(
            x=alt.X('Feature X:N', sort=None),
            y=alt.Y('Feature Y:N', sort=None),
            color=alt.Color('Correlation:Q', scale=alt.Scale(domain=(-1, 1))),
            tooltip=['Feature X', 'Feature Y', alt.Tooltip('Correlation:Q', format='.2f')]
        ).properties(title='Correlation Heatmap')
        return chart


    def partial_dependence_chart(pipe: Pipeline, X: pd.DataFrame, feature_name: str, task: str):
        try:
            pdp = partial_dependence(pipe, X=X, features=[feature_name], kind='average')
            xs = pdp['values'][0]
            ys = pdp['average'][0]
            dfp = pd.DataFrame({feature_name: xs, 'Effect': ys})
            title = f"Partial Dependence — {feature_name}"
            if task == 'classification':
                title += " (on class 1 prob)"
            return alt.Chart(dfp).mark_line().encode(x=f"{feature_name}:Q", y="Effect:Q",
                                                     tooltip=[feature_name, 'Effect']).properties(title=title)
        except Exception as e:
            st.info(f"Partial dependence not available for {feature_name}: {e}")
            return None


    # ---------------- Sidebar Controls -----------------
    st.sidebar.header("⚙️ Setup")
    uploaded = st.sidebar.file_uploader("Upload CSV (water quality)", type=["csv"])

    if uploaded is None:
        st.info(
            "Upload a dataset to begin. Tip: Works with UCI Water Potability or any dataset with a target like 'Potability' or 'WQI'.")
        st.stop()

    with st.spinner("Reading data..."):
        df = read_csv(uploaded)

    st.sidebar.subheader("Target & Split")
    all_cols = list(df.columns)
    default_target = "Potability" if "Potability" in all_cols else ("WQI" if "WQI" in all_cols else None)
    target_col = st.sidebar.selectbox("Select target column", options=[None] + all_cols,
                                      index=(0 if default_target is None else all_cols.index(default_target) + 1))

    try:
        X, y = split_features_target(df, target_col)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Identify column types
    numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    st.sidebar.write("**Detected**")
    st.sidebar.write(f"Numeric: {len(numeric_cols)} · Categorical: {len(categorical_cols)}")

    # Task inference and override
    inferred_task = infer_task(y)
    task = st.sidebar.radio("Task", options=["classification", "regression"],
                            index=(0 if inferred_task == "classification" else 1), help="Override if needed.")

    # Split settings
    size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    random_state = st.sidebar.number_input("Random state", min_value=0, value=42, step=1)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=random_state,
                                                        stratify=(y if task == "classification" else None))

    # Build & train
    pipe = build_pipeline(task, numeric_cols, categorical_cols)
    with st.spinner("Training model..."):
        pipe.fit(X_train, y_train)

    # Predictions
    if task == "classification":
        y_prob = pipe.predict_proba(X_test)[:, 1]
        default_threshold = 0.5
        threshold = st.slider("Decision threshold", 0.05, 0.95, default_threshold, 0.01)
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = pipe.predict(X_test)

    # --------------- Layout -----------------
    left, right = st.columns([1, 1])

    with left:
        st.subheader("📊 Key Metrics")
        if task == "classification":
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_prob)
            st.metric("Accuracy", f"{acc:.3f}")
            st.metric("F1-score", f"{f1:.3f}")
            st.metric("Precision", f"{prec:.3f}")
            st.metric("Recall", f"{rec:.3f}")
            st.metric("ROC AUC", f"{auc:.3f}")
        else:
            r2 = r2_score(y_test, y_pred)
            # sklearn compatibility: some versions don't support 'squared' kwarg
            try:
                rmse = mean_squared_error(y_test, y_pred, squared=False)
            except TypeError:
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            st.metric("R²", f"{r2:.3f}")
            st.metric("RMSE", f"{rmse:.3f}")
            st.metric("MAE", f"{mae:.3f}")

    with right:
        st.subheader("🧪 Columns Detected")
        st.dataframe(pd.DataFrame(
            {"Column": X.columns, "Type": ["Numeric" if c in numeric_cols else "Categorical" for c in X.columns]}))

    st.markdown("---")

    # ----------------- Charts -------------------
    chart_cols = st.columns([1, 1])

    # Feature Importances
    try:
        # Get feature names after preprocessing
        model = pipe.named_steps["model"]
        pre = pipe.named_steps["pre"]
        # numeric names
        num_names = numeric_cols
        # categorical names after one-hot
        cat_encoder = pre.named_transformers_["cat"].named_steps.get("onehot", None) if len(
            categorical_cols) > 0 else None
        if cat_encoder is not None:
            cat_names = list(cat_encoder.get_feature_names_out(categorical_cols))
        else:
            cat_names = []
        all_feature_names = num_names + cat_names

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            with chart_cols[0]:
                st.subheader("🌟 Feature Importance")
                st.altair_chart(feature_importance_chart(all_feature_names, importances), use_container_width=True)
    except Exception as e:
        st.info(f"Feature importance unavailable: {e}")

    # ROC / Pred vs Actual
    with chart_cols[1]:
        if task == "classification":
            st.subheader("📈 ROC Curve")
            st.altair_chart(roc_curve_chart(y_test.to_numpy(), y_prob), use_container_width=True)
        else:
            st.subheader("🎯 Predicted vs Actual")
            st.altair_chart(pred_vs_actual_chart(y_test.to_numpy(), y_pred), use_container_width=True)

    # Confusion Matrix or Residuals
    other_cols = st.columns([1, 1])

    with other_cols[0]:
        if task == "classification":
            st.subheader("🧮 Confusion Matrix")
            st.altair_chart(confusion_matrix_chart(y_test.to_numpy(), y_pred), use_container_width=True)
        else:
            st.subheader("📉 Residual Distribution")
            residuals = y_test - y_pred
            df_res = pd.DataFrame({"Residual": residuals})
            hist = alt.Chart(df_res).mark_bar().encode(x=alt.X("Residual:Q", bin=True), y='count()').properties(
                title="Residuals")
            st.altair_chart(hist, use_container_width=True)

    with other_cols[1]:
        st.subheader("🔗 Correlation Heatmap")
        heat = correlation_heatmap(df)
        if heat is not None:
            st.altair_chart(heat, use_container_width=True)
        else:
            st.info("Need at least two numeric columns for a correlation heatmap.")

    # ----------------- Partial Dependence (Optional) -----------------
    st.subheader("🪄 Partial Dependence (Top 2 features)")
    try:
        if hasattr(pipe.named_steps["model"], "feature_importances_") and len(all_feature_names) >= 1:
            top_idx = np.argsort(importances)[::-1][:2]
            top_feats = [all_feature_names[i] for i in top_idx]
            pd_cols = st.columns(len(top_feats))
            for i, feat in enumerate(top_feats):
                chart = partial_dependence_chart(pipe, X_test, feat, task)
                if chart is not None:
                    with pd_cols[i]:
                        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.info(f"Partial dependence skipped: {e}")

    # ----------------- Single Prediction Sandbox -----------------
    st.markdown("---")
    st.subheader("🧪 Try a Single Prediction")
    st.caption("Fill values for a single sample and see the model's prediction.")

    # Build simple UI for the first up-to-8 columns
    sample_vals = {}
    for col in X.columns:
        if col in numeric_cols:
            val = st.number_input(
                f"{col}",
                value=float(np.nanmean(pd.to_numeric(X[col], errors='coerce').dropna()))
                if pd.to_numeric(X[col], errors='coerce').notna().any()
                else 0.0
            )
        else:
            choices = sorted([str(x) for x in X[col].dropna().unique()])[:25]
            default = choices[0] if choices else ""
            val = st.selectbox(f"{col}", options=choices if choices else [default])
        sample_vals[col] = val

    if st.button("Predict Single Sample", type="primary"):
        sample_df = pd.DataFrame([sample_vals])

        # 🔑 Ensure all training columns are present
        missing_cols = set(X.columns) - set(sample_df.columns)
        for col in missing_cols:
            sample_df[col] = 0  # default fill

        # Reorder columns to match training data
        sample_df = sample_df[X.columns]

        try:
            if task == "classification":
                prob = pipe.predict_proba(sample_df)[:, 1][0]
                pred = int(prob >= threshold)
                st.success(f"Predicted class: {pred} (probability of class 1 = {prob:.3f})")
            else:
                pred = pipe.predict(sample_df)[0]
                st.success(f"Predicted value: {pred:.3f}")

            # Optional: warn if columns were auto-filled
            if missing_cols:
                st.warning(f"Auto-filled missing columns with 0: {', '.join(missing_cols)}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # ----------------- Footer -----------------
    st.markdown(
        """
        <small>Tips: For potability datasets, select **classification** with target **Potability** (0/1). For water quality index (WQI), choose **regression** with target **WQI**. All charts update live as you change the threshold or retrain.</small>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------
# Team Section with edit button
# ---------------------------
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO

def image_circle(img_source, size=120):
    try:
        if img_source:
            # If it's a URL
            if isinstance(img_source, str) and img_source.startswith("http"):
                response = requests.get(img_source, timeout=5)
                image = Image.open(BytesIO(response.content))
            else:  # If it's an uploaded file
                image = Image.open(img_source)
        else:
            raise Exception("No image provided")
    except Exception:
        # Create a local placeholder instead of fetching online
        image = Image.new("RGB", (size, size), color=(260, 260, 260))
        draw = ImageDraw.Draw(image)
        draw.text((size//4, size//3), "N/A", fill=(150, 150, 150))

    # Resize and mask into circle
    image = image.resize((size, size))
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size, size), fill=455)
    output = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    output.paste(image, (0, 0), mask)
    return output



if choice == "Team":   # ✅ Correct start of Team section
    st.markdown(f"<div class='big-title'>Meet the <span style='color:blue'>Team</span></div>", unsafe_allow_html=True)
    st.markdown("""
    <h2 style='text-align: center; 
               font-size: 36px; 
               background: -webkit-linear-gradient(45deg, #FF6B6B, #5F27CD); 
               -webkit-background-clip: text; 
               -webkit-text-fill-color: transparent;'>
    ✨ Meet Our Amazing Team ✨
    </h2>
    """, unsafe_allow_html=True)

    if "team" not in st.session_state:
        st.session_state.team = [
            {"name": "Vishesh Kumar Prajapati", "role": "Founder / ML Engineer",
             "bio": "Passionate ML Engineer building intelligent systems and dashboards.",
             "email": "visheshprajapati7920@gmail.com",
             "linkedin": "https://www.linkedin.com/in/vishesh-kumar-prajapati-45111829a",
             "github": "https://github.com/vishes-i",
             "photo_url": "https://d2gg9evh47fn9z.cloudfront.net/1600px_COLOURBOX37232552.jpg"},
            {"name": "Sumit Yadav", "role": "Web Developer",
             "bio": "Frontend & backend developer who loves crafting responsive web apps.",
             "email": "sy2902913@gmail.com",
             "linkedin": "https://www.linkedin.com/in/sumit-yadav-3b93a92a9",
             "github": "https://github.com/",
             "photo_url": "https://d2gg9evh47fn9z.cloudfront.net/1600px_COLOURBOX37236066.jpg"},
            {"name": "Tejashwani Singh Rathore ", "role": "Web Developer",
             "bio": "Frontend developer who loves crafting responsive web apps.",
             "email": "tejaswanirathore910@gmail.com ",
             "linkedin": "https://www.linkedin.com/in/tejaswanirathore-3b93a92a9",
             "github": "https://github.com/",
             "photo_url": "https://img.freepik.com/premium-photo/young-girl-hr-3d-character-young-working-girl-cartoon-character-professional-girl-character_1002350-2145.jpg?w=2000"},
            {"name": "Vijay Kharwar", "role": "Web Developer",
             "bio": "Frontend developer who loves crafting responsive web apps.",
             "email": "vijaykharwargzp2003@gmail.com",
             "linkedin": "https://www.linkedin.com/in/vijay-kharwar-b290aa2ab?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app",
             "github": "https://github.com/vijaykharwargzp2003-coder",
             "photo_url": "https://img.freepik.com/free-vector/confident-businessman-with-smile_1308-134106.jpg"}

        ]

    # ---- Display Team ----
    for idx, member in enumerate(st.session_state.team):
        with st.container():
            col1, col2 = st.columns([1, 4])

            with col1:
                st.image(image_circle(member.get("photo_url") or member.get("photo")), use_container_width=False)

            with col2:
                st.subheader(member["name"])
                st.caption(member["role"])
                st.write(member.get("bio", "No biography added yet."))

                c1, c2, c3 = st.columns(3)
                with c1:
                    if member.get("email"):
                        st.link_button("📧 Email", f"mailto:{member['email']}")
                with c2:
                    if member.get("linkedin"):
                        st.link_button("💼 LinkedIn", member["linkedin"])
                with c3:
                    if member.get("github"):
                        st.link_button("💻 GitHub", member["github"])

                # Edit Section
                with st.expander("✏️ Edit Member"):
                    with st.form(f"edit_form_{idx}"):
                        member["name"] = st.text_input("Name", member["name"])
                        member["role"] = st.text_input("Role", member["role"])
                        member["bio"] = st.text_area("Biography", member.get("bio", ""))
                        member["email"] = st.text_input("Email", member["email"])
                        member["linkedin"] = st.text_input("LinkedIn", member["linkedin"])
                        member["github"] = st.text_input("GitHub", member["github"])
                        member["photo_url"] = st.text_input("Photo URL", member.get("photo_url", ""))
                        new_photo = st.file_uploader("Upload Photo", type=["png", "jpg", "jpeg"], key=f"photo_{idx}")
                        update = st.form_submit_button("✅ Update")
                        if update:
                            if new_photo:
                                member["photo"] = new_photo
                            st.success(f"Updated {member['name']}")

    # ---- Add New Member ----
    with st.expander("➕ Add New Member"):
        with st.form("add_member_form"):
            st.subheader("Add a New Team Member")
            name = st.text_input("Name")
            role = st.text_input("Role")
            bio = st.text_area("Biography")
            email = st.text_input("Email")
            linkedin = st.text_input("LinkedIn")
            github = st.text_input("GitHub")
            photo_url = st.text_input("Photo URL")
            photo = st.file_uploader("Upload Photo", type=["png", "jpg", "jpeg"], key="new_photo")
            add = st.form_submit_button("Add Member")
            if add and name and role:
                new_member = {"name": name, "role": role, "bio": bio, "email": email,
                              "linkedin": linkedin, "github": github, "photo_url": photo_url, "photo": photo}
                st.session_state.team.append(new_member)
                st.success(f"Added {name} to the team!")

    # ---- Delete Member ----
    with st.form("delete_member_form"):
        st.subheader("🗑️ Delete a Member")
        names = [m['name'] for m in st.session_state.team]
        delete_name = st.selectbox("Select member to delete", [""] + names)
        delete = st.form_submit_button("Delete")
        if delete and delete_name:
            st.session_state.team = [m for m in st.session_state.team if m['name'] != delete_name]
            st.success(f"Deleted {delete_name}")

    # ---- Export ----
    st.download_button(
        "📥 Export Team JSON",
        data=json.dumps(st.session_state.team, indent=2, default=str).encode("utf-8"),
        file_name="team.json"
    )

# ---------------------------
# Additional: World Map Visualization (for Home or other pages)
# ---------------------------
if choice == "Home" or choice == "Portfolio":
    # Example: display a world map with random data points
    import pydeck as pdk

    df_map = pd.DataFrame({
        'lat': np.random.uniform(-60, 60, size=50),
        'lon': np.random.uniform(-180, 180, size=50),
        'value': np.random.rand(50)
    })
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=0,
            longitude=0,
            zoom=1,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=df_map,
                get_position='[lon, lat]',
                get_color='[255, 0, 0, 160]',
                get_radius='value * 10000',
            ),
        ],
    ))

# ---------------------------
# End of app
# ---------------------------
st.info("Tip: You can upload your own models for production. Ensure they include all preprocessing.")
