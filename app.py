import streamlit as st
import numpy as np
import pickle
import pandas as pd
import math

# -----------------------------
# Widen the main content area
# -----------------------------
st.markdown("""
<style>

/* --- widen main content area --- */
.block-container {
    max-width: 1400px;      /* adjust to your preference */
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-top: 1rem !important;   /* trim top whitespace */
}

/* Optional: widen the sidebar */
[data-testid="stSidebar"] {
    width: 320px !important;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>

/* Desktop-only scaling */
@media (min-width: 900px) {

    /* Increase all normal text */
    html, body, p, span, div, [class*="stMarkdown"], [class*="css"] {
        font-size: 1.12rem !important;   /* <= this is the sweet spot */
        line-height: 1.45 !important;
    }

    /* Slightly larger input labels */
    label, .stNumberInput label, .stTextInput label {
        font-size: 1.10rem !important;
    }

    /* Keep headers proportional (do NOT overgrow them) */
    h1 { font-size: 2.1rem !important; }
    h2 { font-size: 1.55rem !important; }
    h3 { font-size: 1.28rem !important; }
}

</style>
""", unsafe_allow_html=True)


# Baseline event rate (for contextualizing predictions)
BASELINE_RATE = 0.15

# Risk strata for human-readable grouping
RISK_BINS = [
    ("Low", 0.0, 0.10, "#cfd8dc"),           # light gray
    ("Moderate", 0.10, 0.25, "#ffcc80"), # soft amber
    ("High", 0.25, 0.40, "#ff8a65"),         # orange-red
    ("Very high", 0.40, 1.0, "#d32f2f"),     # deep red
]

# Ticks to annotate compartment borders (fractions of 1.0)
BOUNDARY_TICKS = [0.10, 0.25, 0.40]


def categorize_risk(p):
    """Return (label, color) for a probability between 0 and 1."""
    for label, lo, hi, color in RISK_BINS:
        if lo <= p < hi:
            return label, color
    # Fallback if outside [0,1]
    return "Uncategorized", "#7391f5"


# ------------------------
# Model and Scaler Loaders
# ------------------------
@st.cache_resource
def load_model():
    """Load the trained machine learning model from disk."""
    # OPTIMIZATION: Implicitly loads xgboost only when this function is called
    with open("model.pkl", "rb") as f:
        return pickle.load(f)
    return None

# xgboost handles NaNs natively; imputer not strictly needed
# @st.cache_resource
# def load_imputer():
#     """Load the fitted imputer from disk."""
#     with open("imputer.pkl", "rb") as f:
#         return pickle.load(f)
#     return None


@st.cache_data
def load_calibration_data(path: str):
    """Load calibration CSV ensuring required columns are present."""
    df = pd.read_csv(path)
    required_cols = {"Age", "Calibrated_prob", "Calibrated_prob_mean", "q30"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in calibration file: {', '.join(sorted(missing_cols))}")
    return df

# ------------------------
# Session state for resettable fields
# ------------------------
default_values = dict(age="", cci="", pbs="", sofa="")
fallback_defaults = dict(age=63, cci=2, pbs=0, sofa=0)  # used when fields are left blank; we used cci=2 because that's the cci for a 63-year-old person with no comorbidities
for k, v in default_values.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_form():
    """Reset all user-editable fields to defaults."""
    for k, v in default_values.items():
        st.session_state[k] = v


# ------------------------
# Sidebar: Branding, About, Disclaimer
# ------------------------
with st.sidebar:
    st.image("LRG.png", width=120)
    st.markdown(
        "<div style='margin-top: -10px; font-size:1.05em; color: #666;'>"
        "Developed at <a href='https://www.lewisresearchgroup.org/' target='_blank' "
        "style='color: #7391f5; text-decoration: none;'>Lewis Research Group</a>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.header("About this Tool")
    st.write(
        "This tool provides a model-based estimate of 30-day mortality probability. "
        "It uses four inputs: Age, Charlson Comorbidity Index (CCI), Pitt Bacteremia Score (PBS), "
        "and SOFA score to generate a probability derived from a machine learning model trained "
        "on historical data. "
        "This estimate should be interpreted as a statistical output, not a clinical determination. "
        "For background on these indices: "
        "[CCI](https://www.mdcalc.com/calc/3917/charlson-comorbidity-index-cci), "
        "[PBS](https://m.medicalalgorithms.com/pitt-bacteremia-score-of-paterson-et-al), "
        "[SOFA](https://www.mdcalc.com/calc/691/sequential-organ-failure-assessment-sofa-score)."
    )

# ------------------------
# Main App Interface
# ------------------------
st.title("Model-Based Estimate of 30-Day Mortality Probability")
st.header("Enter Patient Data")

# Tooltips for each feature
cci_info = (
    "Charlson Comorbidity Index (CCI) is a widely used scoring system "
    "that predicts ten-year mortality based on the presence of comorbidity conditions. "
    "[Learn more.](https://www.mdcalc.com/calc/3917/charlson-comorbidity-index-cci)"
)
pbs_info = (
    "Pitt Bacteremia Score (PBS) is a tool used in infectious disease research to assess the severity of acute "
    "illness and predict mortality. "
    "[Learn more.](https://m.medicalalgorithms.com/pitt-bacteremia-score-of-paterson-et-al)"
)
sofa_info = (
    "SOFA (Sequential Organ Failure Assessment) score quantifies the extent of a patient's organ function or rate of "
    "failure. [Learn more.](https://www.mdcalc.com/calc/691/sequential-organ-failure-assessment-sofa-score)"
)
age_info = "Patient's age in years."

# Feature inputs (linked to session_state for reset functionality)
age = st.text_input("Age (years)", key="age", placeholder="A number between 0 and 100. Default: 63", help=age_info)
cci = st.text_input("Charlson Comorbidity Index (CCI)", key="cci", placeholder="A number between 0 and 17. Default: 2", help=cci_info)
pbs = st.text_input("PBS Score", key="pbs", placeholder="A number between 0 and 14. Default: 0", help=pbs_info)
sofa = st.text_input("SOFA Score", key="sofa", placeholder="A number between 0 and 24. Default: 0", help=sofa_info)

# --------------------------------
# Training ranges (from your dataset)
# --------------------------------
TRAINING_RANGES = {
    "age": {"min": 0, "max": 100},
    "cci": {"min": 0, "max": 17},
    "pbs": {"min": 0, "max": 14},
    "sofa": {"min": 0, "max": 24},
}

# Parse user text inputs; fall back to default scenario if blank/unparseable
def parse_num(val):
    """Return (parsed_value, reason) where reason is None when parsing succeeds."""
    try:
        if val is None:
            return None, "blank"
        stripped = str(val).strip()
        if stripped == "":
            return None, "blank"
        num = float(stripped)
        if not math.isfinite(num):
            return None, "non-finite"
        return num, None
    except (ValueError, TypeError, OverflowError):
        return None, "not-a-number"


# Store parsed values in a dictionary for easier fallback handling
parsed_vals = {
    "age": None,
    "cci": None,
    "pbs": None,
    "sofa": None,
}
parse_reasons = {}

for field_key, raw_val in [("age", age), ("cci", cci), ("pbs", pbs), ("sofa", sofa)]:
    parsed, reason = parse_num(raw_val)
    parsed_vals[field_key] = parsed
    parse_reasons[field_key] = reason

fallback_used = []
fallback_invalid = []
reason_labels = {
    "blank": "left blank",
    "not-a-number": "not a number",
    "non-finite": "not a finite number",
}
for field in ["age", "cci", "pbs", "sofa"]:
    if parsed_vals[field] is None:
        parsed_vals[field] = fallback_defaults[field]
        reason = parse_reasons.get(field)
        reason_label = reason_labels.get(reason, "invalid input")
        entry = f"{field.upper()}: {parsed_vals[field]} ({reason_label})"
        fallback_used.append(entry)
        if reason != "blank":
            fallback_invalid.append(entry)

# Unpack back to individual variables for downstream code
age_val = parsed_vals["age"]
cci_val = parsed_vals["cci"]
pbs_val = parsed_vals["pbs"]
sofa_val = parsed_vals["sofa"]
violations = []
if age_val < TRAINING_RANGES["age"]["min"] or age_val > TRAINING_RANGES["age"]["max"]:
    violations.append(f"Age={age_val} (trained range: {TRAINING_RANGES['age']['min']}-{TRAINING_RANGES['age']['max']})")
if cci_val < TRAINING_RANGES["cci"]["min"] or cci_val > TRAINING_RANGES["cci"]["max"]:
    violations.append(f"CCI={cci_val} (trained range: {TRAINING_RANGES['cci']['min']}-{TRAINING_RANGES['cci']['max']})")
if pbs_val < TRAINING_RANGES["pbs"]["min"] or pbs_val > TRAINING_RANGES["pbs"]["max"]:
    violations.append(f"PBS={pbs_val} (trained range: {TRAINING_RANGES['pbs']['min']}-{TRAINING_RANGES['pbs']['max']})")
if sofa_val < TRAINING_RANGES["sofa"]["min"] or sofa_val > TRAINING_RANGES["sofa"]["max"]:
    violations.append(f"SOFA={sofa_val} (trained range: {TRAINING_RANGES['sofa']['min']}-{TRAINING_RANGES['sofa']['max']})")

if violations:
    st.warning(
        "⚠️ One or more inputs are outside the range observed in the model's training data:\n\n"
        + "\n".join([f"- {v}" for v in violations])
        + "\n\nPredictions in this region may be less reliable."
    )
if fallback_invalid:
    st.warning(
        "⚠️ Some entries were invalid and replaced with defaults:\n\n"
        + "\n".join([f"- {entry}" for entry in fallback_invalid])
    )
elif fallback_used:
    st.caption(f"Using default values for empty fields: {', '.join(fallback_used)}. You can enter values to override these defaults.")

# Predict and Reset buttons side-by-side
col1, col2 = st.columns([1, 1])
with col1:
    predict = st.button("Estimate Probability")
with col2:
    reset = st.button("Reset", on_click=reset_form)

# ------------------------
# Prediction & Results
# ------------------------
if predict:
    # Anchor to scroll results into view after clicking the button
    st.markdown("<div id='result-section' style='height:1px;'></div>", unsafe_allow_html=True)

    # OPTIMIZATION: Load models/data ONLY on click
    model = load_model()
    calibration_df = load_calibration_data("calibration_data.csv")
    calibration_available = calibration_df is not None and not calibration_df.empty
    age_mean = age_q30 = matched_age = None
    age_note = None

    # Prepare and scale features for prediction
    X = pd.DataFrame(np.array([[age_val, cci_val, pbs_val, sofa_val]]),
                     columns=['AHS: NAGE_YR', 'COMORB: Charlson_WIC', 'S.SCORE: PBS', 'S.SCORE: SOFA'])

    # Predict mortality probability
    proba = model.predict_proba(X)[0][1]
    relative = proba / BASELINE_RATE if BASELINE_RATE > 0 else float("nan")
    risk_label, bar_color = categorize_risk(proba)

    # Pull calibration comparison numbers up front
    if calibration_available:
        plot_df = calibration_df.sort_values("Age")
        age_match = plot_df[plot_df["Age"] == age_val]
        if age_match.empty:
            nearest_idx = (plot_df["Age"] - age_val).abs().idxmin()
            age_mean = float(plot_df.loc[nearest_idx, "Calibrated_prob_mean"])
            age_q30 = float(plot_df.loc[nearest_idx, "q30"])
            matched_age = float(plot_df.loc[nearest_idx, "Age"])
            age_note = f"No exact age match; using nearest age in data: {matched_age:.0f}."
        else:
            age_mean = float(age_match["Calibrated_prob_mean"].mean())
            age_q30 = float(age_match["q30"].mean())
            matched_age = age_val

        delta_pct_points = (proba - age_mean) * 100
        direction = "higher" if delta_pct_points >= 0 else "lower"
        delta_q30 = (proba - age_q30) * 100
        direction_q30 = "higher" if delta_q30 >= 0 else "lower"

    # Display probability
    st.write("### Estimated Probability")
    left_col, spacer_col, right_col = st.columns([1.05, 0.08, 1])

    with left_col:
        # Smoothly scroll to this section after prediction
        st.markdown(
            """
            <script>
            setTimeout(() => {
                let el = document.getElementById('result-section');
                if (!el) {
                    // Fallback: Look for the header in the parent document (common in Streamlit iframes)
                    try {
                        const headers = window.parent.document.querySelectorAll('h3');
                        for (const header of headers) {
                            if (header.textContent.includes('Estimated Probability')) {
                                el = header;
                                break;
                            }
                        }
                    } catch (e) { console.log('Access to parent blocked'); }
                }
                if (!el) {
                    // Fallback: Look in current document
                    const headers = document.querySelectorAll('h3');
                    for (const header of headers) {
                        if (header.textContent.includes('Estimated Probability')) {
                            el = header;
                            break;
                        }
                    }
                }
                if (el) { el.scrollIntoView({ behavior: 'smooth', block: 'center' }); }
            }, 300);
            </script>
            """,
            unsafe_allow_html=True,
        )

        if calibration_available:
            st.markdown(
                f"""
                **Model-estimated 30-day mortality probability is {proba * 100:.1f}%**, which is:

                - {abs(delta_pct_points):.1f} percentage points {direction} than the age-matched mean in the BSI cohort ({age_mean*100:.1f}%)
                - {abs(delta_q30):.1f} percentage points {direction_q30} than the age-matched US 30-day mortality baseline ({age_q30*100:.1f}%)
                """
            )
        else:
            st.info(f"Model-estimated 30-day mortality probability is **{proba * 100:.1f}%**.")

        # Segmented severity bar (four compartments with marker)
        marker_left = min(max(proba * 100, 0), 100)
        segments_html = "".join(
            f"<div style='flex: {hi - lo}; background: {color}; height: 100%;'></div>"
            for _, lo, hi, color in RISK_BINS
        )

        ticks_html = "".join(
            f"""
            <div style='position: absolute; left: {tick*100}%; top: 48px; transform: translateX(-50%); width: 1px; height: 10px; background: #666;'></div>
            <div style='position: absolute; left: {tick*100}%; top: 60px; transform: translateX(-50%); font-size: 11px; color: #444;'>
                {tick:.2f}
            </div>
            """
            for tick in BOUNDARY_TICKS
        )

        st.markdown(
            f"""
            <div style='width: 100%; position: relative; margin: 8px 0 72px 0;'>
              <div style='display: flex; border-radius: 22px; overflow: hidden; border: 1px solid #e0e0e0; height: 40px;'>
                {segments_html}
              </div>
              
              <div style='position: absolute; left: {BASELINE_RATE*100}%; top: 44px; transform: translateX(-50%); display: flex; flex-direction: column; align-items: center; gap: 4px; pointer-events: none;'>
              </div>

              <div style='position: absolute; left: {marker_left}%; top: -8px; transform: translateX(-50%); display: flex; flex-direction: column; align-items: center; gap: 2px;'>
                <div style='width: 0; height: 0; border-left: 6px solid transparent; border-right: 6px solid transparent; border-bottom: 8px solid #000;'></div>
                <div style='font-size: 11px; color: #000; white-space: nowrap; background: rgba(255,255,255,0.85); padding: 0 2px; border-radius: 3px;'>{proba * 100:.1f}%</div>
              </div>
              {ticks_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style='margin: 30px 0 6px 0; padding: 10px 12px; border-radius: 10px; background: #eef6ff; border: 1px solid #d9e7ff;'>
              <div style='display: inline-flex; align-items: center; gap: 8px; font-weight: 600;'>
                <span>Risk category:</span>
                <span style='padding: 4px 10px; border-radius: 999px; background: {bar_color}; color: #1a1a1a; border: 1px solid rgba(0,0,0,0.05); box-shadow: 0 1px 2px rgba(0,0,0,0.08);'>
                  {risk_label}
                </span>
              </div>
              <div style='margin-top: 20px; font-size: 12px; color: #444;'>
                Bands: Low &lt;0.10 · Moderate 0.10-0.25 · High 0.25-0.40 · Very high &gt;0.40
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        if calibration_available:
            # OPTIMIZATION: Heavy plotting libraries loaded ONLY here
            import matplotlib
            matplotlib.use('Agg') # Prevents searching for GUI backends (Tk, Qt)
            import matplotlib.pyplot as plt
            import seaborn as sns
            from matplotlib.ticker import FuncFormatter

            if age_note:
                st.caption(age_note)

            fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
            sns.pointplot(data=plot_df, x="Age", y="Calibrated_prob", color="#bbbbbb", errorbar=None, ax=ax, linewidth=0.3)
            sns.pointplot(data=plot_df, x="Age", y="q30", color="#2ca02c", errorbar=None, ax=ax, linewidth=0.4)

            ax.tick_params(axis="both", which="both", length=0)
            sns.despine(left=True, bottom=True)

            ax.set_xticks([0, 20, 40, 60, 80, 100])
            ax.set_xticklabels(["0", "20", "40", "60", "80", "100"])

            # yticks = ax.get_yticks()
            # ax.set_yticklabels([f"{int(t * 100)}" for t in yticks])

            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y * 100)}"))

            ax.scatter([age_val], [proba], color=bar_color, s=90, zorder=4, edgecolor="white", linewidth=0.8)
            ax.scatter([matched_age], [age_mean], color="#bbbbbb", s=70, zorder=5, edgecolor="white", linewidth=0.8)
            ax.scatter([matched_age], [age_q30], color="#2ca02c", s=70, zorder=5, edgecolor="white", linewidth=0.8)
            ax.set_xlabel("Age (Years)", labelpad=10)
            ax.set_ylabel("Calibrated Probability [%]", labelpad=10)

            def annotate_point(x_val, y_val, text, color, dx=6, dy=0.05):
                """Attach a floating label with an arrow to a point."""
                target_y = min(max(y_val + dy, 0), 1)
                ha = "left" if dx >= 0 else "right"
                ax.annotate(
                    text,
                    xy=(x_val, y_val),
                    xytext=(x_val + dx, target_y),
                    textcoords="data",
                    ha=ha,
                    va="center",
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=color, lw=1),
                    arrowprops=dict(arrowstyle="-", color=color, lw=1),
                )

            if age_val > 35:
                annotate_point(age_val, proba, f"Patient: {proba*100:.1f}%", bar_color, dx=8, dy=0.05)
                annotate_point(matched_age, age_mean, f"Cohort: {age_mean*100:.1f}%", "#7a7a7a", dx=-10, dy=0.05)
                annotate_point(matched_age, age_q30, f"US q30: {age_q30*100:.1f}%", "#2ca02c", dx=-4, dy=0.05)
            else:
                annotate_point(age_val, proba, f"Patient: {proba*100:.1f}%", bar_color, dx=8, dy=0.05)
                annotate_point(matched_age, age_mean, f"Cohort: {age_mean*100:.1f}%", "#7a7a7a", dx=-5, dy=0)
                annotate_point(matched_age, age_q30, f"US q30: {age_q30*100:.1f}%", "#2ca02c", dx=5, dy=0)

            ax.grid(axis="y", linestyle=":", alpha=0.35)
            ax.margins(x=0, y=0.02)
            st.pyplot(fig)
        else:
            st.info("Provide a calibration CSV path to see the calibration point plot and comparison to the dataset mean.")


# ================================
# Additional Details (Expandable)
# ================================
with st.expander("Additional Details"):
    st.markdown("### Model Validation Summary")
    st.markdown(
        f"""
        - **Brier score (calibrated):** 0.1086
        - **ECE (Error):** 0.0135
        - **Calibration intercept:** -0.015
        - **Calibration slope:** 1.005
        """
    )

    st.markdown("### Methodological Notes & Citations")
    st.markdown(
        """
        This risk model was trained on the Calgary Bloodstream Infection Cohort (_CBSIC_) using:

        - **XGBoost** classifier with class weighting
        - **Isotonic regression calibration** on a held-out 15% validation set

        **Key sources:**

        - Platt J. *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods* (1999).
        - Zadrozny & Elkan. *Transforming Classifier Scores into Accurate Multiclass Probability Estimates* (2002).
        - Van Calster et al. *Calibration: the Achilles heel of predictive analytics* (2019).
        """
    )

    st.markdown("### Version Information")
    st.markdown(
        """
        - **Model version:** 1.0.0
        - **Calibration method:** Isotonic regression
        - **App version:** 1.0.0
        - **Developer:** Lewis Research Group
        """
    )


# ------------------------
# Footer: Disclaimer
# ------------------------
st.markdown(
    """
    <hr style="margin-top:2em; margin-bottom:1em">
    <span style="color:gray; font-size:0.95em;">
    <b>Disclaimer:</b> This tool is for research and educational purposes only. 
    It does not constitute medical advice. For clinical use, consult a licensed healthcare professional.
    </span>
    """,
    unsafe_allow_html=True,
)
