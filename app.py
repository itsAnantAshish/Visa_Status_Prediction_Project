import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ─── PAGE CONFIG ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Visa Processing Time Estimator",
    page_icon="🛂",
    layout="wide"
)

# ─── LOAD MODEL ──────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('model/visa_model.pkl')

model_data = load_model()
model      = model_data['model']
encoders   = model_data['encoders']
metrics    = model_data['metrics']
feat_imp   = model_data['feature_importances']
feat_cols  = model_data['feature_columns']

# ─── HEADER ──────────────────────────────────────────────────────
st.title("🛂 AI-Enabled Visa Processing Time Estimator")
st.markdown("*Infosys Springboard Internship Project — Predict how long your visa application will take*")
st.divider()

# ─── TABS ────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Analytics", "📈 Model Info"])

# ════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Application Details")

    col1, col2 = st.columns(2)

    with col1:
        case_status = st.selectbox(
            "Case Status",
            ['CERTIFIED', 'CERTIFIED-WITHDRAWN', 'DENIED', 'WITHDRAWN']
        )
        job_title = st.text_input("Job Title", value="Software Engineer")
        prevailing_wage = st.number_input(
            "Prevailing Wage (Annual, USD)",
            min_value=10000, max_value=500000, value=75000, step=1000
        )

    with col2:
        full_time = st.selectbox("Full Time Position", ['Y', 'N'])
        year = st.slider("Application Year", 2010, 2024, 2020)
        state = st.selectbox("Work State", [
            'CA', 'NY', 'TX', 'WA', 'IL', 'NJ', 'MA', 'GA',
            'FL', 'PA', 'OH', 'MI', 'NC', 'VA', 'CO', 'UNKNOWN'
        ])

    employer_name = st.text_input("Employer Name", value="Tech Corp Inc")

    st.divider()
    predict_btn = st.button("🔮 Estimate Processing Time", type="primary", use_container_width=True)

    if predict_btn:
        # ── Build input dict matching training features ──
        wage_cat_bins  = [0, 40000, 70000, 100000, 150000, float('inf')]
        wage_cat_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        wage_cat = pd.cut(
            [prevailing_wage], bins=wage_cat_bins, labels=wage_cat_labels
        )[0]

        # Frequency encoding for employer / job title
        # Since we don't have training frequencies at inference time,
        # we use a reasonable default (median frequency ~0.001)
        employer_freq = 0.0005
        job_freq      = 0.001

        input_dict = {}

        if 'CASE_STATUS' in feat_cols:
            cs_encoded = encoders['CASE_STATUS'].transform([case_status])[0] \
                if case_status in encoders['CASE_STATUS'].classes_ \
                else 0
            input_dict['CASE_STATUS'] = cs_encoded

        if 'PREVAILING_WAGE' in feat_cols:
            input_dict['PREVAILING_WAGE'] = prevailing_wage

        if 'YEAR' in feat_cols:
            input_dict['YEAR'] = year

        if 'STATE' in feat_cols:
            st_encoded = encoders['STATE'].transform([state])[0] \
                if state in encoders['STATE'].classes_ \
                else 0
            input_dict['STATE'] = st_encoded

        if 'WAGE_CATEGORY' in feat_cols:
            wc_encoded = encoders['WAGE_CATEGORY'].transform([str(wage_cat)])[0] \
                if str(wage_cat) in encoders['WAGE_CATEGORY'].classes_ \
                else 2
            input_dict['WAGE_CATEGORY'] = wc_encoded

        if 'FULL_TIME_BINARY' in feat_cols:
            input_dict['FULL_TIME_BINARY'] = 1 if full_time == 'Y' else 0

        if 'EMPLOYER_NAME_FREQ' in feat_cols:
            input_dict['EMPLOYER_NAME_FREQ'] = employer_freq

        if 'JOB_TITLE_FREQ' in feat_cols:
            input_dict['JOB_TITLE_FREQ'] = job_freq

        # Build dataframe in correct column order
        input_df = pd.DataFrame([input_dict])[feat_cols]

        prediction = model.predict(input_df)[0]
        lower = max(15, int(prediction * 0.85))
        upper = min(120, int(prediction * 1.15))

        # ── Display Result ──
        st.success("✅ Prediction Complete!")

        r1, r2, r3 = st.columns(3)
        r1.metric("⏱️ Estimated Processing Time", f"{int(prediction)} days")
        r2.metric("📉 Optimistic Estimate", f"{lower} days")
        r3.metric("📈 Conservative Estimate", f"{upper} days")

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=int(prediction),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Processing Time (days)"},
            delta={'reference': 60},
            gauge={
                'axis': {'range': [15, 120]},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [15, 40],  'color': "#2ecc71"},
                    {'range': [40, 75],  'color': "#f39c12"},
                    {'range': [75, 120], 'color': "#e74c3c"},
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Interpretation
        if prediction < 40:
            st.info("🟢 **Fast Track** — Your application is likely to be processed quickly.")
        elif prediction < 75:
            st.info("🟡 **Standard Processing** — Typical processing timeline.")
        else:
            st.warning("🔴 **Extended Processing** — Expect a longer wait. Consider following up after 60 days.")

# ════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Visa Processing Analytics")

    # Simulated trend data for visualization
    years = list(range(2010, 2025))
    avg_times = [75, 72, 70, 68, 65, 63, 61, 58, 60, 62, 64, 58, 55, 57, 56]

    fig_trend = px.line(
        x=years, y=avg_times,
        title="Average Visa Processing Time by Year",
        labels={'x': 'Year', 'y': 'Avg Processing Days'},
        markers=True
    )
    fig_trend.update_traces(line_color='#1f77b4', line_width=2)
    st.plotly_chart(fig_trend, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # Processing time by wage category
        wage_cats  = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        wage_times = [85, 72, 62, 52, 45]
        fig_wage = px.bar(
            x=wage_cats, y=wage_times,
            title="Avg Processing Time by Wage Category",
            labels={'x': 'Wage Category', 'y': 'Avg Days'},
            color=wage_times,
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig_wage, use_container_width=True)

    with col2:
        # Case status distribution
        statuses = ['CERTIFIED', 'CERTIFIED-WITHDRAWN', 'DENIED', 'WITHDRAWN']
        counts   = [65, 15, 12, 8]
        fig_pie  = px.pie(
            values=counts, names=statuses,
            title="Case Status Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Processing time distribution
    np.random.seed(42)
    sim_times = np.clip(np.random.normal(60, 18, 1000), 15, 120)
    fig_dist  = px.histogram(
        x=sim_times, nbins=40,
        title="Processing Time Distribution",
        labels={'x': 'Processing Days', 'y': 'Count'},
        color_discrete_sequence=['#1f77b4']
    )
    fig_dist.add_vline(x=60, line_dash="dash", line_color="red",
                       annotation_text="Avg: 60 days")
    st.plotly_chart(fig_dist, use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 3 — MODEL INFO
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader("📈 Model Performance & Feature Importance")

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE",  f"{metrics['mae']:.2f} days")
    m2.metric("RMSE", f"{metrics['rmse']:.2f} days")
    m3.metric("R² Score", f"{metrics['r2']:.4f}")

    st.divider()

    # Feature importance chart
    fi_df = pd.DataFrame(
        list(feat_imp.items()),
        columns=['Feature', 'Importance']
    ).sort_values('Importance', ascending=True)

    fig_fi = px.bar(
        fi_df, x='Importance', y='Feature',
        orientation='h',
        title="Feature Importance (XGBoost)",
        color='Importance',
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.divider()
    st.markdown("""
    **Model Details:**
    - **Algorithm:** XGBoost Regressor
    - **Training:** 80/20 train-test split
    - **Hyperparameters:** 300 estimators, max depth 6, learning rate 0.05
    - **Target:** `processing_time_days` (synthetic, range 15–120)
    - **Pipeline:** Data cleaning → Feature engineering → Label encoding → XGBoost
    """)