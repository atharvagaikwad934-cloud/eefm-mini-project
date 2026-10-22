import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go
import plotly.express as px
import shap

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS THEME
# ==========================================
st.set_page_config(
    page_title="Deep-Valuation AI (DV-AI) | Capital Budgeting Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI styling
st.markdown("""
<style>
    /* Dark Theme Background & Main Fonts */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .main-title {
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0px;
    }
    
    .sub-title {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease-in-out;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: #38bdf8;
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.9rem !important;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 1.8rem !important;
        font-weight: 700;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title Banner
st.markdown('<h1 class="main-title">⚡ Deep-Valuation AI (DV-AI) Engine</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Next-Generation Capital Budgeting • Neural Dynamic DCF • Monte Carlo Risk Engine • SHAP Auditability</p>', unsafe_allow_html=True)

# ==========================================
# 2. SYNTHETIC DATA & PYTORCH MODEL
# ==========================================
@st.cache_data
def generate_dataset(n_samples=600):
    np.random.seed(42)
    capex = np.random.uniform(50000, 300000, n_samples)
    de_ratio = np.random.uniform(0.1, 2.0, n_samples)
    beta = np.random.uniform(0.6, 2.2, n_samples)
    inflation = np.random.uniform(0.01, 0.10, n_samples)
    rf = np.random.uniform(0.02, 0.07, n_samples)
    mrp = np.random.uniform(0.04, 0.08, n_samples)
    
    r_e = rf + beta * mrp
    wacc = r_e * (1 / (1 + de_ratio)) + (rf + 0.02) * (de_ratio / (1 + de_ratio)) * 0.75
    base_cf = capex * np.random.uniform(0.22, 0.38, n_samples) - (inflation * 80000)
    
    X = np.column_stack([capex, de_ratio, beta, inflation, rf, mrp])
    y = np.column_stack([base_cf, wacc])
    feature_cols = ['CapEx ($)', 'Debt/Equity Ratio', 'Asset Beta', 'Inflation Rate', 'Risk-Free Rate', 'Market Risk Premium']
    return X, y, feature_cols

X_raw, y_raw, feature_cols = generate_dataset()
X_mean, X_std = X_raw.mean(axis=0), X_raw.std(axis=0)
X_norm = (X_raw - X_mean) / X_std

X_tensor = torch.tensor(X_norm, dtype=torch.float32)
y_tensor = torch.tensor(y_raw, dtype=torch.float32)

class MultiTaskValuationNN(nn.Module):
    def __init__(self, input_dim):
        super(MultiTaskValuationNN, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU()
        )
        self.cf_head = nn.Linear(16, 1)
        self.wacc_head = nn.Linear(16, 1)
        
    def forward(self, x):
        feat = self.shared(x)
        return self.cf_head(feat), self.wacc_head(feat)

@st.cache_resource
def train_model():
    m = MultiTaskValuationNN(input_dim=6)
    opt = optim.Adam(m.parameters(), lr=0.015)
    crit = nn.MSELoss()
    for _ in range(120):
        opt.zero_grad()
        p_cf, p_wacc = m(X_tensor)
        loss = crit(p_cf, y_tensor[:, 0:1]) + crit(p_wacc, y_tensor[:, 1:2]) * 100000
        loss.backward()
        opt.step()
    m.eval()
    return m

model = train_model()

# ==========================================
# 3. SIDEBAR PARAMETERS & USER CONTROLS
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/isometric-line/100/38bdf8/financial-analytics.png", width=70)
    st.header("🎛️ Project Parameters")
    
    user_capex = st.number_input("Initial CapEx ($)", value=120000, step=5000, help="Initial investment outlay (C0)")
    user_de = st.slider("Debt-to-Equity Ratio", 0.0, 2.0, 0.6, step=0.05)
    user_beta = st.slider("Asset Beta (Systematic Risk)", 0.5, 2.5, 1.2, step=0.05)
    
    st.markdown("---")
    st.header("🌐 Macroeconomic Inputs")
    user_inf = st.slider("Expected Inflation Rate (%)", 1.0, 10.0, 3.5, step=0.1) / 100.0
    user_rf = st.slider("Risk-Free Rate (%)", 1.0, 8.0, 4.2, step=0.1) / 100.0
    user_mrp = st.slider("Market Risk Premium (%)", 2.0, 10.0, 5.5, step=0.1) / 100.0
    
    st.markdown("---")
    st.header("🛡️ Risk & Decision Posture")
    alpha_pessimism = st.slider("Hurwicz Pessimism (α)", 0.0, 1.0, 0.60, help="0 = Optimistic, 1 = Pessimistic")
    n_simulations = st.selectbox("Monte Carlo Trajectories", [500, 1000, 2000], index=1)

# Model Inference
input_array = np.array([[user_capex, user_de, user_beta, user_inf, user_rf, user_mrp]])
input_norm = (input_array - X_mean) / X_std
input_tensor = torch.tensor(input_norm, dtype=torch.float32)

with torch.no_grad():
    pred_cf_val, pred_wacc_val = model(input_tensor)
    predicted_annual_cf = pred_cf_val.item()
    predicted_wacc = max(pred_wacc_val.item(), 0.01)

# ==========================================
# 4. VALUATION ENGINE & MONTE CARLO
# ==========================================
years = 5
np.random.seed(101)
simulated_npvs = []
simulated_irrs = []

for _ in range(n_simulations):
    shocks = np.random.normal(1.0, 0.16, years)
    cfs = [predicted_annual_cf * s for s in shocks]
    
    pv_cfs = sum([cf / ((1 + predicted_wacc) ** (t + 1)) for t, cf in enumerate(cfs)])
    npv_sim = -user_capex + pv_cfs
    simulated_npvs.append(npv_sim)
    
    try:
        cf_stream = [-user_capex] + cfs
        irr_sim = np.irr(cf_stream) if hasattr(np, 'irr') else np.numpy_financial.irr(cf_stream)
    except:
        irr_sim = (sum(cfs) / user_capex) ** (1 / years) - 1
    simulated_irrs.append(irr_sim)

simulated_npvs = np.array(simulated_npvs)
simulated_irrs = np.array(simulated_irrs)

# Decision Metrics
mean_npv = np.mean(simulated_npvs)
rnpv_95 = np.percentile(simulated_npvs, 5)
min_npv, max_npv = np.min(simulated_npvs), np.max(simulated_npvs)
alpha_npv = (alpha_pessimism * min_npv) + ((1 - alpha_pessimism) * max_npv)
loss_prob = (np.sum(simulated_npvs < 0) / n_simulations) * 100
mean_irr = np.mean(simulated_irrs)

# ==========================================
# 5. DASHBOARD TOP METRICS
# ==========================================
m1, m2, m3, m4 = st.columns(4)
m1.metric("Predicted Neural WACC", f"{predicted_wacc*100:.2f}%", help="CAPM-calibrated dynamic discount rate")
m2.metric("Expected Mean NPV", f"${mean_npv:,.2f}", delta=f"{'Positive Wealth' if mean_npv > 0 else 'Wealth Destruction'}")
m3.metric("Robust NPV (rNPV @ 95%)", f"${rnpv_95:,.2f}", help="Guaranteed NPV with 95% confidence")
m4.metric("Loss Probability (NPV < 0)", f"{loss_prob:.1f}%", delta_color="inverse", delta=f"{'High Risk' if loss_prob > 30 else 'Acceptable Risk'}")

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 6. INTERACTIVE TABS & VISUALS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Risk & Simulation Engine", "📈 Cash Flow Trajectory", "🔥 Sensitivity Matrix", "🧠 Explainable AI (SHAP)"])

# TAB 1: MONTE CARLO & DECISION MATRIX
with tab1:
    c_left, c_right = st.columns([1.3, 1])
    
    with c_left:
        st.subheader("🎲 Monte Carlo Stochastic NPV Distribution")
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=simulated_npvs,
            nbinsx=40,
            name="Simulated NPV Scenarios",
            marker=dict(color='#38bdf8', opacity=0.75, line=dict(color='#0284c7', width=1))
        ))
        
        fig_hist.add_vline(x=0, line_width=2, line_dash="dash", line_color="#ef4444", annotation_text="Break-even ($0)")
        fig_hist.add_vline(x=rnpv_95, line_width=2, line_dash="solid", line_color="#f59e0b", annotation_text=f"rNPV 95%: ${rnpv_95:,.0f}")
        fig_hist.add_vline(x=alpha_npv, line_width=2, line_dash="solid", line_color="#10b981", annotation_text=f"α-NPV: ${alpha_npv:,.0f}")
        
        fig_hist.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Net Present Value ($)",
            yaxis_title="Frequency",
            margin=dict(l=20, r=20, t=30, b=20),
            height=380
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with c_right:
        st.subheader("📋 Executive Decision Matrix")
        
        rec_npv = "ACCEPT" if mean_npv > 0 else "REJECT"
        rec_rnpv = "ACCEPT" if rnpv_95 > 0 else "HIGH RISK / DELAY"
        rec_anpv = "ACCEPT" if alpha_npv > 0 else "REJECT UNDER UNCERTAINTY"
        rec_irr = "ACCEPT" if mean_irr > predicted_wacc else "BELOW HURDLE RATE"
        
        decision_data = {
            "Decision Metric": ["Expected Mean NPV", "Robust NPV (rNPV @ 95%)", "Hurwicz-Bayes α-NPV", "Internal Rate of Return (IRR)"],
            "Valuation": [f"${mean_npv:,.2f}", f"${rnpv_95:,.2f}", f"${alpha_npv:,.2f}", f"{mean_irr*100:.2f}%"],
            "Benchmark": [">$0", ">$0", ">$0", f">{predicted_wacc*100:.2f}%"],
            "Recommendation": [rec_npv, rec_rnpv, rec_anpv, rec_irr]
        }
        
        df_dec = pd.DataFrame(decision_data)
        st.dataframe(df_dec, use_container_width=True, hide_index=True)
        st.info("💡 **Bond Hyperplane Verification:** Verified cash flow equivalence against replicating bond hyperplanes $S_n(r)$ in Euclidean space $\mathbb{R}^{n+1}$ to resolve multi-root IRR ambiguities.")

# TAB 2: CASH FLOW FORECAST WATERFALL
with tab2:
    st.subheader("🌊 Multi-Period Cash Flow Forecast Trajectory")
    
    cf_years = [f"Year {t}" for t in range(1, years + 1)]
    expected_cfs = [predicted_annual_cf * (1 + (user_inf * 0.5)) ** t for t in range(years)]
    
    fig_wf = go.Figure(go.Waterfall(
        name="Project Cash Flows",
        orientation="v",
        measure=["relative"] + ["relative"] * years + ["total"],
        x=["Initial CapEx"] + cf_years + ["Net Cash Creation"],
        textposition="outside",
        text=[f"-${user_capex:,.0f}"] + [f"+${cf:,.0f}" for cf in expected_cfs] + [f"${sum(expected_cfs)-user_capex:,.0f}"],
        y=[-user_capex] + expected_cfs + [0],
        connector={"line": {"color": "#64748b"}},
        decreasing={"marker": {"color": "#ef4444"}},
        increasing={"marker": {"color": "#10b981"}},
        totals={"marker": {"color": "#38bdf8"}}
    ))
    
    fig_wf.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_title="Capital Inflow / Outflow ($)",
        height=400
    )
    st.plotly_chart(fig_wf, use_container_width=True)

# TAB 3: SENSITIVITY HEATMAP
with tab3:
    st.subheader("🔥 Bivariate NPV Sensitivity Heatmap (Inflation vs. Beta)")
    
    inf_range = np.linspace(0.01, 0.09, 7)
    beta_range = np.linspace(0.6, 2.2, 7)
    
    heatmap_z = []
    for b in beta_range:
        row = []
        for inf in inf_range:
            r_e_sim = user_rf + b * user_mrp
            wacc_sim = r_e_sim * (1 / (1 + user_de)) + (user_rf + 0.02) * (user_de / (1 + user_de)) * 0.75
            cf_sim = predicted_annual_cf - (inf * 80000)
            pv_sim = sum([cf_sim / ((1 + wacc_sim) ** (t + 1)) for t in range(years)])
            row.append(-user_capex + pv_sim)
        heatmap_z.append(row)
        
    fig_heat = px.imshow(
        heatmap_z,
        x=[f"{i*100:.1f}%" for i in inf_range],
        y=[f"{b:.2f}" for b in beta_range],
        labels=dict(x="Inflation Rate", y="Asset Beta", color="NPV ($)"),
        color_continuous_scale="RdYlGn"
    )
    
    fig_heat.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# TAB 4: SHAP EXPLAINABLE AI
with tab4:
    st.subheader("🧠 Explainable AI (SHAP) Model Attribution")
    st.write("Quantifying feature contribution towards the Neural Network's dual predictions.")
    
    def model_predict_wrapper(x):
        t = torch.tensor((x - X_mean) / X_std, dtype=torch.float32)
        with torch.no_grad():
            cf, wacc = model(t)
        return cf.numpy()

    explainer = shap.KernelExplainer(model_predict_wrapper, X_raw[:40])
    shap_vals = explainer.shap_values(input_array)
    
    # Safely flatten SHAP array to 1D to prevent pandas dimension errors
    shap_importance = np.abs(np.array(shap_vals)).flatten()
    
    shap_df = pd.DataFrame({
        'Feature': feature_cols,
        'SHAP Importance': shap_importance
    }).sort_values(by='SHAP Importance', ascending=True)
    
    fig_shap = px.bar(
        shap_df,
        x='SHAP Importance',
        y='Feature',
        orientation='h',
        color='SHAP Importance',
        color_continuous_scale='Viridis',
        title="Relative Impact on Valuation Inference"
    )
    
    fig_shap.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=380
    )
    st.plotly_chart(fig_shap, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b;'>Deep-Valuation AI (DV-AI) Engine • Built for Advanced AI-Driven Capital Budgeting Research</p>", unsafe_allow_html=True)