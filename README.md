# ⚡ Deep-Valuation AI (DV-AI) Engine

An Explainable, Multi-Task Neural Network & Monte Carlo Decision Engine for Capital Budgeting (NPV/IRR Analysis).

## 📌 Features
- **Multi-Task PyTorch Neural Network:** Predicts dynamic expected annual cash flows ($C_t$) and dynamic Weighted Average Cost of Capital (WACC) simultaneously.
- **Monte Carlo Risk Engine:** Simulates 1,000+ stochastic market trajectories under fat-tailed shocks.
- **Advanced Decision Framework:** Calculates Chance-Constrained Robust NPV ($r\text{NPV}_{95\%}$), Hurwicz-Bayes ($\alpha\text{-NPV}$) index under Knightian uncertainty, and Euclidean Bond Hyperplane equivalence verification.
- **Explainable AI (SHAP):** Game-theoretic feature attribution for boardroom auditability.
- **Interactive UI:** Built with Streamlit and Plotly.

## 🚀 Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py