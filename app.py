import streamlit as st
import numpy as np
from utils.pricer import monte_carlo_hybrid

st.set_page_config(page_title="MacroQuant AI Hybrid", layout="wide", page_icon="🚀")

st.title("🚀 MacroQuant AI Hybrid Payoff Engine")
st.markdown("**Citi Interview Demo** — Contingent Hybrid Payoff (Index + Commodity)")

# ====================== Asset Selection ======================
st.subheader("Select Assets")

col1, col2 = st.columns(2)

with col1:
    selected_index = st.selectbox("Select Index", ["HSI", "KOSPI"])
    index_spot = st.number_input(f"Current {selected_index} Spot", 
                                 value=26160 if selected_index == "HSI" else 2550, step=50)

with col2:
    selected_comm = st.selectbox("Select Commodity", ["WTI Crude Oil", "Gold"])
    comm_spot = st.number_input(f"Current {selected_comm} Price", 
                                value=78.0 if selected_comm == "WTI Crude Oil" else 2650.0, step=1.0)
    comm_barrier = st.number_input(f"{selected_comm} Barrier", 
                                   value=80.0 if selected_comm == "WTI Crude Oil" else 2700.0, step=1.0)

condition = st.radio("Payoff Condition", ["below", "above"], 
                     format_func=lambda x: f"Active if {selected_comm} is {x} the barrier")

# ====================== Payoff Parameters ======================
st.subheader("Bull Call Spread Parameters")
lower_strike = st.number_input("Lower Strike (Long Call)", value=int(index_spot * 0.98), step=100)
higher_strike = st.number_input("Higher Strike (Short Call)", value=int(index_spot * 1.05), step=100)

days = st.slider("Days to Expiration", 30, 180, 90)
index_vol = st.slider("Index Volatility (%)", 15, 45, 22) / 100.0
comm_vol = st.slider(f"{selected_comm} Volatility (%)", 15, 60, 28) / 100.0
correlation = st.slider("Correlation (Index vs Commodity)", -0.9, 0.5, -0.35, 0.05)

user_input = st.text_area("Your Macro Idea:", 
                          "Bullish on Asian equities if energy prices remain suppressed", height=100)

# ====================== Run Analysis ======================
if st.button("🚀 Run AI Analysis & Monte Carlo Pricing", type="primary"):
    with st.spinner("Running Monte Carlo simulation..."):
        pricing = monte_carlo_hybrid(
            index_spot=index_spot,
            comm_spot=comm_spot,
            lower=lower_strike,
            higher=higher_strike,
            barrier=comm_barrier,
            days=days,
            index_vol=index_vol,
            comm_vol=comm_vol,
            correlation=correlation,
            condition=condition
        )
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📌 Macro Thesis")
        st.info(f"If {selected_comm} stays {condition} its barrier, it should support {selected_index} through improved risk sentiment.")
        st.subheader("📊 Quant Analysis")
        st.info("Correlation incorporated in Monte Carlo simulation.")
    
    with col2:
        st.subheader("💰 Hybrid Payoff Pricing")
        st.success(f"Contingent Bull Call Spread on {selected_index}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected Payoff", f"{pricing['expected_payoff']} pts")
        c2.metric("Probability Condition Met", f"{pricing['prob_condition']}%")
        c3.metric("Fair Value", f"{pricing['fair_value']} pts")

st.caption("MacroQuant Hybrid Payoff Engine | Index + Commodity | Monte Carlo priced")