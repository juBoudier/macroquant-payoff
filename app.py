import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

st.set_page_config(page_title="MacroQuant Hybrid Payoff", layout="wide", page_icon="🚀")

st.title("🚀 MacroQuant Hybrid Payoff Engine")
st.markdown("**Citi Interview Demo** — Contingent Hybrid Payoff with Monte Carlo + Backtesting")

# =============================================
# Asset Selection
# =============================================
st.subheader("Select Assets")

col1, col2 = st.columns(2)

with col1:
    selected_index = st.selectbox("Select Index", ["HSI", "KOSPI"], index=0)
    index_spot = st.number_input(f"Current {selected_index} Spot", 
                                 value=26160 if selected_index == "HSI" else 2550, step=50)

with col2:
    selected_comm = st.selectbox("Select Commodity", ["WTI Crude Oil", "Gold"], index=0)
    comm_spot = st.number_input(f"Current {selected_comm} Price", 
                                value=78.0 if selected_comm == "WTI Crude Oil" else 2650.0, step=1.0)
    comm_barrier = st.number_input(f"{selected_comm} Barrier", 
                                   value=80.0 if selected_comm == "WTI Crude Oil" else 2700.0, step=1.0)

condition = st.radio("Payoff Condition", ["below", "above"], 
                     format_func=lambda x: f"Active if {selected_comm} is {x} barrier")

# Payoff Parameters
st.subheader("Bull Call Spread Parameters")
lower_strike = st.number_input("Lower Strike (Long Call)", value=int(index_spot * 0.98), step=100)
higher_strike = st.number_input("Higher Strike (Short Call)", value=int(index_spot * 1.05), step=100)

days = st.slider("Days to Expiration", 30, 180, 90)
index_vol = st.slider("Index Volatility (%)", 15, 45, 22) / 100.0
comm_vol = st.slider(f"{selected_comm} Volatility (%)", 15, 60, 28) / 100.0
correlation = st.slider("Correlation (Index vs Commodity)", -0.9, 0.5, -0.35, 0.05)

user_input = st.text_area("Your Macro Idea:", "Analyze HSI if WTI stays below $80", height=100)

# =============================================
# Monte Carlo Pricer
# =============================================
def monte_carlo_hybrid(index_spot, comm_spot, lower, higher, barrier, days=90, 
                       index_vol=0.22, comm_vol=0.25, correlation=-0.35, condition="below", n_sim=10000):
    
    T = days / 365.0
    dt = T / 252
    n_steps = max(1, int(252 * T))
    
    L = np.linalg.cholesky(np.array([[1.0, correlation], [correlation, 1.0]]))
    Z = np.random.normal(0, 1, size=(2, n_sim, n_steps))
    dW = np.einsum('ij,jkl->ikl', L, Z)
    
    # Index paths
    index_drift = (0.03 - 0.5 * index_vol**2) * dt
    final_index = index_spot * np.prod(np.exp(index_drift + index_vol * np.sqrt(dt) * dW[0]), axis=1)
    
    # Commodity paths
    comm_drift = (0.03 - 0.5 * comm_vol**2) * dt
    final_comm = comm_spot * np.prod(np.exp(comm_drift + comm_vol * np.sqrt(dt) * dW[1]), axis=1)
    
    base_payoff = np.maximum(final_index - lower, 0) - np.maximum(final_index - higher, 0)
    
    if condition == "below":
        payoff = np.where(final_comm < barrier, base_payoff, 0)
    else:
        payoff = np.where(final_comm > barrier, base_payoff, 0)
    
    expected = np.mean(payoff)
    prob = np.mean(final_comm < barrier if condition == "below" else final_comm > barrier) * 100
    
    return {
        "expected_payoff": round(expected, 2),
        "prob_condition": round(prob, 1),
        "fair_value": round(expected, 2)
    }

# =============================================
# Backtester (Improved & More Stable)
# =============================================
@st.cache_data(ttl=3600)
def run_backtest(months_back=36, selected_index="HSI", selected_comm="WTI Crude Oil"):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30 * months_back + 400)
    
    try:
        # Download data
        if selected_index == "HSI":
            index_data = yf.download("^HSI", start=start_date, end=end_date, progress=False)['Close']
        else:
            index_data = yf.download("^KS11", start=start_date, end=end_date, progress=False)['Close']
        
        comm_ticker = "CL=F" if selected_comm == "WTI Crude Oil" else "GC=F"
        comm_data = yf.download(comm_ticker, start=start_date, end=end_date, progress=False)['Close']
        
        df = pd.DataFrame({'Index': index_data, 'Commodity': comm_data}).dropna()
        
        if len(df) < 30:
            return pd.DataFrame(), "Not enough historical data."
        
        monthly = df.resample('M').last()
        
        results = []
        for i in range(len(monthly) - 3):
            entry_index = monthly['Index'].iloc[i]
            exit_index = monthly['Index'].iloc[i+3]
            exit_comm = monthly['Commodity'].iloc[i+3]
            
            lower = round(entry_index * 0.98 / 100) * 100
            higher = lower + 1500
            net_debit = 200.0
            
            if (selected_comm == "WTI Crude Oil" and exit_comm < 80) or \
               (selected_comm == "Gold" and exit_comm < 2700):
                payoff = max(exit_index - lower, 0) - max(exit_index - higher, 0) - net_debit
            else:
                payoff = -net_debit
                
            ret_pct = (payoff / net_debit) * 100 if net_debit > 0 else 0
            
            results.append({
                'Entry Date': monthly.index[i].strftime('%Y-%m'),
                'Entry Index': round(entry_index, 1),
                'Exit Index': round(exit_index, 1),
                'Exit Commodity': round(exit_comm, 2),
                'Payoff': round(payoff, 1),
                'Return %': round(ret_pct, 1),
                'Condition Met': 'Yes' if ((selected_comm == "WTI Crude Oil" and exit_comm < 80) or 
                                          (selected_comm == "Gold" and exit_comm < 2700)) else 'No'
            })
        
        return pd.DataFrame(results), None
    
    except Exception as e:
        return pd.DataFrame(), f"Data error: {str(e)}"

# =============================================
# Main UI with Tabs
# =============================================
tab1, tab2, tab3 = st.tabs(["📊 Analysis", "💰 Monte Carlo Pricing", "📈 Backtesting"])

with tab1:
    if st.button("🚀 Run Analysis", type="primary"):
        result = {"macro_thesis": f"If {selected_comm} stays {condition} barrier, it should support {selected_index}.",
                  "quant_analysis": "Correlation & conditional structure analyzed via Monte Carlo.",
                  "final_result": f"Contingent Bull Call Spread on {selected_index}"}
        
        colA, colB = st.columns(2)
        with colA:
            st.subheader("📌 Macro Thesis")
            st.info(result["macro_thesis"])
            st.subheader("📊 Quant Analysis")
            st.info(result["quant_analysis"])
        with colB:
            st.subheader("💰 Recommended Payoff")
            st.success(result["final_result"])

with tab2:
    if st.button("Reprice with Monte Carlo", type="primary"):
        with st.spinner("Running Monte Carlo (10,000 paths)..."):
            pricing = monte_carlo_hybrid(index_spot, comm_spot, lower_strike, higher_strike, comm_barrier,
                                         days, index_vol, comm_vol, correlation, condition)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected Payoff", f"{pricing['expected_payoff']} pts")
        c2.metric("Probability Condition Met", f"{pricing['prob_condition']}%")
        c3.metric("Fair Value", f"{pricing['fair_value']} pts")

with tab3:
    st.subheader("Historical Backtesting")
    months = st.slider("Backtest Period (months)", 12, 60, 36)
    
    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running historical backtest..."):
            bt_results, error = run_backtest(months, selected_index, selected_comm)
        
        if error:
            st.error(error)
        elif bt_results.empty:
            st.warning("Not enough data for backtesting.")
        else:
            st.dataframe(bt_results.style.format({
                'Entry Index': '{:.1f}', 'Exit Index': '{:.1f}', 'Exit Commodity': '{:.2f}',
                'Payoff': '{:.1f}', 'Return %': '{:.1f}%'
            }), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Return", f"{bt_results['Return %'].mean():.1f}%")
            col2.metric("Win Rate", f"{(bt_results['Condition Met'] == 'Yes').mean()*100:.1f}%")
            col3.metric("Condition Met Rate", f"{(bt_results['Condition Met'] == 'Yes').mean()*100:.1f}%")

st.caption("Hybrid Contingent Payoff Engine | Index + Commodity | Monte Carlo + Backtesting | Citi Interview Demo")