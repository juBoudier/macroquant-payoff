# MacroQuant Hybrid Payoff Engine

**Citi Interview Demo** — Intelligent Hybrid Options Payoff Tool

A Streamlit-based application that combines **macro research**, **quantitative analysis**, and **Monte Carlo pricing** for conditional hybrid payoffs involving equity indices and commodities.

### Features

- **Dynamic Asset Selection**
  - Indices: HSI (Hong Kong) or KOSPI (South Korea)
  - Commodities: WTI Crude Oil or Gold

- **Contingent Hybrid Payoff**
  - Bull Call Spread on selected index
  - Payoff is **conditional** (activated only if commodity meets barrier condition — below or above)

- **Monte Carlo Pricing**
  - Simulates 10,000 paths with correlation between index and commodity
  - Calculates expected payoff and probability of condition being met

- **AI-style Research Layer**
  - Generates professional macro thesis and quant analysis based on user input

- **Backtesting** (in progress)
  - Historical performance simulation of the strategy

### How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/JuBoudier/macroquant-payoff.git

# 2. Go to project folder
cd macroquant-payoff

# 3. Install dependencies
pip install streamlit numpy pandas yfinance

# 4. Run the app
streamlit run app.py
