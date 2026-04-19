import numpy as np

def monte_carlo_hybrid(index_spot, comm_spot, lower, higher, barrier, days=90, 
                       index_vol=0.22, comm_vol=0.25, correlation=-0.35, 
                       condition="below", n_sim=10000):
    
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