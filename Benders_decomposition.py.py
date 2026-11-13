import pyomo.environ as pyo
import pandas as pd
from pyomo.opt import SolverFactory
from dataclasses import dataclass
import Read_data as rd


"""
Benders decomposition of the two-stage stochastic BESS model.
Based on the code given in Lecture (TET4565) (Benders_decomposition.py, farmer's problem).
"""
 
@dataclass
class BatteryParams: 
    P_max: float= 9.79       #Power limit [MW]
    E_max: float= 39.16           #Maximum energy of the BESS [MWh] (4 h battery)
    Eta_ch: float = 0.968       #Charging efficiency [-]
    Eta_dis: float = 0.968      #Discharging efficiency [-]
    SOC_max: float = 0.8       #Maximum state of charge [-]
    SOC_min: float = 0.2       #Minimum state of charge [-]
    SOC_init: float = 0.5      #Initial state of charge [-]


#--------
#Get data 
#-----------
Delta=0.25 #Duration of one time step [h]

#read in the prices, using the functions made in Read_data.py
df=rd.read_spot_prices(rd.DATA_PATH)

#stage 1 prices: 
df_stage1= rd.get_stage1_prices(df,rd.STAGE1_DAY)
price_stage1= list(df_stage1["Price"].values)


#Stage 2 prices: one price profile for each scenario
scenarios_price= rd.get_stage2_prices(df, rd.STAGE2_DAYS) #Dictonary of stage 2 prices, one df for each scenario

 
 
# -----------------------------
# Master problem (1st stage)
# -----------------------------
def ModelSetUp_1st(batt, price_stage1, Cuts):
    m = pyo.ConcreteModel()
 
    # Sets
    T_1 = list(range(1, len(price_stage1) + 1))  # Quarters 1-96
    m.T_1 = pyo.Set(initialize=T_1)
 
    # Parameters
    m.DA_price_stage_1 = pyo.Param(m.T_1, initialize=dict(zip(T_1, price_stage1)))
 
    # Variables
    m.p_ch_stage1 = pyo.Var(m.T_1, within=pyo.NonNegativeReals)
    m.p_dis_stage1 = pyo.Var(m.T_1, within=pyo.NonNegativeReals)
    m.SOC_t_stage1 = pyo.Var(m.T_1, within=pyo.NonNegativeReals)
 
    """Cuts information"""
    #Set for cuts
    m.Cut = pyo.Set(initialize=Cuts["Set"]) #set of cuts
    #Parameter for cuts
    m.Phi = pyo.Param(m.Cut, initialize=Cuts["Phi"]) # Sum of phi_s from iteration c
    m.Rho = pyo.Param(m.Cut, initialize=Cuts["rho"]) # Sum of rho_s from iteration c
    m.SOC_hat = pyo.Param(m.Cut, initialize=Cuts["SOC_hat"]) # SOC_96 used in iteration c
    #Variable for alpha. Note: The boundary values are only set to avoid infeasible issues in the first iteration, if not alpha is not limited and can be infinity (and beyond)
    m.alpha = pyo.Var(bounds=(-1000000, 1000000))

    """
    The following functions define the objective function and constraints of the first stage of the optimization problem
    """
 
    # Objective: first-stage profit + expected second-stage profit (alpha)
    def Obj_1st(m):
        stage1 = sum((m.p_dis_stage1[t] - m.p_ch_stage1[t]) * m.DA_price_stage_1[t] * Delta for t in m.T_1)
        return stage1 + m.alpha

    m.obj = pyo.Objective(rule=Obj_1st, sense=pyo.maximize)
 
    # Constraints (2), (3), (6), (8), (11)
    def P_ch_limit_max_stage1(m, t):
        return m.p_ch_stage1[t] <= batt.P_max
 
    def P_dis_limit_max_stage1(m, t):
        return m.p_dis_stage1[t] <= batt.P_max
 
    def SOC_limit_max_stage1(m, t):
        return m.SOC_t_stage1[t] <= batt.SOC_max
 
    def SOC_limit_min_stage1(m, t):
        return m.SOC_t_stage1[t] >= batt.SOC_min
 
    def SOC_update_stage1(m, t):
        if t == T_1[0]:  # Start condition, t=1
            return m.SOC_t_stage1[t] == batt.SOC_init + (m.p_ch_stage1[t] * batt.Eta_ch - m.p_dis_stage1[t] / batt.Eta_dis) * Delta / batt.E_max
        else:
            return m.SOC_t_stage1[t] == m.SOC_t_stage1[t - 1] + (m.p_ch_stage1[t] * batt.Eta_ch - m.p_dis_stage1[t] / batt.Eta_dis) * Delta / batt.E_max
 
    # Benders cut: alpha <= Phi + Rho * (SOC_96 - SOC_hat)
    def CreateCuts(m, c):
        return m.alpha <= m.Phi[c] + m.Rho[c] * (m.SOC_t_stage1[T_1[-1]] - m.SOC_hat[c])
 
    m.P_ch_limit_max_stage1 = pyo.Constraint(m.T_1, rule=P_ch_limit_max_stage1)
    m.P_dis_limit_max_stage1 = pyo.Constraint(m.T_1, rule=P_dis_limit_max_stage1)
    m.SOC_limit_max_stage1 = pyo.Constraint(m.T_1, rule=SOC_limit_max_stage1)
    m.SOC_limit_min_stage1 = pyo.Constraint(m.T_1, rule=SOC_limit_min_stage1)
    m.SOC_update_stage1 = pyo.Constraint(m.T_1, rule=SOC_update_stage1)
    m.CreateCuts = pyo.Constraint(m.Cut, rule=CreateCuts)
 
    return m
 
 
# -----------------------------
# Subproblem (2nd stage), one scenario
# -----------------------------
def ModelSetUp_2nd(batt, price_stage2, SOC_hat):
    m = pyo.ConcreteModel()
 
    # Sets
    T_2 = list(range(97, 97 + len(price_stage2)))  # Quarters 97-192 (stage 2)
    m.T_2 = pyo.Set(initialize=T_2)
 
    # Parameters
    m.DA_price_stage_2 = pyo.Param(m.T_2, initialize=dict(zip(T_2, price_stage2)))
    m.pi = pyo.Param(initialize=0.2)  # Probabilities equal for each scenario
    m.SOC_hat = pyo.Param(initialize=SOC_hat)  # SOC_96 from the master problem
 
    # Variables
    m.p_ch_stage2 = pyo.Var(m.T_2, within=pyo.NonNegativeReals)
    m.p_dis_stage2 = pyo.Var(m.T_2, within=pyo.NonNegativeReals)
    m.SOC_t_stage2 = pyo.Var(m.T_2, within=pyo.NonNegativeReals)
    m.SOC_96 = pyo.Var(within=pyo.NonNegativeReals)  # Complicating variable, fixed by SOC_hat
 
    # Objective: phi_s = pi_s * second-stage profit
    def Obj_2nd(m):
        return m.pi * sum((m.p_dis_stage2[t] - m.p_ch_stage2[t]) * m.DA_price_stage_2[t] * Delta for t in m.T_2)

    m.obj = pyo.Objective(rule=Obj_2nd, sense=pyo.maximize)
 
    # Constraints (4), (5), (7), (9), (10), (12)
    def P_ch_limit_max_stage2(m, t):
        return m.p_ch_stage2[t] <= batt.P_max
 
    def P_dis_limit_max_stage2(m, t):
        return m.p_dis_stage2[t] <= batt.P_max
 
    def SOC_limit_max_stage2(m, t):
        return m.SOC_t_stage2[t] <= batt.SOC_max
 
    def SOC_limit_min_stage2(m, t):
        return m.SOC_t_stage2[t] >= batt.SOC_min
 
    def SOC_update_stage2(m, t):
        if t == T_2[0]:  # Start condition, t=97, linked to SOC_96
            return m.SOC_t_stage2[t] == m.SOC_96 + (m.p_ch_stage2[t] * batt.Eta_ch - m.p_dis_stage2[t] / batt.Eta_dis) * Delta / batt.E_max
        else:
            return m.SOC_t_stage2[t] == m.SOC_t_stage2[t - 1] + (m.p_ch_stage2[t] * batt.Eta_ch - m.p_dis_stage2[t] / batt.Eta_dis) * Delta / batt.E_max
 
    def SOC_end(m):
        return m.SOC_t_stage2[T_2[-1]] == batt.SOC_init
 
    def SOC_fix(m):
        """
        This constraint is meant to connect the parameter from the first stage decision to the second-stage problem
        This is perfect for finding the dual value of the decision variable, so we can generate cuts
        """
        return m.SOC_96 == m.SOC_hat

 
    m.P_ch_limit_max_stage2 = pyo.Constraint(m.T_2, rule=P_ch_limit_max_stage2)
    m.P_dis_limit_max_stage2 = pyo.Constraint(m.T_2, rule=P_dis_limit_max_stage2)
    m.SOC_limit_max_stage2 = pyo.Constraint(m.T_2, rule=SOC_limit_max_stage2)
    m.SOC_limit_min_stage2 = pyo.Constraint(m.T_2, rule=SOC_limit_min_stage2)
    m.SOC_update_stage2 = pyo.Constraint(m.T_2, rule=SOC_update_stage2)
    m.SOC_end = pyo.Constraint(rule=SOC_end)
    m.SOC_fix = pyo.Constraint(rule=SOC_fix)
 
    return m
 
 
# -----------------------------
# Solve
# -----------------------------
def Solve(m):
    opt = SolverFactory("gurobi")
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    results = opt.solve(m, load_solutions=True)
    return results, m

 
 
# -----------------------------
# Add a new cut from the subproblems
# -----------------------------
def Cut_manage(Cuts, subproblems, SOC_hat):
    cut = len(Cuts["Set"])
    Cuts["Set"].append(cut)
 
    # Combine the scenarios (pi_s is already included in phi_s and rho_s)
    Cuts["Phi"][cut] = sum(pyo.value(subproblems[s].obj) for s in subproblems)
    Cuts["rho"][cut] = sum(subproblems[s].dual[subproblems[s].SOC_fix] for s in subproblems)
    Cuts["SOC_hat"][cut] = SOC_hat
    return Cuts


#--------------
# Save results to excel
#-------------
def save_results(results, i, SOC_hat, UB, LB, subproblems, m_1st,alpha):
    row = {"Iteration": i + 1,
           "SOC_96": SOC_hat,
           "Stage 1 profit": stage1_profit,
           "Stage 2 profit": sum(pyo.value(subproblems[s].obj) for s in subproblems),
           "alpha": alpha,
           "UB": UB,
           "LB": LB,
           "Gap": UB - LB}
    for s in subproblems:
        row["rho_" + str(s)] = subproblems[s].dual[subproblems[s].SOC_fix]
    results.append(row)

    stage1 = pd.DataFrame({
        "P_ch [MW]": m_1st.p_ch_stage1.extract_values(),
        "P_dis [MW]": m_1st.p_dis_stage1.extract_values(),
        "SOC [-]": m_1st.SOC_t_stage1.extract_values(),
    })

    with pd.ExcelWriter("Benders_Decomposition_results.xlsx") as writer:
        pd.DataFrame(results).to_excel(writer, sheet_name="Convergence", index=False)
        stage1.to_excel(writer, sheet_name="Stage 1", index_label="Quarter")
    return results

 
def check_alpha_bound(batt, scenarios_price, Delta, ALPHA_UP):
    # Upper bound assuming maximum discharge in every quarter,
    # ignoring charging costs and energy limitations.
    profit_upper_bound = 0.0

    for s in scenarios_price:
        prices = scenarios_price[s]["Price"]
        profit_upper_bound += 0.2 * prices.sum() * batt.P_max * Delta

    print(f"Stage-2 profit upper bound: {profit_upper_bound:.2f} EUR")
    print(f"Below ALPHA_UP: {profit_upper_bound <= ALPHA_UP}")

    return profit_upper_bound

 
# -----------------------------
# Benders algorithm
# -----------------------------
if __name__ == "__main__":
 
    batt = BatteryParams()
    S = list(range(1,6))  # Scenarios 1-5
    epsilon = 0.01  # Convergence tolerance [EUR], assuming this value is small enough. 
    Iterations = 20
 
    # Pre-step: empty cut data
    Cuts={}
    Cuts["Set"]=[]
    Cuts["Phi"]={}
    Cuts["rho"]={}
    Cuts["SOC_hat"]={}

    results = []

    for i in range(Iterations):
 
        # 1) Solve master problem (first stage-problem)
        m_1st = ModelSetUp_1st(batt, price_stage1, Cuts)
        Solve(m_1st)
        SOC_hat = m_1st.SOC_t_stage1[96].value
        UB = pyo.value(m_1st.obj) # Upper bound
        stage1_profit = UB - m_1st.alpha.value 
 
        #2) Solve one subproblem per scenario with SOC_96 fixed (SOC_hat)
        subproblems = {}
        for s in S:
            price_stage2 = list(scenarios_price[s]["Price"].values)
            m_2nd = ModelSetUp_2nd(batt, price_stage2, SOC_hat)
            Solve(m_2nd)
            subproblems[s] = m_2nd
 
        LB = stage1_profit + sum(pyo.value(subproblems[s].obj) for s in S)  # Lower bound
 
        #3) Create new cut for the master problem
        Cuts = Cut_manage(Cuts, subproblems, SOC_hat)
 
        #4) Save results 
        results = save_results(results, i, SOC_hat, UB, LB, subproblems, m_1st, m_1st.alpha.value)

        #5) Convergence check
        if UB - LB <= epsilon:
            break

    # Validation check: Check if the upper bound of alpha is reasonable
    check_alpha_bound(batt, scenarios_price, Delta, 1000000)  # Check if the upper bound of alpha is reasonable
