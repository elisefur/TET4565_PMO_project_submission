import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
import Read_data as rd
from dataclasses import dataclass
import matplotlib.pyplot as plt

#Making a Stochastic BESS model for 48 hours. 
#assumption: BESS is price-taker and all scenarios have equal probability. 

#-----------
#Battery-Paramters 
#------------

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

#--------
#Build model
#----------

def build_model(batt,price_stage1, scenarios_price): 

    #Creating model
    m=pyo.ConcreteModel()

    #Sets
    #Set of quarters
    S= list(range(1,6)) # List of the  5 scenarios. Starting at 1 to 5. 
    T_1=list(range(1,len(price_stage1)+1)) #list of quarters. Starting at 1, ending at 96.
    T_2= list(range(len(price_stage1)+1,len(price_stage1)+len(scenarios_price[S[0]])+1))#list of quarters. Starting at 97, ending at 192.

    m.T_1=pyo.Set(initialize=T_1)
    m.T_2=pyo.Set(initialize=T_2)
    m.S=pyo.Set(initialize=S)

    #Parameters
    DA_price_stage_1=dict(zip(T_1, price_stage1)) #DA-price [EUR/MWh] for first 24 h
    DA_price_stage_2={} #Making a Matrix. 
    for s in S:
        for t, price in zip(T_2, scenarios_price[s]["Price"]):
            DA_price_stage_2[(t, s)] = price 

    m.DA_price_stage_1=pyo.Param(m.T_1, initialize=DA_price_stage_1) # DA-price [EUR/MWh] in quarter t in T1
    m.DA_price_stage_2=pyo.Param(m.T_2, m.S, initialize=DA_price_stage_2)# DA-price [EUR/MWh] in quarter t in T2 for each scenario s in S. 
    
    m.pi = pyo.Param(m.S, initialize=0.2) #Equal probability for the 5 scenarios

    #Variables 
    m.p_ch_stage1=pyo.Var(m.T_1,within=pyo.NonNegativeReals) #Amount [MW] charged in quarter t in T1
    m.p_ch_stage2=pyo.Var(m.T_2, m.S, within=pyo.NonNegativeReals)#Amount [MW] charged in quarter t in T2 for each scenario s in S
    m.p_dis_stage1=pyo.Var(m.T_1, within=pyo.NonNegativeReals) #Amount [MW] discharged in quarter t in T1
    m.p_dis_stage2=pyo.Var(m.T_2, m.S, within=pyo.NonNegativeReals)#Amount [MW] discharged in quarter t in T2 for each scenario s in S
    m.SOC_t_stage1=pyo.Var(m.T_1,within=pyo.NonNegativeReals) #State of charge [-] in quarter t in T1
    m.SOC_t_stage2=pyo.Var(m.T_2, m.S, within=pyo.NonNegativeReals) #State of charge [-] in quarter t in T2 for each scenario s in S
    #Since we have non-neg DA-prices and includes effciency =! 100%, the model will never charge and discharge at the same time. Therefore no binary variable needed. 

    #Objective: Maximize profit = Revuenue - Cost
    def OBJ(m): 
        stage1= sum((m.p_dis_stage1[t]*m.DA_price_stage_1[t]-m.p_ch_stage1[t]*m.DA_price_stage_1[t])*Delta  for t in m.T_1)
        stage2= sum(m.pi[s]* sum((m.p_dis_stage2[t,s]*m.DA_price_stage_2[t,s]- m.p_ch_stage2[t,s]*m.DA_price_stage_2[t,s])*Delta  for t in m.T_2) for s in m.S)
        return stage1+ stage2

    m.OBJ=pyo.Objective(rule=OBJ, sense=pyo.maximize)

    #Constraints: 
    #Charge/ Discharge / SOC-limits: 
    def P_ch_limit_max_stage1(m,t):
        return m.p_ch_stage1[t] <= batt.P_max #Defined as non-neg, do not need lower limit of zero. 

    def P_ch_limit_max_stage2(m,t,s):
        return m.p_ch_stage2[t,s] <= batt.P_max 

    def P_dis_limit_max_stage1(m,t): 
        return m.p_dis_stage1[t] <= batt.P_max

    def P_dis_limit_max_stage2(m,t,s): 
        return m.p_dis_stage2[t,s] <= batt.P_max

    def SOC_limit_max_stage1(m,t): 
        return m.SOC_t_stage1[t] <= batt.SOC_max

    def SOC_limit_max_stage2(m,t,s): 
        return m.SOC_t_stage2[t,s] <= batt.SOC_max

    def SOC_limit_min_stage1(m,t): 
        return m.SOC_t_stage1[t] >= batt.SOC_min

    def SOC_limit_min_stage2(m,t,s): 
        return m.SOC_t_stage2[t,s] >= batt.SOC_min

    m.P_ch_limit_max_stage1=pyo.Constraint(m.T_1, rule=P_ch_limit_max_stage1)
    m.P_ch_limit_max_stage2=pyo.Constraint(m.T_2, m.S, rule=P_ch_limit_max_stage2)

    m.P_dis_limit_max_stage1=pyo.Constraint(m.T_1, rule=P_dis_limit_max_stage1)
    m.P_dis_limit_max_stage2=pyo.Constraint(m.T_2,m.S, rule=P_dis_limit_max_stage2)

    m.SOC_limit_max_stage1=pyo.Constraint(m.T_1, rule=SOC_limit_max_stage1)
    m.SOC_limit_max_stage2=pyo.Constraint(m.T_2, m.S, rule=SOC_limit_max_stage2)

    m.SOC_limit_min_stage1=pyo.Constraint(m.T_1, rule=SOC_limit_min_stage1)
    m.SOC_limit_min_stage2=pyo.Constraint(m.T_2, m.S, rule=SOC_limit_min_stage2)

    #Update state of charge: 
    #SOC_t= last state+ energien batteriet faktisk lader med - energien som faktisk tas ut av batteriet. 


    def SOC_update_stage1(m,t):
        if t==T_1[0]: #Start condition, t=1
            return m.SOC_t_stage1[t] == (batt.SOC_init + (m.p_ch_stage1[t]*batt.Eta_ch - m.p_dis_stage1[t]/batt.Eta_dis)*Delta/batt.E_max)
        
        else:   
            return m.SOC_t_stage1[t] == (m.SOC_t_stage1[t-1] + (m.p_ch_stage1[t]*batt.Eta_ch - m.p_dis_stage1[t]/batt.Eta_dis)*Delta/batt.E_max)

    def SOC_update_stage2(m,t,s):
        if t==T_2[0]: #Start condition, t=97
            return m.SOC_t_stage2[t,s] == (m.SOC_t_stage1[t-1] + (m.p_ch_stage2[t,s]*batt.Eta_ch - m.p_dis_stage2[t,s]/batt.Eta_dis)*Delta/batt.E_max)
        
        else:   
            return m.SOC_t_stage2[t,s] == (m.SOC_t_stage2[t-1,s] + (m.p_ch_stage2[t,s]*batt.Eta_ch - m.p_dis_stage2[t,s]/batt.Eta_dis)*Delta/batt.E_max)


    def SOC_end(m,s): 
        return m.SOC_t_stage2[len(m.T_1)+len(m.T_2),s]==batt.SOC_init

    m.SOC_update_stage1=pyo.Constraint(m.T_1, rule=SOC_update_stage1)
    m.SOC_update_stage2=pyo.Constraint(m.T_2,m.S, rule=SOC_update_stage2)
    m.SOC_end=pyo.Constraint(m.S, rule=SOC_end)

    return m


#-----
#Solve model 
#-------

def solve_model(m): 

    #Specify the solver 
    opt= SolverFactory('gurobi')

    #Solve the model 
    results= opt.solve(m,load_solutions=True) # Solving the model and loading the solution into the model.
    print(results)

    return results

def get_results(m): 


    objective_value = pyo.value(m.OBJ)
    print(f"Optimal Profit: {objective_value} EUR")

    
    results = {}
    for s in m.S:
        results[s] = pd.DataFrame({
            "Quarter": list(m.T_1) + list(m.T_2),
            "Price [EUR/MWh]": [pyo.value(m.DA_price_stage_1[t]) for t in m.T_1] + [pyo.value(m.DA_price_stage_2[t,s]) for t in m.T_2],
            "P_ch [MW]": [m.p_ch_stage1[t].value for t in m.T_1] + [m.p_ch_stage2[t,s].value for t in m.T_2],
            "P_dis [MW]": [m.p_dis_stage1[t].value for t in m.T_1] + [m.p_dis_stage2[t,s].value for t in m.T_2],
            "SOC [-]": [m.SOC_t_stage1[t].value for t in m.T_1] + [m.SOC_t_stage2[t,s].value for t in m.T_2],
        })
    return results


def save_results_to_excel(all_results, filename):
    with pd.ExcelWriter(filename) as writer:
        for s in all_results:
            all_results[s].to_excel(writer, sheet_name=f"Scenario {s}", index=False)
    print(f"Results saved to {filename}")



if __name__ == "__main__": #Only runs when this file is run directly, not imported

    batt = BatteryParams()
    m = build_model(batt, price_stage1, scenarios_price)
    solve_model(m)
    results = get_results(m)
    save_results_to_excel(results, "BESS_Stochastic_results.xlsx")