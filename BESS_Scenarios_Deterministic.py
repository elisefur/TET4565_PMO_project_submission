import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
import Read_data as rd
from dataclasses import dataclass
import matplotlib.pyplot as plt

#Making a determinstic BESS model for 48 hours. 
#Assumtions: Perfect foresight, BESS is price-taker. 

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

#stage 2 prices: 
scenarios_price= rd.get_stage2_prices(df, rd.STAGE2_DAYS) #Dictonary of stage 2 prices. 


#-----------
#Build model
#---------

def build_model(batt,price_stage1,price_stage2): 
    

    #Creating model
    m=pyo.ConcreteModel()

    #Sets
    #Set of quarters
    T_1=list(range(1,len(price_stage1)+1)) #list of quarters. Starting at 1, ending at 96.
    T_2= list(range(len(price_stage1)+1,len(price_stage1)+len(price_stage2)+1))#list of quarters. Starting at 97, ending at 192.
    m.T_1=pyo.Set(initialize=T_1)
    m.T_2=pyo.Set(initialize=T_2)


    #Data
    DA_price_stage_1=dict(zip(T_1, price_stage1)) #DA-price [EUR/MWh] for first 24 h
    DA_price_stage_2=dict(zip(T_2, price_stage2)) #DA-price [EUR/MWh] for next 24 h 
    m.DA_price_stage_1=pyo.Param(m.T_1, initialize=DA_price_stage_1) # DA-price [EUR/MWh] in quarter t in T1
    m.DA_price_stage_2=pyo.Param(m.T_2, initialize=DA_price_stage_2)# DA-price [EUR/MWh] in quarter t in T2


    #Variables 
    m.p_ch_stage1=pyo.Var(m.T_1,within=pyo.NonNegativeReals) #Amount [MW] charged in quarter t in T1
    m.p_ch_stage2=pyo.Var(m.T_2,within=pyo.NonNegativeReals)#Amount [MW] charged in quarter t in T2
    m.p_dis_stage1=pyo.Var(m.T_1,within=pyo.NonNegativeReals) #Amount [MW] discharged in quarter t in T1
    m.p_dis_stage2=pyo.Var(m.T_2,within=pyo.NonNegativeReals)#Amount [MW] discharged in quarter t in T2
    m.SOC_t_stage1=pyo.Var(m.T_1,within=pyo.NonNegativeReals) #State of charge [-] in quarter t in T1
    m.SOC_t_stage2=pyo.Var(m.T_2,within=pyo.NonNegativeReals) #State of charge [-] in quarter t in T2. 
    #Since we have non-neg DA-prices and includes effciency =! 100%, the model will never charge and discharge at the same time. Therefore no binary variable needed. 

    #Objective: Maximize profit = Revuenue - Cost
    def OBJ(m): 
        revenue_stage1= sum(m.p_dis_stage1[t]*Delta*m.DA_price_stage_1[t] for t in m.T_1)
        revenue_stage2= sum(m.p_dis_stage2[t]*Delta*m.DA_price_stage_2[t] for t in m.T_2)
        cost_stage1 = sum(m.p_ch_stage1[t]*Delta*m.DA_price_stage_1[t] for t in m.T_1)
        cost_stage2 = sum(m.p_ch_stage2[t]*Delta*m.DA_price_stage_2[t] for t in m.T_2)
        return revenue_stage1 + revenue_stage2 - cost_stage1 - cost_stage2

    m.OBJ=pyo.Objective(rule=OBJ, sense=pyo.maximize)

    #Constraints: 
    #Charge/ Discharge / SOC-limits: 
    def P_ch_limit_max_stage1(m,t):
        return m.p_ch_stage1[t] <= batt.P_max #Defined as non-neg, do not need lower limit of zero. 

    def P_ch_limit_max_stage2(m,t):
        return m.p_ch_stage2[t] <= batt.P_max 

    def P_dis_limit_max_stage1(m,t): 
        return m.p_dis_stage1[t] <= batt.P_max

    def P_dis_limit_max_stage2(m,t): 
        return m.p_dis_stage2[t] <= batt.P_max

    def SOC_limit_max_stage1(m,t): 
        return m.SOC_t_stage1[t] <= batt.SOC_max

    def SOC_limit_max_stage2(m,t): 
        return m.SOC_t_stage2[t] <= batt.SOC_max

    def SOC_limit_min_stage1(m,t): 
        return m.SOC_t_stage1[t] >= batt.SOC_min

    def SOC_limit_min_stage2(m,t): 
        return m.SOC_t_stage2[t] >= batt.SOC_min

    m.P_ch_limit_max_stage1=pyo.Constraint(m.T_1, rule=P_ch_limit_max_stage1)
    m.P_ch_limit_max_stage2=pyo.Constraint(m.T_2, rule=P_ch_limit_max_stage2)

    m.P_dis_limit_max_stage1=pyo.Constraint(m.T_1, rule=P_dis_limit_max_stage1)
    m.P_dis_limit_max_stage2=pyo.Constraint(m.T_2, rule=P_dis_limit_max_stage2)

    m.SOC_limit_max_stage1=pyo.Constraint(m.T_1, rule=SOC_limit_max_stage1)
    m.SOC_limit_max_stage2=pyo.Constraint(m.T_2, rule=SOC_limit_max_stage2)

    m.SOC_limit_min_stage1=pyo.Constraint(m.T_1, rule=SOC_limit_min_stage1)
    m.SOC_limit_min_stage2=pyo.Constraint(m.T_2, rule=SOC_limit_min_stage2)

    #Update state of charge: 
    #SOC_t= last state+ energien batteriet faktisk lader med - energien som faktisk tas ut av batteriet. 


    def SOC_update_stage1(m,t):
        if t==T_1[0]: #Start condition, t=1
            return m.SOC_t_stage1[t] == (batt.SOC_init + (m.p_ch_stage1[t]*batt.Eta_ch - m.p_dis_stage1[t]/batt.Eta_dis)*Delta/batt.E_max)
        
        else:   
            return m.SOC_t_stage1[t] == (m.SOC_t_stage1[t-1] + (m.p_ch_stage1[t]*batt.Eta_ch - m.p_dis_stage1[t]/batt.Eta_dis)*Delta/batt.E_max)

    def SOC_update_stage2(m,t):
        if t==T_2[0]: #Start condition, t=97
            return m.SOC_t_stage2[t] == (m.SOC_t_stage1[t-1] + (m.p_ch_stage2[t]*batt.Eta_ch - m.p_dis_stage2[t]/batt.Eta_dis)*Delta/batt.E_max)
        
        else:   
            return m.SOC_t_stage2[t] == (m.SOC_t_stage2[t-1] + (m.p_ch_stage2[t]*batt.Eta_ch - m.p_dis_stage2[t]/batt.Eta_dis)*Delta/batt.E_max)


    def SOC_end(m): 
        return m.SOC_t_stage2[len(m.T_1)+len(m.T_2)]==batt.SOC_init

    m.SOC_update_stage1=pyo.Constraint(m.T_1, rule=SOC_update_stage1)
    m.SOC_update_stage2=pyo.Constraint(m.T_2, rule=SOC_update_stage2)
    m.SOC_end=pyo.Constraint(rule=SOC_end)

    return m 

#-------
#Solve model
#-----------

def solve_model(m): 
    #Specify the solver 
    opt= SolverFactory('gurobi')

    #Solve the model 
    results= opt.solve(m,load_solutions=True) # Solving the model and loading the solution into the model.
    print(results)
    return results

#----------
#Results 
#---------

def get_results(m):



    objective_value = pyo.value(m.OBJ)
    print(f"Optimal Profit: {objective_value} EUR")

    T = list(m.T_1) + list(m.T_2)
    price = [pyo.value(m.DA_price_stage_1[t]) for t in m.T_1] + [pyo.value(m.DA_price_stage_2[t]) for t in m.T_2]
    p_ch = [m.p_ch_stage1[t].value for t in m.T_1] + [m.p_ch_stage2[t].value for t in m.T_2]
    p_dis = [m.p_dis_stage1[t].value for t in m.T_1] + [m.p_dis_stage2[t].value for t in m.T_2]
    SOC = [m.SOC_t_stage1[t].value for t in m.T_1] + [m.SOC_t_stage2[t].value for t in m.T_2]
    
    results_df = pd.DataFrame({
    "Quarter": T,
    "Price [EUR/MWh]": price,
    "P_ch [MW]": p_ch,
    "P_dis [MW]": p_dis,
    "SOC [-]": SOC,
    })

    return results_df

def save_results_to_excel(all_results, filename):
    with pd.ExcelWriter(filename) as writer:
        for s in all_results:
            all_results[s].to_excel(writer, sheet_name=f"Scenario {s}", index=False)
    print(f"Results saved to {filename}")



if __name__== "__main__": # Only runs when this file is run directly, not imported

    batt=BatteryParams()

    all_results = {}   # Dictionary to store results_df for each scenario

    for s in scenarios_price: 
        price_stage2= list(scenarios_price[s]["Price"].values) #Extracting scenario s by key.
        m=build_model(batt, price_stage1,price_stage2)
        solve_model(m)
        print(f"-------------SCENARIO {s}------------------")
        results_df=get_results(m)
        all_results[s] = results_df

    save_results_to_excel(all_results, "BESS_Scenarios_Deterministic_results.xlsx")