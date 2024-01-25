import os
import numpy as np
import math
import pyomo.environ as pyo
import pyomo.opt as po


#  Define pyomo solver

solvername='glpk'
solverpath_exe='C:\\glpk\\w64\\glpsol'

solver=pyo.SolverFactory(solvername,executable=solverpath_exe)



# Remark: In some of the constraints, you will notice that the indices [q] and [q-1] are used for the same quarter. This is due to Python lists counting from 0 and Pyomo Variable lists counting from 1.  





def step1_optimize_DAA(n_cycles, energy_cap, power_cap, DAA_price_vector):

    """
    Calculates optimal charge/discharge schedule on the day-ahead for a given 96-d DAA_price_vector.

    Parameters:
    - n_cycles: Maximum number of allowed cycles
    - energy_cap: Energy capacity
    - power_cap: Power capacity
    - DAA_price_vector: 96-dimensional DAA price vector

    Returns:
    - step1_soc_DAA: Resulting state of charge schedule
    - step1_cha_DAA: Resulting charge schedule / Positions on DA Auction
    - step1_dis_DAA: Resulting discharge schedule / Positions on DA Auction
    - step1_profit_DAA: Profit from Day-ahead auction trades
    """
    


    # Initialize pyomo model:

    model = pyo.ConcreteModel()



    # Set parameters:

    # Number of hours
    model.H = pyo.RangeSet(0,len(DAA_price_vector)/4-1) 

    # Number of quarters
    model.Q = pyo.RangeSet(1,len(DAA_price_vector))         

    # Number of quarters plus 1
    model.Q_plus_1 = pyo.RangeSet(1,len(DAA_price_vector)+1)  

    # Daily discharged energy limit
    volume_limit = energy_cap * n_cycles                    



    # Initialize variables:

    # State of charge
    model.soc = pyo.Var(model.Q_plus_1, domain=pyo.Reals)

    # Charges on the Day-ahead auction
    model.cha_DAA = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0,1)) 

    # Discharges on the Day-ahead auction
    model.dis_DAA = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0,1))



    # Set Constraints: 

    # (Constr. 1.1) State of charge can never be higher than Energy Capacity:
    def constr_1_1(model, i):
        return model.soc[i] <= energy_cap
    model.constr_1_1 = pyo.Constraint(model.Q_plus_1, rule=constr_1_1)

    # (Constr. 1.2) State of charge can never be less than 0.
    def constr_1_2(model, i):
        return model.soc[i] >= 0
    model.constr_1_2 = pyo.Constraint(model.Q_plus_1, rule=constr_1_2)

    # (Constr. 1.3) State of charge at first quarter must be 0.
    model.constr_1_3 = pyo.Constraint(rule = model.soc[1] == 0)

    # (Constr. 1.4) State of charge at quarter 97 (i.e. first quarter of next day) must be 0.
    model.constr_1_4 = pyo.Constraint(rule = model.soc[len(DAA_price_vector)+1] == 0)

    # (Constr 1.5) The state of charge of each quarter equals the state if charge of the previous quarter plus charges minus discharges.
    def constr_1_5(model, i):
        return model.soc[i+1] == model.soc[i] + power_cap/4  * model.cha_DAA[i] - power_cap/4 * model.dis_DAA[i]
    model.constr_1_5 = pyo.Constraint(model.Q, rule=constr_1_5)

    # (Constr. 1.6) Sum of all charges has to be below the daily limit
    model.constr_1_6 = pyo.Constraint(expr = (sum(model.cha_DAA[i]  * power_cap/4 for i in model.Q) <= volume_limit))

    # (Constr. 1.7) Sum of all discharges has to be below the daily limit
    model.constr_1_7 = pyo.Constraint(expr = (sum(model.dis_DAA[i] * power_cap/4 for i in model.Q) <= volume_limit))

    # (Constr. 1.8) On the DA Auction, positions in all 4 quarters of the hour have to be identical, since in practice trades are taken in hourly blocks.
    def constr_1_8_1(model, i):
        return model.cha_DAA[4*i+1] == model.cha_DAA[4*i+2]
    
    def constr_1_8_2(model, i):
        return model.cha_DAA[4*i+2] == model.cha_DAA[4*i+3]
    
    def constr_1_8_3(model, i):
        return model.cha_DAA[4*i+3] == model.cha_DAA[4*i+4]

    model.constr_1_8_1 = pyo.Constraint(model.H, rule=constr_1_8_1)
    model.constr_1_8_2 = pyo.Constraint(model.H, rule=constr_1_8_2)
    model.constr_1_8_3 = pyo.Constraint(model.H, rule=constr_1_8_3)

    # (Constr. 1.9) On the DA Auction, positions in all 4 quarters of the hour have to be identical, since in practice trades are taken in hourly blocks.
    def constr_1_9_1(model, i):
        return model.dis_DAA[4*i+1] == model.dis_DAA[4*i+2]
    
    def constr_1_9_2(model, i):
        return model.dis_DAA[4*i+2] == model.dis_DAA[4*i+3]
    
    def constr_1_9_3(model, i):
        return model.dis_DAA[4*i+3] == model.dis_DAA[4*i+4]
    
    model.constr_1_9_1 = pyo.Constraint(model.H, rule=constr_1_9_1)
    model.constr_1_9_2 = pyo.Constraint(model.H, rule=constr_1_9_2)
    model.constr_1_9_3 = pyo.Constraint(model.H, rule=constr_1_9_3)



    # Define objective function and solve the optimization problem

    # (Objective) Find charge-discharge-schedule, which generates highest profit.
    model.obj = pyo.Objective(expr=sum(power_cap/4 * DAA_price_vector[i-1] * (model.dis_DAA[i] - model.cha_DAA[i])  for i in model.Q), sense=pyo.maximize)

    # Solve
    solver.solve(model, timelimit=5)



    # Retrieve arrays of resulting optimal soc/charge/discharge schedules after the DA Auction:
    step1_soc_DAA = [model.soc[i].value for i in range(1, len(DAA_price_vector) + 1)]
    step1_cha_DAA = [model.cha_DAA[i].value for i in range(1, len(DAA_price_vector) + 1)]
    step1_dis_DAA = [model.dis_DAA[i].value for i in range(1, len(DAA_price_vector) + 1)]

    # Calculate profit from Day-ahead auction trades:
    step1_profit_DAA = sum([power_cap/4 * DAA_price_vector[i] * (step1_dis_DAA[i] -  step1_cha_DAA[i]) for i in range(len(DAA_price_vector))])

    return(step1_soc_DAA, step1_cha_DAA, step1_dis_DAA, step1_profit_DAA)





def step2_optimize_IDA(n_cycles, energy_cap, power_cap, IDA_price_vector, step1_cha_DAA, step1_dis_DAA):

    """
    Calculates optimal charge/discharge schedule on the day-ahead for a given 96-d IDA_price_vector.

    Parameters:
    - n_cycles: Maximum number of allowed cycles
    - energy_cap: Energy capacity
    - power_cap: Power capacity
    - IDA_price_vector: 96-dimensional IDA price vector
    - step1_cha_DAA: Previous Buys on the Day-Ahead auction
    - step1_dis_DAA: Previous Sells on the Day-Ahead auction

    Returns:
    - step2_soc_IDA: Resulting state of charge schedule
    - step2_cha_IDA: Resulting charges on ID Auction
    - step2_dis_IDA: Resulting discharges on ID Auction
    - step2_cha_IDA_close: Resulting charges on ID Auction to close previous DA Auction positions
    - step2_dis_IDA_close: Resulting discharge on ID Auction to close previous DA Auction positions
    - step2_profit_IDA: Profit from Day-ahead auction trades
    - step2_cha_DAAIDA: Combined charges from DA Auction and ID Auction
    - step2_dis_DAAIDA: Combined discharges from DA Auction and ID Auction
    """



    # Initialize pyomo model:

    model = pyo.ConcreteModel()



    # Set parameters:

    # Number of hours
    model.H = pyo.RangeSet(0,len(IDA_price_vector)/4-1) 

    # Number of quarters
    model.Q = pyo.RangeSet(1,len(IDA_price_vector))         

    # Number of quarters plus 1
    model.Q_plus_1 = pyo.RangeSet(1,len(IDA_price_vector)+1)  

    # Daily discharged energy limit
    volume_limit = energy_cap * n_cycles 



    # Initialize variables:

    # State of charge
    model.soc = pyo.Var(model.Q_plus_1, domain=pyo.Reals)

    # Charges on the intraday auction
    model.cha_IDA = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0,1)) 

    # Discharges on the intraday auction
    model.dis_IDA = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0,1))

    # Charges on the intraday auction to close previous positions from the day-ahead auction
    model.cha_IDA_close = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0,1)) 

    # Charges on the intraday auction to close previous positions from the day-ahead auction
    model.dis_IDA_close = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0,1))



    # Set Constraints: 

    # (Constr. 2.1) State of charge can never be higher than Energy Capacity:
    def constr_2_1(model, i):
        return model.soc[i] <= energy_cap
    model.constr_2_1 = pyo.Constraint(model.Q_plus_1, rule=constr_2_1)

    # (Constr. 2.2) State of charge can never be less than 0.
    def constr_2_2(model, i):
        return model.soc[i] >= 0
    model.constr_2_2 = pyo.Constraint(model.Q_plus_1, rule=constr_2_2)

    # (Constr. 2.3) State of charge at first quarter must be 0.
    model.constr_2_3 = pyo.Constraint(rule = model.soc[1] == 0)

    # (Constr. 2.4) State of charge at quarter 97 (i.e. first quarter of next day) must be 0.
    model.constr_2_4 = pyo.Constraint(rule = model.soc[len(IDA_price_vector)+1] == 0)

    # (Constr. 2.5) The state of charge of each quarter equals the state if charge of the previous quarter plus charges minus discharges.
    def constr_2_5(model, i):
        return model.soc[i+1] == model.soc[i] + power_cap/4  * (model.cha_IDA[i] - model.dis_IDA[i] + model.cha_IDA_close[i] - model.dis_IDA_close[i] + step1_cha_DAA[i-1] - step1_dis_DAA[i-1])
    model.constr_2_5 = pyo.Constraint(model.Q, rule=constr_2_5)

    # (Constr. 2.6) Sum of all charges has to be below the daily limit
    model.constr_2_6 = pyo.Constraint(expr = ((np.sum(step1_cha_DAA) + sum(model.cha_IDA[i] for i in model.Q) - sum(model.dis_IDA_close[i] for i in model.Q)) * power_cap/4 <= volume_limit))

    # (Constr. 2.7) Sum of all discharges has to be below the daily limit
    model.constr_2_7 = pyo.Constraint(expr = ((np.sum(step1_dis_DAA) + sum(model.dis_IDA[i] for i in model.Q) - sum(model.cha_IDA_close[i] for i in model.Q)) * power_cap/4 <= volume_limit))

    # (Constr. 2.8) cha_IDA_close can only close or reduce existing dis_DAA positions. They can only be placed, where dis_DAA positions exist. 
    def constr_2_8(model, i):
        return model.cha_IDA_close[i] <= step1_dis_DAA[i-1]
    model.constr_2_8 = pyo.Constraint(model.Q, rule=constr_2_8)

    # (Constr. 2.9) dis_IDA_close can only close or reduce existing cha_DAA positions. They can only be placed, where cha_DAA positions exist. 
    def constr_2_9(model, i):
        return model.dis_IDA_close[i] <= step1_cha_DAA[i-1]
    model.constr_2_9 = pyo.Constraint(model.Q, rule=constr_2_9)

    # (Constr. 2.10) Sum of cha_IDA[i] and cha_DAA[i] has to be less or equal to 1.
    def constr_2_10(model, i):
        return model.cha_IDA[i] + step1_cha_DAA[i-1] <= 1
    model.constr_2_10 = pyo.Constraint(model.Q, rule=constr_2_10)

   # (Constr. 2.11) Sum of dis_IDA[i] and dis_DAA[i] has to be less or equal to 1.
    def constr_2_11(model, i):
        return model.dis_IDA[i] + step1_dis_DAA[i-1] <= 1
    model.constr_2_11 = pyo.Constraint(model.Q, rule=constr_2_11)



    # Define objective function and solve the optimization problem

    # (Objective) define objective function.
    model.obj = pyo.Objective(expr=sum(IDA_price_vector[i-1] * power_cap/4 * (model.dis_IDA[i] + model.dis_IDA_close[i] - model.cha_IDA[i] - model.cha_IDA_close[i]) for i in model.Q), sense=pyo.maximize)

    # Solve
    solver.solve(model, timelimit=5)



    # Retrieve arrays of resulting optimal soc/charge/discharge schedules after the ID Auction:
    step2_soc_IDA = [model.soc[i].value for i in range(1, len(IDA_price_vector) + 1)]
    step2_cha_IDA = [model.cha_IDA[i].value for i in range(1, len(IDA_price_vector) + 1)]                   
    step2_dis_IDA = [model.dis_IDA[i].value for i in range(1, len(IDA_price_vector) + 1)]                   
    step2_cha_IDA_close = [model.cha_IDA_close[i].value for i in range(1, len(IDA_price_vector) + 1)]       
    step2_dis_IDA_close = [model.dis_IDA_close[i].value for i in range(1, len(IDA_price_vector) + 1)]       

    # Calculate profit from Day-ahead auction trades:   
    step2_profit_IDA = np.sum(((np.asarray(step2_dis_IDA) + step2_dis_IDA_close) - (np.asarray(step2_cha_IDA) + step2_cha_IDA_close)) * IDA_price_vector) * power_cap/4
    
    # Calculate total physical charge discharge schedules of combined day-ahead and intraday auction trades:
    step2_cha_DAAIDA = np.asarray(step1_cha_DAA) - step2_dis_IDA_close + step2_cha_IDA
    step2_dis_DAAIDA = np.asarray(step1_dis_DAA) - step2_cha_IDA_close + step2_dis_IDA

    return(step2_soc_IDA, step2_cha_IDA, step2_dis_IDA, step2_cha_IDA_close, step2_dis_IDA_close, step2_profit_IDA, step2_cha_DAAIDA, step2_dis_DAAIDA)
    




def step3_optimize_IDC(n_cycles, energy_cap, power_cap, IDC_price_vector, step2_cha_DAAIDA, step2_dis_DAAIDA):
    
    """
    Calculates optimal charge/discharge schedule on the day-ahead for a given 96-d IDC_price_vector.

    Parameters:
    - n_cycles: Maximum number of allowed cycles
    - energy_cap: Energy capacity
    - power_cap: Power capacity
    - IDA_price_vector: 96-dimensional IDA price vector
    - step2_cha_DAAIDA: Previous combined Buys on the DA Auction and ID Auction
    - step2_dis_DAAIDA: Previous combined Sells on the DA Auction and ID Auction

    Returns:
    - step3_soc_IDC: Resulting state of charge schedule
    - step3_cha_IDC: Resulting charges on ID Continuous
    - step3_dis_IDC: Resulting discharges on ID Continuous
    - step3_cha_IDC_close: Resulting charges on ID Continuous to close previous DA or ID Auction positions
    - step3_dis_IDC_close: Resulting discharge on ID Continuous to close previous DA or ID Auction positions
    - step3_profit_IDC: Profit from Day-ahead auction trades
    - step3_cha_DAAIDAIDC: Combined charges from DA Auction, ID Auction and ID Continuous
    - step3_dis_DAAIDAIDC: Combined discharges from DA Auction, ID Auction and ID Continuous
    """



    # Initialize pyomo model:

    model = pyo.ConcreteModel()



    # Set parameters:

    # Number of hours
    model.H = pyo.RangeSet(0,len(IDC_price_vector)/4-1) 

    # Number of quarters
    model.Q = pyo.RangeSet(1,len(IDC_price_vector))         

    # Number of quarters plus 1
    model.Q_plus_1 = pyo.RangeSet(1,len(IDC_price_vector)+1)  

    # Daily discharged energy limit
    volume_limit = energy_cap * n_cycles 



    # Initialize variables:

    # State of charge
    model.soc = pyo.Var(model.Q_plus_1, domain=pyo.Reals)

    # Charges on the intraday auction
    model.cha_IDC = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0,1)) 

    # Discharges on the intraday auction
    model.dis_IDC = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0,1))

    # Charges on the intraday auction to close previous positions from the day-ahead auction
    model.cha_IDC_close = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0,1)) 

    # Charges on the intraday auction to close previous positions from the day-ahead auction
    model.dis_IDC_close = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0,1))



    # Set Constraints: 

    # (Constr. 3.1) State of charge can never be higher than Energy Capacity:
    def constr_3_1(model, i):
        return model.soc[i] <= energy_cap
    model.constr_3_1 = pyo.Constraint(model.Q_plus_1, rule=constr_3_1)

    # (Constr. 3.2) State of charge can never be less than 0.
    def constr_3_2(model, i):
        return model.soc[i] >= 0
    model.constr_3_2 = pyo.Constraint(model.Q_plus_1, rule=constr_3_2)

    # (Constr. 3.3) State of charge at first quarter must be 0.
    model.constr_3_3 = pyo.Constraint(rule = model.soc[1] == 0)

    # (Constr. 3.4) State of charge at quarter 97 (i.e. first quarter of next day) must be 0.
    model.constr_3_4 = pyo.Constraint(rule = model.soc[len(IDC_price_vector)+1] == 0)

    # (Constr. 3.5) The state of charge of each quarter equals the state if charge of the previous quarter plus charges minus discharges.
    def constr_3_5(model, i):
        return model.soc[i+1] == model.soc[i] + power_cap/4  * (model.cha_IDC[i] - model.dis_IDC[i] + model.cha_IDC_close[i] - model.dis_IDC_close[i] + step2_cha_DAAIDA[i-1] - step2_dis_DAAIDA[i-1])
    model.constr_3_5 = pyo.Constraint(model.Q, rule=constr_3_5)

    # (Constr. 3.6) Sum of all charges has to be below the daily limit
    model.constr_3_6 = pyo.Constraint(expr = ((np.sum(step2_dis_DAAIDA) + sum(model.dis_IDC[i] for i in model.Q) - sum(model.cha_IDC_close[i] for i in model.Q)) * power_cap/4 <= volume_limit))

    # (Constr. 3.7) Sum of all discharges has to be below the daily limit
    model.constr_3_7 = pyo.Constraint(expr = ((np.sum(step2_dis_DAAIDA) + sum(model.dis_IDC[i] for i in model.Q) - sum(model.cha_IDC_close[i] for i in model.Q)) * power_cap/4 <= volume_limit))

    # (Constr. 3.8) cha_IDC_close can only close or reduce existing dis_DAAIDA positions. They can only be placed, where dis_DAAIDA positions exist. 
    def constr_3_8(model, i):
        return model.cha_IDC_close[i] <= step2_dis_DAAIDA[i-1]
    model.constr_3_8 = pyo.Constraint(model.Q, rule=constr_3_8)

    # (Constr. 3.9) dis_IDC_close can only close or reduce existing cha_DAAIDA positions. They can only be placed, where cha_DAAIDA positions exist. 
    def constr_3_9(model, i):
        return model.dis_IDC_close[i] <= step2_cha_DAAIDA[i-1]
    model.constr_3_9 = pyo.Constraint(model.Q, rule=constr_3_9)

    # (Constr. 3.10) Sum of cha_IDC[i] and cha_DAAIDA[i] has to be less or equal to 1.
    def constr_3_10(model, i):
        return model.cha_IDC[i] + step2_cha_DAAIDA[i-1] <= 1
    model.constr_3_10 = pyo.Constraint(model.Q, rule=constr_3_10)

   # (Constr. 2.11) Sum of dis_IDC[i] and dis_DAAIDA[i] has to be less or equal to 1.
    def constr_3_11(model, i):
        return model.dis_IDC[i] + step2_dis_DAAIDA[i-1] <= 1
    model.constr_3_11 = pyo.Constraint(model.Q, rule=constr_3_11)



    # Define objective function and solve the optimization problem

    # (Objective) define objective function.
    model.obj = pyo.Objective(expr=sum([IDC_price_vector[i-1] * power_cap/4 * (model.dis_IDC[i]+model.dis_IDC_close[i]-model.cha_IDC[i]-model.cha_IDC_close[i]) for i in model.Q]), sense=pyo.maximize) 

    # Solve
    solver.solve(model, timelimit=5)



    # Retrieve arrays of resulting optimal soc/charge/discharge schedules after the ID Auction:
    step3_soc_IDC = [model.soc[i].value for i in range(1, len(IDC_price_vector) + 1)]
    step3_cha_IDC = [model.cha_IDC[i].value for i in range(1, len(IDC_price_vector) + 1)]                   
    step3_dis_IDC = [model.dis_IDC[i].value for i in range(1, len(IDC_price_vector) + 1)]                   
    step3_cha_IDC_close = [model.cha_IDC_close[i].value for i in range(1, len(IDC_price_vector) + 1)]       
    step3_dis_IDC_close = [model.dis_IDC_close[i].value for i in range(1, len(IDC_price_vector) + 1)]       

    # Calculate profit from Day-ahead auction trades:   
    step3_profit_IDC = np.sum(((np.asarray(step3_dis_IDC) + step3_dis_IDC_close) - (np.asarray(step3_cha_IDC) + step3_cha_IDC_close)) * IDC_price_vector) * power_cap/4
    
    # Calculate total physical charge discharge schedules of combined day-ahead and intraday auction trades:
    step3_cha_DAAIDAIDC = np.asarray(step2_cha_DAAIDA) - step3_dis_IDC_close + step3_cha_IDC
    step3_dis_DAAIDAIDC = np.asarray(step2_dis_DAAIDA) - step3_cha_IDC_close + step3_dis_IDC

    return(step3_soc_IDC, step3_cha_IDC, step3_dis_IDC, step3_cha_IDC_close, step3_dis_IDC_close, step3_profit_IDC, step3_cha_DAAIDAIDC, step3_dis_DAAIDAIDC)
    


