# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:23:42 2026

@author: JoPHOBEA
"""

from pyomo.environ import *

# ----------------------------
# DONNÉES
# ----------------------------
T = range(24)

load = [5]*24

renewable = [2,2,3,5,7,8,10,12,10,8,6,5,
             4,3,3,2,2,1,1,0,0,1,2,3]

price_buy = [0.2]*24
price_sell = [0.1]*24  # tarif de revente

# ----------------------------
# COÛTS INVESTISSEMENT
# ----------------------------
capex_energy = 400   # €/kWh
capex_power  = 250   # €/kW

eta_ch = 0.95
eta_dis = 0.95

E_init_ratio = 0.5  # SOC initial fraction

# ----------------------------
# MODÈLE
# ----------------------------
model = ConcreteModel()

model.T = Set(initialize=T)

# ----------------------------
# VARIABLES DESIGN (MILP)
# ----------------------------
model.E_bat = Var(domain=NonNegativeReals)  # kWh
model.P_bat = Var(domain=NonNegativeReals)  # kW

# ----------------------------
# VARIABLES OPÉRATIONNELLES
# ----------------------------
model.grid_import = Var(model.T, domain=NonNegativeReals)
model.grid_export = Var(model.T, domain=NonNegativeReals)

model.charge = Var(model.T, domain=NonNegativeReals)
model.discharge = Var(model.T, domain=NonNegativeReals)

model.soc = Var(model.T, domain=NonNegativeReals)

# binaries pour exclusivité
model.u_ch = Var(model.T, domain=Binary)
model.u_dis = Var(model.T, domain=Binary)

# ----------------------------
# OBJECTIF
# ----------------------------
def objective_rule(m):
    capex = capex_energy * m.E_bat + capex_power * m.P_bat

    opex = sum(price_buy[t] * m.grid_import[t] -
               price_sell[t] * m.grid_export[t]
               for t in m.T)

    return capex + opex

model.obj = Objective(rule=objective_rule, sense=minimize)

# ----------------------------
# CONTRAINTES
# ----------------------------

def balance_rule(m, t):
    return (renewable[t]
            + m.discharge[t]
            + m.grid_import[t]
            ==
            load[t]
            + m.charge[t]
            + m.grid_export[t])

model.balance = Constraint(model.T, rule=balance_rule)

# ----------------------------
# CONTRAINTES BATTERIE
# ----------------------------

def charge_limit(m, t):
    return m.charge[t] <= m.P_bat * m.u_ch[t]

def discharge_limit(m, t):
    return m.discharge[t] <= m.P_bat * m.u_dis[t]

def exclusivity(m, t):
    return m.u_ch[t] + m.u_dis[t] <= 1

model.c1 = Constraint(model.T, rule=charge_limit)
model.c2 = Constraint(model.T, rule=discharge_limit)
model.c3 = Constraint(model.T, rule=exclusivity)

# ----------------------------
# DYNAMIQUE SOC
# ----------------------------

def soc_rule(m, t):
    if t == 0:
        return m.soc[t] == (E_init_ratio * m.E_bat +
                            eta_ch * m.charge[t] -
                            m.discharge[t] / eta_dis)
    else:
        return m.soc[t] == (m.soc[t-1] +
                            eta_ch * m.charge[t] -
                            m.discharge[t] / eta_dis)

model.soc_dyn = Constraint(model.T, rule=soc_rule)

# ----------------------------
# LIMITES SOC
# ----------------------------

def soc_max(m, t):
    return m.soc[t] <= m.E_bat

model.soc_lim = Constraint(model.T, rule=soc_max)

# ----------------------------
# SOLVEUR
# ----------------------------
solver = SolverFactory('glpk')  # ou 'cbc', 'gurobi'
solver.solve(model, tee=True)

# ----------------------------
# RÉSULTATS
# ----------------------------

print("\n===== DESIGN OPTIMAL =====")
print("Batterie énergie (kWh):", model.E_bat.value)
print("Batterie puissance (kW):", model.P_bat.value)

print("\n===== EXEMPLE D'EXPLOITATION =====")
for t in T:
    print(
        t,
        "Import:", model.grid_import[t].value,
        "Export:", model.grid_export[t].value,
        "SOC:", model.soc[t].value
    )

print("\nCoût total =", model.obj(), "€")