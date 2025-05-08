from gurobipy import GRB, Model

m = Model("orkun_exp")

x1 = m.addVar(lb = 0,ub = GRB.INFINITY, vtype = GRB.CONTINUOUS, name = "x1")
x2 = m.addVar(lb = 0,ub = GRB.INFINITY, vtype = GRB.CONTINUOUS, name = "x2")

m.addConstr(4 * x1 + 4 * x2 + 0 <= 70)
m.addConstr(3 * x1 + 2 * x2 + 2 <= 50)

# m.

m.optimize()

m.printAttr("objVal")
