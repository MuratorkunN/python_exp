import gurobipy as gp

m = gp.Model("Knapsack_01")

item_set = range(6)

values = {0:175, 1:90, 2:20, 3:50, 4:10, 5:200}
weights = {0:10, 1:9, 2:4, 3:2, 4:1, 5:20}
capacity = 20

x = m.addVars(item_set, vtype=gp.GRB.BINARY, name = "item")

obj_expr = gp.quicksum(values[item] * x[item] for item in item_set)

m.setObjective(obj_expr, sense= gp.GRB.MAXIMIZE)

m.addConstr(gp.quicksum(weights[item] * x[item] for item in item_set) <= capacity)

m.optimize()
m.printAttr("ObjVal")
m.printAttr("x")