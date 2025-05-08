from gurobipy import GRB, Model, quicksum

m = Model("lab")

table = [[71, 40, 36, 64, 32, 43, 39, 50],
         [56, 87, 82, 22, 83, 80, 35, 57],
         [45, 68, 20, 19, 52, 90, 75, 15],
         [99, 46, 53, 49, 41, 44, 78, 62],
         [96, 98, 30, 97, 10, 94, 73, 17],
         [79, 86, 12, 60, 77, 69, 51, 31]]

i_set, j_set = range(len(table)), range(len(table[0]))

x = m.addVars(i_set, j_set, vtype=GRB.BINARY, name="x")

obj_expr = quicksum(table[i][j] * x[(i,j)] for  i in i_set for j in j_set)

m.setObjective(obj_expr, sense=GRB.MAXIMIZE)

for i in i_set:
    lhs = quicksum(x[(i,j)] for j in j_set)
    m.addConstr(lhs <= 2)

for j in j_set:
    lhs = quicksum(x[(i,j)] for i in i_set)
    m.addConstr(lhs <= 2)

total = quicksum(x[(i,j)] for i in i_set for j in j_set)
m.addConstr(total <= 10)



m.optimize()
m.printAttr('x')
m.printAttr('objVal')

