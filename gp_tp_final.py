import gurobipy as gp

m = gp.Model("TransportationProblem")

num_sources, num_destinations = 3, 4
source_set, destination_set = range(num_sources), range(num_destinations)

cost_table = {(0,0):10, (0,1):0, (0,2):20, (0,3):11,
(1,0):12, (1,1):7, (1,2):9, (1,3):20,
(2,0):0, (2,1):14, (2,2):16, (2,3):18}
supplies = {0:1400, 1:2300, 2:800}
demands = {0:500, 1:1500, 2:1500, 3:1000}

#print(cost_table[0][1])

x = m.addVars(source_set, destination_set, vtype= gp.GRB.CONTINUOUS, name = "amount", lb = 0)

print(type(x))

obj_expr=(gp.quicksum(cost_table[(src, dest)] * x[(src, dest)]
                      for src in source_set for dest in destination_set))

#obj_expr = gp.quicksum(cost_table[(i,j)] * x[(i,j)] for (i,j) in x.keys())


m.setObjective(obj_expr, sense = gp.GRB.MINIMIZE)

for i in source_set:
    lhs = gp.quicksum(x[(i, j)] for j in destination_set)
    m.addConstr(lhs <= supplies[i])
    
for j in destination_set:
    lhs = gp.quicksum(x[(i, j)] for i in source_set)
    m.addConstr(lhs >= demands[j])
    
m.optimize()
m.printAttr("objval")
m.printAttr("x")
