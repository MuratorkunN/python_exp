from math import inf

my_items = ("book", "clock", "computer", "painting", "radio", "vase")
my_capacity = 20
my_vals = dict(zip(my_items, (9, 175, 200, 99, 20, 50)))
my_weights = dict(zip(my_items, (1, 10, 20, 9, 4, 2)))


print(my_vals["book"])
#print(my_vals[2])

def knapsack01_greedy(items, capacity, values, weights):
    if capacity < 0:
        obj_val, solution = -inf, []
    
    else:
        obj_val, solution, slack = 0, [], capacity
        v2w_ratio = {item: values[item] / weights[item] for item in items}
        ordered = sorted(items, key = lambda item:v2w_ratio[item], reverse=True)
        
        for item in ordered:
            if weights[item] <= slack:
                solution.append(item)
                slack -= weights[item]
                obj_val += values[item]
        
    
    return obj_val, solution

my_result = knapsack01_greedy(my_items, my_capacity, my_vals, my_weights)
print(my_result)