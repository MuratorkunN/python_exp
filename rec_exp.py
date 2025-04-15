import numpy as np


mylist = [3,4,5,6]

print(enumerate((mylist)))

for i in enumerate(mylist):
    print(i)
    
def assign(workers, jobs, costs):
    
    if (len(workers) == 1):
        pair = (workers[0], jobs[0])
        obj = costs[pair]
        sol = [pair]
        
        return (obj, sol)
    
    else:
        chosen_workers = workers[0]
        remaining_workers = workers[1:]
        
        #candidate_assignments