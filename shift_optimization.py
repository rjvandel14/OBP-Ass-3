# Part e: Integer Linear Optimization
import pulp 
import numpy as np

# required agent-hours per day from (d)
required_hours = [105.25, 539.00, 454.25, 389.75, 456.75, 371.75, 151.25]
required_hours = [int(np.ceil(h)) for h in required_hours]  # rounding up

# shift patterns from question
shift_patterns = [
    # 3x8h shifts
    [8, 8, 8, 0, 0, 0, 0],   # monday, tuesday, wednesday
    [0, 8, 8, 8, 0, 0, 0],   # tuesday, wednesday, thursday
    [0, 0, 8, 8, 8, 0, 0],   # wednesday, thursday, friday
    [0, 0, 0, 8, 8, 8, 0],   # thursday, friday, saturday
    [0, 0, 0, 0, 8, 8, 8],   # friday, saturday, sunday
    # 4x6h shifts (Mon-Thu to Thu-Sun)
    [6, 6, 6, 6, 0, 0, 0],   # monday to thursday
    [0, 6, 6, 6, 6, 0, 0],   # tuesday to friday
    [0, 0, 6, 6, 6, 6, 0],   # wednesday to saturday
    [0, 0, 0, 6, 6, 6, 6],   # thursday to sunday
]

num_shifts = len(shift_patterns)
days = range(7)

# the ILP model 
model = pulp.LpProblem("Scheduling_Shifts", pulp.LpMinimize)

# decision variable: number of times each shift is assigned
x = [pulp.LpVariable(f"x_{i}", lowBound=0, cat='Integer') for i in range(num_shifts)]

# objective: minimize total scheduled hours
total_scheduled = pulp.lpSum(x[i] * sum(shift_patterns[i]) for i in range(num_shifts))
model += total_scheduled

# constraints: required agent-hours per day must be covered
for d in days:
    model += pulp.lpSum(x[i] * shift_patterns[i][d] for i in range(num_shifts)) >= required_hours[d]

# solve model
solver = pulp.PULP_CBC_CMD(msg=False)
model.solve(solver)


print("Optimal Shift Assignments:")
total_hours = 0
for i in range(num_shifts):
    shift_count = int(pulp.value(x[i]))
    shift_hours = sum(shift_patterns[i])
    total_hours += shift_count * shift_hours
    print(f"Shift {i+1}: {shift_count} times ({shift_hours}h/shift) → {shift_count * shift_hours}h")

required_total = sum(required_hours)
inefficiency = (total_hours - required_total) / required_total

print(f"\nTotal required hours: {required_total}")
print(f"Total scheduled hours: {total_hours}")
print(f"Shift inefficiency: {inefficiency:.2%}")
