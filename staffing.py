# 	Parts c + d: Required agents & hours
import pandas as pd
import numpy as np
from erlang_a import min_agents_erlang_a 
from forecasting import forecast_and_evaluate

# Parameters
mu = 1 / 5        # Service rate (calls per minute)
gamma = 1 / 10    # Abandonment rate (per minute)
max_delay_prob = 0.4
opening_hours = 14
minutes_per_day = opening_hours * 60

# Load and clean data
df = pd.read_csv("actuals.csv", index_col=0)
df.rename(columns={"x": "volume"}, inplace=True)

# Get forecasted volumes for week 260 (days 1813–1819)
forecast, days, wape = forecast_and_evaluate(df, week_num=260, degree=3)

total_agent_hours = 0
print("\n(c) Required agents and total agent hours")

for day, volume in zip(days, forecast):
    lambda_ = volume / minutes_per_day  # calls per minute
    agents = min_agents_erlang_a(lambda_, mu, gamma, max_delay_prob)
    daily_hours = agents * opening_hours
    total_agent_hours += daily_hours
    print(f"Day {day}: volume={volume:.0f}, λ={lambda_:.2f} → {agents} agents → {daily_hours:.1f} agent-hours")

print(f"\n Total agent hours for week 260: {total_agent_hours:.1f}")

print("\n(d) Varying arrival rate over quarters")

total_agent_hours_d = 0

for day, volume in zip(days, forecast):
    quarters = 56  # 14 hours × 4
    weights = np.arange(1, quarters + 1)  # linear ramp: 1 → 56
    weight_sum = np.sum(weights)

    # Arrival per quarter: total volume distributed by weight
    arrivals_per_quarter = volume * (weights / weight_sum)
    lambda_per_quarter = arrivals_per_quarter / 15  # 15 minutes per quarter → calls per minute

    agent_hours = 0
    for lam in lambda_per_quarter:
        agents = min_agents_erlang_a(lam, mu, gamma, max_delay_prob)
        agent_hours += agents * 0.25  # 15 minutes = 0.25 hours

    total_agent_hours_d += agent_hours
    print(f"Day {day}: total volume={volume:.0f} → {agent_hours:.2f} agent-hours")

print(f"\n Total agent-hours for week 260 (linear ramp-up): {total_agent_hours_d:.2f}")
