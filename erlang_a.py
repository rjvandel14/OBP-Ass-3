# Part b: Erlang A model implementation
# Check with Erlang X calculator

import numpy as np

def erlang_a_stationary_probs(lambda_, mu, gamma, s, max_states=500):
    probs = np.zeros(max_states)
    probs[0] = 1.0  # initial value

    # Compute unnormalized probabilities
    for n in range(1, max_states):
        if n <= s:
            # There is no queue
            probs[n] = probs[n - 1] * lambda_ / (n * mu)
        else:
            # There is a queue
            probs[n] = probs[n - 1] * lambda_ / (s * mu + (n - s) * gamma)

    # Normalize
    probs /= np.sum(probs)

    # Delay probability = sum of probabilities of all states with ≥ s customers
    delay_prob = np.sum(probs[s:])
    return delay_prob


def min_agents_erlang_a(lambda_, mu, gamma, max_delay_prob, s_max=300):
    for s in range(1, s_max):
        delay_prob = erlang_a_stationary_probs(lambda_, mu, gamma, s)
        if delay_prob <= max_delay_prob:
            return s
    raise ValueError("No s found that satisfies the condition up to s_max.")

lambda_ = 3    # calls per minute
mu = 1 / 5       # 1 call per 5 minutes
gamma = 1 / 10   # patience = 10 minutes
max_delay_prob = 0.4  # 60% customers may wait

s_needed = min_agents_erlang_a(lambda_, mu, gamma, max_delay_prob)
print(f"Minimum agents needed: {s_needed}")
