# Lab 5: Geometric Random Walks and Bay Checkerspot Butterflies

> **Python version of the original MAT/BIS 107 MATLAB lab.**  
> This lesson uses basic Python packages only: `math`, `random`/`numpy`, `pandas` when reading CSV files, and `matplotlib.pyplot` for plots. The original MATLAB labs asked students to submit a single script; this version is designed for a Jupyter notebook, where students run each code cell and write short explanations in markdown cells.

**Instructor note.** The code cells are intentionally readable rather than highly optimized. For class use, encourage students to run each cell, inspect intermediate variables, and answer the written questions in their own words.


## Lesson goals
Students will model multiplicative population growth, transform products into sums using logarithms, simulate geometric random walks, compare simulations to Normal approximations, and use rainfall data to evaluate population-risk hypotheses.

## Expanded background
A population that changes by a growth multiplier each generation follows:

`N(t) = R(t) × N(t-1)`

After many generations, this becomes a product of many random multipliers. Products are hard to analyze directly, but logarithms turn products into sums:

`log N(t) = log N(0) + sum(log R(i))`

This makes the expected trend easier to describe. If the average value of `log R` is positive, the population tends to grow in the log domain. If it is negative, the population tends to shrink.

The second part uses Bay checkerspot butterfly rainfall data. The lab's model converts annual rainfall into a population growth coefficient. Students compare pre-1971 and post-1971 rainfall periods.

### Caveats
This model assumes growth rates are independent samples from past rainfall-derived values and ignores density dependence, habitat loss, dispersal, predation, disease, and conservation interventions. It is useful for exploring risk, but it is not a complete ecological forecast.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(107)

def sim_geometric_population(N0, R, probs, T, rng=None):
    # Simulate N(t)=R(t)N(t-1) for T time points.
    if rng is None:
        rng = np.random.default_rng()
    R = np.asarray(R, dtype=float)
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    N = np.empty(T, dtype=float)
    N[0] = N0
    for t in range(1, T):
        R_t = rng.choice(R, p=probs)
        N[t] = R_t * N[t-1]
    return N

def simulate_many_geometric(N0, R, probs, T, n_sims, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    out = np.empty((n_sims, T), dtype=float)
    for i in range(n_sims):
        out[i] = sim_geometric_population(N0, R, probs, T, rng)
    return out

def normal_pdf(x, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)
```

## Student Questions and Code
The following sections are written as student-facing prompts. Students should run the relevant code cells, inspect the output, and write their own answers before looking at the instructor key.

## Part 1: Theory and simulations
### Question 1
Let `R(t)` be either `1.2` or `0.8`, with equal probability.

1a. What is `E[log R(t)]`?  
1b. What is `Var(log R(t))`?  
1c. Simulate 1000 populations over 100 generations starting with `N(0)=100`, and plot `log N(t)`.  
1d. Add the expected growth line `E[log N(t)] = log N(0) + t E[log R(t)]`.  
1e. What proportion of simulations have `log N(t) < 0` at `t=100`?

```python
T = 100
n_sims = 1000
N0 = 100
R = np.array([1.2, 0.8])
probs = np.array([0.5, 0.5])

EV_logR = np.sum(probs * np.log(R))
EV_logR2 = np.sum(probs * (np.log(R) ** 2))
Var_logR = EV_logR2 - EV_logR**2
print("E[log R]:", EV_logR)
print("Var(log R):", Var_logR)

Nt1 = simulate_many_geometric(N0, R, probs, T, n_sims, rng)
t = np.arange(T)
plt.plot(t, np.log(Nt1.T), alpha=0.1)
plt.plot(t, np.log(N0) + t * EV_logR, linewidth=3, label='Expected log growth')
plt.xlabel('Time')
plt.ylabel('log N(t)')
plt.title('Geometric random walk: R = 1.2 or 0.8')
plt.legend()
plt.show()

prop_under = np.mean(np.log(Nt1[:, -1]) < 0)
print("Proportion with log N(100) < 0:", prop_under)
```

### Question 2
Repeat Question 1 using `R(t)` equal to `5` or `0.25`, with equal probability.

### Question 3
Which model produces a higher proportion of populations greater than `log N(t)=0` at `t=100`?

```python
R = np.array([5, 0.25])
probs = np.array([0.5, 0.5])
EV_logR = np.sum(probs * np.log(R))
EV_logR2 = np.sum(probs * (np.log(R) ** 2))
Var_logR = EV_logR2 - EV_logR**2
print("E[log R]:", EV_logR)
print("Var(log R):", Var_logR)

Nt2 = simulate_many_geometric(N0, R, probs, T, n_sims, rng)
plt.plot(t, np.log(Nt2.T), alpha=0.1)
plt.plot(t, np.log(N0) + t * EV_logR, linewidth=3, label='Expected log growth')
plt.xlabel('Time')
plt.ylabel('log N(t)')
plt.title('Geometric random walk: R = 5 or 0.25')
plt.legend()
plt.show()

prop_under2 = np.mean(np.log(Nt2[:, -1]) < 0)
prop_above2 = np.mean(np.log(Nt2[:, -1]) > 0)
print("Proportion with log N(100) < 0:", prop_under2)
print("Proportion with log N(100) > 0:", prop_above2)
```

### Question 4
Using the simulations from Question 2, extract log-population sizes at `t=100`.

4a. Plot a histogram.  
4b. Compute theoretical expected value and variance of `log N(t)` at `t=100`.  
4c. Overlay a Normal density using that mean and variance. How well does it fit?

```python
logN_T = np.log(Nt2[:, -1])
mu = np.log(N0) + (T-1) * EV_logR
var = (T-1) * Var_logR
sigma = np.sqrt(var)
print("theoretical mean:", mu)
print("theoretical variance:", var)

plt.hist(logN_T, bins=30, density=True, alpha=0.7, label='Simulations')
xs = np.linspace(logN_T.min(), logN_T.max(), 300)
plt.plot(xs, normal_pdf(xs, mu, sigma), linewidth=3, label='Normal approximation')
plt.xlabel('log N(100)')
plt.ylabel('Density')
plt.title('Distribution of final log-population sizes')
plt.legend()
plt.show()
```

## Part 2: Bay checkerspot butterflies
The file `checkerspots.csv` contains rainfall data for San Jose from 1932-1999. Place this CSV file in the same folder as the notebook.

The lab model converts rainfall `x` in millimeters to a growth coefficient:

`R = exp(A - B/x)`

where `A = 0.932171` and `B = 3262.42`. Low rainfall can produce very small growth coefficients.

### Question 5
Compute average annual rainfall before 1971 and from 1971 onward. Which period is greater?

### Question 6
Convert rainfall values to growth coefficients. Which period has greater average `log R`?

```python
rain = pd.read_csv('checkerspots.csv')
print(rain.head())

x_pre1971 = rain.loc[rain['rain.yr'] < 1971, 'rain'].to_numpy()
x_post1971 = rain.loc[rain['rain.yr'] >= 1971, 'rain'].to_numpy()

print("Average rainfall pre-1971:", x_pre1971.mean())
print("Average rainfall post-1971:", x_post1971.mean())

A = 0.932171
B = 3262.42
R_pre = np.exp(A - B / x_pre1971)
R_post = np.exp(A - B / x_post1971)

print("Average log R pre-1971:", np.mean(np.log(R_pre)))
print("Average log R post-1971:", np.mean(np.log(R_post)))

plt.plot(rain['rain.yr'], rain['rain'], marker='o')
plt.axvline(1971, linestyle='--')
plt.xlabel('Year')
plt.ylabel('Rainfall')
plt.title('Annual rainfall data')
plt.show()
```

### Questions 7-8
Simulate 1000 populations for 100 generations using growth coefficients sampled uniformly from the pre-1971 period and then from the post-1971 period. Use `N(0)=100`.

For each period:

- Plot trajectories of `log N(t)` with the expected trend line.
- Compute the proportion below `log N(t)=0` at `t=100`.

### Question 9
Do the rainfall data and model support the idea that large climate fluctuations can damage the Bay checkerspot population?

```python
def run_rainfall_sim(R_values, label):
    probs = np.ones(len(R_values)) / len(R_values)
    Nt = simulate_many_geometric(N0=100, R=R_values, probs=probs, T=100, n_sims=1000, rng=rng)
    EV_logR = np.mean(np.log(R_values))
    t = np.arange(100)
    plt.plot(t, np.log(Nt.T), alpha=0.08)
    plt.plot(t, np.log(100) + t * EV_logR, linewidth=3, label='Expected log growth')
    plt.xlabel('Generation')
    plt.ylabel('log N(t)')
    plt.title(label)
    plt.legend()
    plt.show()
    prop_under = np.mean(np.log(Nt[:, -1]) < 0)
    print(label, "proportion below log N(100)=0:", prop_under)
    return Nt, prop_under

Nt_pre, prop_pre = run_rainfall_sim(R_pre, 'Pre-1971 rainfall-derived growth')
Nt_post, prop_post = run_rainfall_sim(R_post, 'Post-1971 rainfall-derived growth')
```

## Instructor Answer Key Summary

**Question 1a.** For `R={1.2, 0.8}` with equal probability, `E[log R] ≈ -0.0204`.

**Question 1b.** `Var(log R) ≈ 0.0411`.

**Question 1c-d.** The simulated log-population trajectories should fluctuate around the expected line `log(100) + t E[log R]`, which slopes downward because `E[log R]` is negative.

**Question 1e.** Simulation results vary. With the provided random seed, about `0.105` of simulations have `log N(100) < 0`.

**Question 2.** For `R={5, 0.25}` with equal probability, `E[log R] ≈ 0.1116` and `Var(log R) ≈ 2.2436`. The mean trend is positive, but the variance is much larger.

**Question 3.** The second model usually produces a higher proportion of populations above `log N(100)=0`, but it also produces more extreme low and high outcomes because its variance is much larger.

**Question 4.** For the second model at `t=100` using the notebook indexing, the theoretical mean of final `log N` is about `15.65` and the theoretical variance is about `222.12`. The Normal approximation should capture the general bell shape, but simulation discreteness and extreme tails may be visible.

**Question 5.** In the provided rainfall data, average rainfall is about `67.30` before 1971 and `73.85` from 1971 onward, so the post-1971 period has higher average rainfall.

**Question 6.** The average `log R` is about `-51.27` before 1971 and `-50.98` from 1971 onward. The post-1971 period is slightly less negative by this simple average, but both are strongly negative under this particular model and unit scale.

**Questions 7-8.** With the current code and parameters, both pre- and post-1971 simulations generally fall below `log N(100)=0` by generation 100. This part is sensitive to units and numerical underflow; using log-domain simulation is preferable for very small growth coefficients.

**Question 9.** The exercise supports the qualitative idea that environmental variation can strongly affect population persistence, but the conclusion should be cautious. The model is simplified and does not include habitat loss, density dependence, dispersal, or conservation management.
