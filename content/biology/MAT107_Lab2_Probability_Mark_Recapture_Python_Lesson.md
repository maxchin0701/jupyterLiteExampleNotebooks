# Lab 2: Probabilities, Bayes' Formula, and Estimating Population Size

> **Python version of the original MAT/BIS 107 MATLAB lab.**  
> This lesson uses basic Python packages only: `math`, `random`/`numpy`, `pandas` when reading CSV files, and `matplotlib.pyplot` for plots. The original MATLAB labs asked students to submit a single script; this version is designed for a Jupyter notebook, where students run each code cell and write short explanations in markdown cells.

**Instructor note.** The code cells are intentionally readable rather than highly optimized. For class use, encourage students to run each cell, inspect intermediate variables, and answer the written questions in their own words.


## Lesson goals
Students will compute conditional probabilities, use Bayes' formula, simulate random systems, and compare two mark-recapture sampling models: sampling without replacement and sampling with replacement.

## Background
Conditional probability asks: "What is the probability of event A, given that event B happened?" Bayes' formula reverses conditional probabilities:

`P(A | B) = P(B | A) P(A) / P(B)`

The second half of the lab uses **mark-recapture**, a method for estimating animal population size. Researchers capture and tag animals, release them, then capture again and count how many tagged animals are found. If the tagged animals mix well into the population, the tagged proportion in the second sample can estimate the tagged proportion in the whole population.

### Caveats
Mark-recapture depends on strong assumptions: tagged animals must mix back into the population, tags must not affect survival or capture probability, the population should not change too much between captures, and every animal should be equally likely to be sampled. Violations can bias estimates.

```python
import math
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(107)

def binomial_pmf(k, n, p):
    if k < 0 or k > n:
        return 0.0
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

def hypergeometric_pmf(N, B, n, k):
    # Probability of k tagged animals in n samples without replacement.
    # N = population size, B = tagged animals, n = second sample size.
    if k < 0 or k > B or n-k > N-B:
        return 0.0
    return math.comb(B, k) * math.comb(N - B, n - k) / math.comb(N, n)
```

## Student Questions and Code
The following sections are written as student-facing prompts. Students should run the relevant code cells, inspect the output, and write their own answers before looking at the instructor key.

## Part 1: Conditional probability and Bayes' formula
Jar 1 contains 3 green balls and 2 red balls. Jar 2 contains 2 green balls and 2 red balls. One ball is moved from Jar 1 to Jar 2, but we do not know its color. Then one ball is drawn from Jar 2.

Use `1` for green and `0` for red.

### Question 1
What is the probability that a red ball was transferred from Jar 1 to Jar 2 if the ball drawn from Jar 2 is red?

```python
# Work through Bayes' formula.
# P(red transferred | red drawn) = P(red drawn | red transferred) * P(red transferred) / P(red drawn)

p_red_transferred = 2/5
p_green_transferred = 3/5
p_red_drawn_given_red_transferred = 3/5
p_red_drawn_given_green_transferred = 2/5

p_red_drawn = (p_red_drawn_given_red_transferred * p_red_transferred +
               p_red_drawn_given_green_transferred * p_green_transferred)

p_red_transferred_given_red_drawn = p_red_drawn_given_red_transferred * p_red_transferred / p_red_drawn
print(p_red_transferred_given_red_drawn)
```

### Questions 2-3: Simulation check
Run the simulation. Does the simulated proportion of green balls drawn from Jar 2 agree with the theoretical probability `13/25`? Then estimate the probability that a green ball was transferred given that a green ball was drawn.

```python
N_simulations = 10_000
transferred_balls = np.empty(N_simulations, dtype=int)
sampled_balls = np.empty(N_simulations, dtype=int)

for i in range(N_simulations):
    jar1 = np.array([1, 1, 1, 0, 0])
    jar2 = np.array([1, 1, 0, 0])
    transferred = rng.choice(jar1)
    jar2 = np.append(jar2, transferred)
    sampled = rng.choice(jar2)
    transferred_balls[i] = transferred
    sampled_balls[i] = sampled

prop_green_drawn = sampled_balls.mean()
print("Proportion green drawn:", prop_green_drawn)
print("Theoretical P(green drawn):", 13/25)

# Among cases where a green ball was drawn, what proportion had a green ball transferred?
prob_green_transfer_given_green_draw = transferred_balls[sampled_balls == 1].mean()
print("P(green transferred | green drawn), simulated:", prob_green_transfer_given_green_draw)
print("Theoretical:", 9/13)
```

## Part 2: Mark-recapture without replacement
If `B` animals were tagged, `n` animals were sampled later, and `k` tagged animals were found, a simple estimate is:

`estimated population size = B / (k/n)`

### Question 4
If 10 animals are tagged and 10 animals are sampled later, with 2 tagged animals found, what is the estimated population size?

### Question 5
If zero tagged animals are found in the second capture, what does that suggest? Why is this a difficult case?

```python
B = 10
n = 10
k = 2
N_estimated = B / (k / n)
print(N_estimated)

# If k = 0, the simple formula divides by zero. This is a sign that the sample is not informative enough.
```

### Questions 6-8: Hypergeometric probabilities
In a lake of `N=100` fish, suppose `B=20` are tagged and `n=15` are sampled without replacement.

The probability of observing exactly `k` tagged fish is:

`C(B,k) C(N-B,n-k) / C(N,n)`

Question 6: What is the probability of collecting exactly 4 tagged fish?  
Question 7: What is the probability of collecting exactly 0 tagged fish?  
Question 8: How does the probability of sampling 0 tagged animals change as population size increases?

```python
N = 100
B = 20
n = 15
print("P(k=4):", hypergeometric_pmf(N, B, n, 4))
print("P(k=0):", hypergeometric_pmf(N, B, n, 0))

# Distribution over k
ks = np.arange(n + 1)
probs = np.array([hypergeometric_pmf(N, B, n, int(k)) for k in ks])
plt.bar(ks, probs)
plt.xlabel('Number of recaptured animals that are tagged (k)')
plt.ylabel('Probability')
plt.title('Hypergeometric mark-recapture distribution')
plt.show()

# Q8: probability of zero tagged animals across population sizes
N_values = np.array([50, 100, 150, 200, 300, 400])
B_values = np.array([5, 10, 15, 20, 25])
B_fixed = 10
p0 = []
for N_val in N_values:
    p0.append(hypergeometric_pmf(int(N_val), B_fixed, B_fixed, 0))

plt.bar(N_values.astype(str), p0)
plt.xlabel('Population size')
plt.ylabel('Probability of sampling 0 tagged animals')
plt.title('Zero recaptures become more likely as population size increases')
plt.show()
```

## Part 3: Sampling with replacement
If sampled animals are released before the next draw, each draw has the same probability `B/N` of being tagged. Then the number of tagged animals follows a binomial distribution.

### Questions 9-10
For each scenario, calculate the probability that the recaptured count is within 1 of the "perfect" value using both methods.

- A: `N=20`, `B=5`, `n=4`, use `k in {0,1,2}`
- B: `N=1000`, `B=50`, `n=40`, use `k in {1,2,3}`
- C: `N=1000`, `B=200`, `n=40`, use `k in {7,8,9}`

Which method seems more reliable under this metric?

```python
scenarios = [
    (20, 5, 4, [0, 1, 2]),
    (1000, 50, 40, [1, 2, 3]),
    (1000, 200, 40, [7, 8, 9]),
]

for N, B, n, good_ks in scenarios:
    p_without = sum(hypergeometric_pmf(N, B, n, k) for k in good_ks)
    p_with = sum(binomial_pmf(k, n, B/N) for k in good_ks)
    print(f"N={N}, B={B}, n={n}, k={good_ks}")
    print("  without replacement:", p_without)
    print("  with replacement:   ", p_with)
```

## Part 4: Simulation comparison
### Question 11
Simulate the mark-recapture experiment many times with and without replacement. Which method gives better population-size predictions in this simulation?

```python
N = 100
B = 20
n = 15
N_sims = 10_000

population = np.zeros(N, dtype=int)
tags = rng.choice(np.arange(N), size=B, replace=False)
population[tags] = 1

est_wr = []
est_wor = []
for _ in range(N_sims):
    # with replacement
    samples = rng.choice(np.arange(N), size=n, replace=True)
    k_tagged = population[samples].sum()
    est_wr.append(np.inf if k_tagged == 0 else B / (k_tagged / n))

    # without replacement
    samples = rng.choice(np.arange(N), size=n, replace=False)
    k_tagged = population[samples].sum()
    est_wor.append(np.inf if k_tagged == 0 else B / (k_tagged / n))

est_wr = np.array(est_wr)
est_wor = np.array(est_wor)
finite_wr = est_wr[np.isfinite(est_wr)]
finite_wor = est_wor[np.isfinite(est_wor)]

bins = np.arange(0, 5*N + 5, 5)
plt.hist(finite_wr, bins=bins, alpha=0.6, label='With replacement')
plt.hist(finite_wor, bins=bins, alpha=0.6, label='Without replacement')
plt.axvline(N, linestyle='--', label='True population size')
plt.xlabel('Estimated population size')
plt.ylabel('Number of simulations')
plt.legend()
plt.show()

print("Mean estimate with replacement:", finite_wr.mean())
print("Mean estimate without replacement:", finite_wor.mean())
print("Zero-tag simulations with replacement:", np.sum(~np.isfinite(est_wr)))
print("Zero-tag simulations without replacement:", np.sum(~np.isfinite(est_wor)))
```

## Instructor Answer Key Summary

**Question 1.** `P(red transferred | red drawn) = 0.5`.

**Question 2.** The theoretical probability of drawing green from Jar 2 is `13/25 = 0.52`. A simulation with 10,000 trials should be close to this value, but will vary slightly.

**Question 3.** `P(green transferred | green drawn) = 9/13 ≈ 0.6923`. A simulation should be close to this value.

**Question 4.** The population-size estimate is `10 / (2/10) = 50` animals.

**Question 5.** Finding zero tagged animals suggests the population may be large relative to the number tagged and sampled, but the simple estimator is undefined because it divides by zero. This is a low-information outcome.

**Question 6.** For `N=100`, `B=20`, `n=15`, `P(k=4) ≈ 0.2004`.

**Question 7.** `P(k=0) ≈ 0.0262`.

**Question 8.** With a fixed number tagged and recaptured, the probability of sampling zero tagged animals increases as the total population size increases.

**Questions 9-10.** For the three scenarios in the notebook, the probability of being within the specified “good” range is approximately:

| Scenario | Without replacement | With replacement |
|---|---:|---:|
| `N=20, B=5, n=4, k={0,1,2}` | `0.9680` | `0.9492` |
| `N=1000, B=50, n=40, k={1,2,3}` | `0.7426` | `0.7333` |
| `N=1000, B=200, n=40, k={7,8,9}` | `0.4541` | `0.4459` |

In these examples, sampling without replacement is slightly more reliable by this metric, although the difference can be small for large populations.

**Question 11.** Simulation results vary. In general, sampling without replacement tends to give a somewhat tighter distribution of estimates because the second sample cannot repeatedly select the same animal. Both methods can produce unstable or infinite estimates when zero tagged animals are recaptured.
