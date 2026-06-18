# MAT/BIS 107 Lab 9 Lesson Plan: Branching Processes

## Instructor overview

This lesson converts the original MATLAB-based Lab 9 into a Python/Jupyter lesson. Students model disease spread using branching processes. They first simulate outbreaks using a Poisson offspring distribution, then compare this with a Negative Binomial offspring distribution fitted to secondary infection data from the 2003 SARS-CoV-1 outbreak in Singapore.

**Estimated time:** one 90–120 minute lab period.

**Files needed:** none required for this converted version. The secondary infection counts are included directly in the notebook as a small Python list. A CSV version, `secondary_infections_counts.csv`, is also provided as a convenience.

**Python packages used:** `numpy`, `math`, and `matplotlib.pyplot`.

## Learning goals

By the end of this lesson, students should be able to:

1. Define a branching process and an offspring distribution.
2. Simulate outbreaks across transmission generations.
3. Explain the difference between outbreak size at one generation and total cumulative infections.
4. Compare Poisson and Negative Binomial offspring models.
5. Explain why overdispersion and superspreading can increase both extinction probability and outbreak variability.

## Background for instructors

A **branching process** begins with an initial infected individual. Each infected individual produces a random number of secondary infections. Those secondary infections then produce their own secondary infections, and the process repeats over generations.

The **offspring distribution** is the probability model for how many people one infected person infects. A Poisson distribution is simple because its mean and variance are equal. However, many real disease datasets are **overdispersed**: most infected people cause zero or few secondary infections, while a small number of individuals cause many. A Negative Binomial distribution is often used because it can represent this extra variation.

A key teaching point is that two models can have similar average transmissibility but different outbreak behavior. The Negative Binomial model can produce many extinct outbreaks and a few very large outbreaks. This is useful for discussing superspreading.

## Important caveats

- Branching process “time” means transmission generation, not calendar days.
- These models ignore susceptible depletion, behavior change, interventions, and contact networks.
- Simulations are random and can differ noticeably with only 100 runs.
- The Negative Binomial fit in this pure-Python lesson uses a simple likelihood search. MATLAB's `nbinfit` may give very slightly different values depending on optimization details.
- This is a historical modeling exercise about SARS-CoV-1, not a real-time forecasting tool.

## Student notebook content

```python
import numpy as np
import math
import matplotlib.pyplot as plt
```

### Helper functions

```python
def sim_branching_poisson(lam, n_max, rng=None):
    """Simulate a branching process with Poisson offspring."""
    if rng is None:
        rng = np.random.default_rng()
    X = np.zeros(n_max + 1, dtype=int)
    X[0] = 1
    for n in range(n_max):
        if X[n] == 0:
            X[n + 1] = 0
        else:
            X[n + 1] = rng.poisson(lam, size=X[n]).sum()
    return X


def sim_branching_nbin(r, p, n_max, rng=None):
    """Simulate a branching process with Negative Binomial offspring.

    This uses the same parameterization as MATLAB's nbinrnd(r, p):
    number of failures before r successes, with success probability p.
    Mean = r * (1 - p) / p.
    """
    if rng is None:
        rng = np.random.default_rng()
    X = np.zeros(n_max + 1, dtype=int)
    X[0] = 1
    for n in range(n_max):
        if X[n] == 0:
            X[n + 1] = 0
        else:
            X[n + 1] = rng.negative_binomial(r, p, size=X[n]).sum()
    return X


def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def nbin_pmf(k, r, p):
    """Negative Binomial PMF for k failures before r successes."""
    log_prob = (math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
                + r * math.log(p) + k * math.log(1 - p))
    return math.exp(log_prob)


def fit_negative_binomial_mle(counts):
    """Simple pure-Python likelihood search for NB r and p.

    For a fixed r, the MLE for p is r / (r + sample_mean).
    We search over log(r) using a ternary-search style optimization.
    """
    counts = np.asarray(counts, dtype=float)
    sample_mean = counts.mean()

    def log_likelihood_for_log_r(log_r):
        r = math.exp(log_r)
        p = r / (r + sample_mean)
        total = 0.0
        for k in counts:
            total += (math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
                      + r * math.log(p) + k * math.log(1 - p))
        return total

    lo, hi = -10, 5
    for _ in range(100):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if log_likelihood_for_log_r(m1) < log_likelihood_for_log_r(m2):
            lo = m1
        else:
            hi = m2

    r = math.exp((lo + hi) / 2)
    p = r / (r + sample_mean)
    return r, p
```

### Case study data

```python
secondary_infections = np.array([
    33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 8, 10, 0, 0, 0, 3
])

print("Number of observations:", len(secondary_infections))
print("Sample mean:", secondary_infections.mean())
```

### Questions 1–6: Poisson branching process

**Question 1.** Simulate a branching process with a Poisson(1.88) offspring distribution through the 10th generation. Plot outbreak size over time.

```python
rng = np.random.default_rng(107)
lam = 1.88
single_outbreak = sim_branching_poisson(lam, 10, rng=rng)
print(single_outbreak)

plt.figure()
plt.plot(np.arange(0, 11), single_outbreak, marker="o")
plt.xlabel("Generation")
plt.ylabel("Outbreak size")
plt.title("One Poisson branching-process simulation")
plt.show()
```

**Question 2.** Produce 100 independent simulations and plot all trajectories in one figure.

```python
rng = np.random.default_rng(107)
n_sims = 100
n_max = 10
poisson_sims = np.zeros((n_sims, n_max + 1), dtype=int)

for i in range(n_sims):
    poisson_sims[i] = sim_branching_poisson(lam, n_max, rng=rng)

plt.figure()
for i in range(n_sims):
    plt.plot(np.arange(0, n_max + 1), poisson_sims[i], alpha=0.35)
plt.xlabel("Generation")
plt.ylabel("Outbreak size")
plt.title("100 Poisson branching-process simulations")
plt.show()
```

**Question 3.** What is the mean outbreak size at the 10th generation?

```python
poisson_mean_generation_10 = poisson_sims[:, 10].mean()
poisson_mean_generation_10
```

**Question 4.** In what proportion of simulations did the outbreak stop by the 10th generation?

```python
poisson_extinct_by_10 = np.mean(poisson_sims[:, 10] == 0)
poisson_extinct_by_10
```

**Question 5.** What is the expected total number of infections by the 10th generation if λ = 1.88?

```python
expected_total = sum(lam ** n for n in range(0, 11))
expected_total
```

**Question 6.** Use `cumsum` to compute total infections by the 10th generation in your simulations. How does the distribution compare with Question 5?

```python
poisson_cumulative = np.cumsum(poisson_sims, axis=1)
poisson_total_by_10 = poisson_cumulative[:, 10]

print("Mean total by generation 10:", poisson_total_by_10.mean())
print("Expected total from formula:", expected_total)

plt.figure()
plt.hist(poisson_total_by_10, bins=20)
plt.axvline(expected_total, linestyle="--", label="Analytical expectation")
plt.xlabel("Total infections through generation 10")
plt.ylabel("Number of simulations")
plt.legend()
plt.show()
```

### Questions 7–8: Negative Binomial model

**Question 7a.** Fit a Negative Binomial distribution to the secondary infection data. What values of `r` and `p` do you get?

```python
r_hat, p_hat = fit_negative_binomial_mle(secondary_infections)
print("r_hat:", r_hat)
print("p_hat:", p_hat)
print("NB mean:", r_hat * (1 - p_hat) / p_hat)
```

**Question 7b.** Plot a normalized histogram of the data and overlay the fitted Negative Binomial probability distribution.

```python
x_values = np.arange(0, 36)
nb_probs = np.array([nbin_pmf(int(k), r_hat, p_hat) for k in x_values])

plt.figure()
plt.hist(secondary_infections, bins=np.arange(-0.5, 36.5, 1), density=True, label="Data")
plt.plot(x_values, nb_probs, marker="o", label="Negative Binomial")
plt.xlabel("Secondary infections")
plt.ylabel("Probability")
plt.title("Negative Binomial fit")
plt.legend()
plt.show()
```

**Question 7c.** Overlay the Poisson(1.88) distribution. Which model better captures the observed offspring distribution?

```python
poisson_probs = np.array([poisson_pmf(int(k), lam) for k in x_values])

plt.figure()
plt.hist(secondary_infections, bins=np.arange(-0.5, 36.5, 1), density=True, label="Data")
plt.plot(x_values, nb_probs, marker="o", label="Negative Binomial")
plt.plot(x_values, poisson_probs, marker="s", label="Poisson")
plt.xlabel("Secondary infections")
plt.ylabel("Probability")
plt.title("Poisson vs. Negative Binomial offspring models")
plt.legend()
plt.show()
```

**Question 8.** Repeat the 100 simulations using the fitted Negative Binomial distribution.

**8a.** What is the mean outbreak size at the 10th generation?

**8b.** In what proportion of simulations did the outbreak stop by the 10th generation?

**8c.** Use `cumsum` to compute total infections by the 10th generation.

**8d.** How do the total infections and extinction proportion compare with the Poisson model?

```python
rng = np.random.default_rng(107)
nb_sims = np.zeros((n_sims, n_max + 1), dtype=int)

for i in range(n_sims):
    nb_sims[i] = sim_branching_nbin(r_hat, p_hat, n_max, rng=rng)

nb_mean_generation_10 = nb_sims[:, 10].mean()
nb_extinct_by_10 = np.mean(nb_sims[:, 10] == 0)
nb_cumulative = np.cumsum(nb_sims, axis=1)
nb_total_by_10 = nb_cumulative[:, 10]

print("NB mean outbreak size at generation 10:", nb_mean_generation_10)
print("NB extinction proportion by generation 10:", nb_extinct_by_10)
print("NB mean total infections by generation 10:", nb_total_by_10.mean())
print("Poisson mean total infections by generation 10:", poisson_total_by_10.mean())
print("Poisson extinction proportion by generation 10:", poisson_extinct_by_10)

plt.figure()
for i in range(n_sims):
    plt.plot(np.arange(0, n_max + 1), nb_sims[i], alpha=0.35)
plt.xlabel("Generation")
plt.ylabel("Outbreak size")
plt.title("100 Negative Binomial branching-process simulations")
plt.show()
```

## Instructor Answer Key Summary

- Q1: A single Poisson branching-process simulation should produce a vector of 11 values, from generation 0 through 10. The trajectory may grow or go extinct depending on the random seed.
- Q2: The 100 trajectories should show high variability. Some outbreaks go extinct; others grow quickly.
- Q3: The mean outbreak size at generation 10 varies by simulation set. It is often roughly in the `500–650` range with 100 simulations, but can vary substantially.
- Q4: The extinction proportion by generation 10 is often around `0.20–0.25` for 100 simulations.
- Q5: The expected total number of infections is `1 + 1.88 + ... + 1.88^10 ≈ 1177.16`.
- Q6: The simulated mean cumulative total should be near the analytical expectation, but the distribution is wide.
- Q7a: The pure-Python MLE search gives approximately `r ≈ 0.12` and `p ≈ 0.061`. MATLAB `nbinfit` in the original solution reported about `r = 0.1126`, `p = 0.0612`.
- Q7b–Q7c: The Negative Binomial should fit the data better than the Poisson because it captures many zeroes plus rare large secondary infection counts.
- Q8a: The mean generation-10 outbreak size is often comparable to the Poisson model but has much larger variance.
- Q8b: The extinction proportion is much higher under the Negative Binomial model, often around `0.90` with 100 simulations.
- Q8c: Mean total infections may be similar to the Poisson model but much more variable.
- Q8d: The Negative Binomial model produces more extinct outbreaks and occasional very large outbreaks, reflecting overdispersion/superspreading.
