# Lab 6: Genetic Drift and the Wright-Fisher Model

> **Python version of the original MAT/BIS 107 MATLAB lab.**  
> This lesson uses basic Python packages only: `math`, `random`/`numpy`, `pandas` when reading CSV files, and `matplotlib.pyplot` for plots. The original MATLAB labs asked students to submit a single script; this version is designed for a Jupyter notebook, where students run each code cell and write short explanations in markdown cells.

**Instructor note.** The code cells are intentionally readable rather than highly optimized. For class use, encourage students to run each cell, inspect intermediate variables, and answer the written questions in their own words.


## Lesson goals
Students will use binomial probability to model genetic drift, simulate Wright-Fisher allele-count trajectories, build transition matrices, and compare simulations to Peter Buri's fruit fly data.

## Expanded background
A gene can have different versions called alleles. In a diploid population of `N` individuals, there are `2N` gene copies. Genetic drift is random change in allele frequencies caused by random inheritance. Even if two alleles have equal fitness, one allele can eventually become fixed or lost by chance.

The Wright-Fisher model assumes a constant population size, non-overlapping generations, random mating, and no mutation or selection. If there are `i` copies of allele A in a gene pool of size `2N`, the probability that a sampled gene copy in the next generation is A is `i/(2N)`. The next generation count follows a binomial distribution.

### Caveats
The Wright-Fisher model is intentionally simple. Real populations may have selection, mutation, migration, overlapping generations, nonrandom mating, changing population sizes, and population structure.

```python
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(107)

def binomial_pmf(k, n, p):
    if k < 0 or k > n:
        return 0.0
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

def binomial_cdf(k, n, p):
    return sum(binomial_pmf(j, n, p) for j in range(0, k+1))

def sim_wright_fisher(gene_pool_size, initial_count, n_generations, n_simulations, rng=None):
    # Simulate Wright-Fisher allele counts.
    # Returns chains with shape (n_generations+1, n_simulations) and counts table.
    if rng is None:
        rng = np.random.default_rng()
    chains = np.empty((n_generations + 1, n_simulations), dtype=int)
    chains[0, :] = initial_count
    for gen in range(1, n_generations + 1):
        p = chains[gen-1, :] / gene_pool_size
        chains[gen, :] = rng.binomial(gene_pool_size, p)
    counts = np.zeros((n_generations + 1, gene_pool_size + 1), dtype=int)
    for gen in range(n_generations + 1):
        counts[gen] = np.bincount(chains[gen], minlength=gene_pool_size + 1)
    return chains, counts

def transition_matrix_wright_fisher(gene_pool_size):
    P = np.zeros((gene_pool_size + 1, gene_pool_size + 1))
    for i in range(gene_pool_size + 1):
        p = i / gene_pool_size
        P[i, :] = [binomial_pmf(k, gene_pool_size, p) for k in range(gene_pool_size + 1)]
    return P
```

## Student Questions and Code
The following sections are written as student-facing prompts. Students should run the relevant code cells, inspect the output, and write their own answers before looking at the instructor key.

## Part 1: Genetic drift probabilities
Start with 10 diploid individuals, so there are 20 gene copies. Suppose 8 copies are allele A and 12 copies are allele B.

### Question 1
If you sample one allele, what is the probability it is allele A?

### Question 2
If you sample two alleles with replacement, what is the probability both are allele B?

### Question 3
Using the binomial formula, what is the probability that the next generation has exactly 8 copies of allele A?

### Question 4
What is the probability that the next generation does not have exactly 8 copies of allele A?

### Question 5
Compute the probability that the number of copies of A increases. Is A more likely to increase or decrease in the next generation?

```python
gene_pool_size = 20
initial_A = 8
p_A = initial_A / gene_pool_size
p_B = 1 - p_A

print("Q1 P(A):", p_A)
print("Q2 P(B then B):", p_B ** 2)

p_exact_8 = binomial_pmf(8, gene_pool_size, p_A)
p_not_8 = 1 - p_exact_8
p_decrease = binomial_cdf(7, gene_pool_size, p_A)
p_increase = 1 - binomial_cdf(8, gene_pool_size, p_A)

print("Q3 P(k=8):", p_exact_8)
print("Q4 P(k != 8):", p_not_8)
print("P(decrease):", p_decrease)
print("Q5 P(increase):", p_increase)
```

## Part 2: Simulating populations
### Question 6
Run 100 simulations of genetic drift for 15 generations, with gene pool size 20 and initial count 10. Plot allele-count trajectories.

### Question 7
Plot the distribution of allele counts in the first generation.

### Question 8
Plot the distribution of allele counts in the 15th generation. About how many simulations reached 0 or 20 copies?

```python
chains, counts = sim_wright_fisher(gene_pool_size=20, initial_count=10, n_generations=15, n_simulations=100, rng=rng)

gens = np.arange(16)
plt.plot(gens, chains, alpha=0.5)
plt.xlabel('Generation')
plt.ylabel('Copies of allele A')
plt.title('Wright-Fisher simulations')
plt.show()

plt.hist(chains[1, :], bins=np.arange(22)-0.5)
plt.xlabel('Copies of allele A')
plt.ylabel('Number of simulations')
plt.title('Distribution after first generation')
plt.show()

plt.hist(chains[15, :], bins=np.arange(22)-0.5)
plt.xlabel('Copies of allele A')
plt.ylabel('Number of simulations')
plt.title('Distribution after 15 generations')
plt.show()

n_fixed = np.sum((chains[15, :] == 0) | (chains[15, :] == 20))
print("Number fixed at 0 or 20 by generation 15:", n_fixed)
```

## Part 3: Wright-Fisher transition matrices
The Wright-Fisher model can be described as a Markov chain. State `i` means there are `i` copies of allele A. The transition probability from state `i` to state `k` is:

`P(i -> k) = C(2N,k) (i/(2N))^k (1 - i/(2N))^(2N-k)`

### Question 9
Now use a gene pool of size 20 starting with 10 copies of allele A.

9a. Generate the transition matrix.  
9b. What is the initial distribution vector?  
9c. Use the Chapman-Kolmogorov equation to obtain and plot the distribution in the second generation.  
9d. Plot the distribution in the 15th generation.  
9e. Compare to the simulations in Questions 7 and 8.

```python
P = transition_matrix_wright_fisher(20)
print("Transition matrix shape:", P.shape)
print("First row:", P[0])
print("Last row:", P[-1])

pi0 = np.zeros(21)
pi0[10] = 1
print("Initial distribution has all probability at state 10:")
print(pi0)

pi2 = pi0 @ np.linalg.matrix_power(P, 2)
pi15 = pi0 @ np.linalg.matrix_power(P, 15)

states = np.arange(21)
plt.bar(states, pi2)
plt.xlabel('Copies of allele A')
plt.ylabel('Probability')
plt.title('Theoretical distribution after 2 generations')
plt.show()

plt.bar(states, pi15)
plt.xlabel('Copies of allele A')
plt.ylabel('Probability')
plt.title('Theoretical distribution after 15 generations')
plt.show()
```

## Part 4: Comparing to Peter Buri's real data
Peter Buri studied genetic drift in fruit flies. The file `buri_data.csv` contains counts of observed populations by allele-copy number across generations.

### Question 10
Load and plot Buri's data. Are the results consistent with genetic drift theory?

### Question 11
Reproduce Buri's experiment using Wright-Fisher simulations: 107 simulations, gene pool size 32, initial allele count 16, and 19 generations. Compare the simulation to the real data. What is similar? What is different?

```python
buri_data = pd.read_csv('buri_data.csv', header=None)
print(buri_data.head())

plt.imshow(buri_data.iloc[1:, :], aspect='auto', origin='lower')
plt.colorbar(label='Number of observations')
plt.xlabel('Copies of allele')
plt.ylabel('Generation')
plt.title("Peter Buri's fruit fly data")
plt.show()

chains_buri, counts_buri = sim_wright_fisher(gene_pool_size=32, initial_count=16, n_generations=19, n_simulations=107, rng=rng)

plt.imshow(counts_buri[1:, :], aspect='auto', origin='lower')
plt.colorbar(label='Number of simulations')
plt.xlabel('Copies of allele')
plt.ylabel('Generation')
plt.title('Wright-Fisher simulation matching Buri design')
plt.show()

print("Real final generation fixed at 0 or 32:", int(buri_data.iloc[-1, 0] + buri_data.iloc[-1, 32]))
print("Simulated final generation fixed at 0 or 32:", int(counts_buri[-1, 0] + counts_buri[-1, 32]))
```

## Instructor Answer Key Summary

**Question 1.** `P(A) = 8/20 = 0.4`.

**Question 2.** `P(B then B) = (12/20)^2 = 0.36`.

**Question 3.** `P(k=8) = C(20,8)(0.4)^8(0.6)^12 ≈ 0.1797`.

**Question 4.** `P(k != 8) = 1 - 0.1797 ≈ 0.8203`.

**Question 5.** `P(k ≥ 9) ≈ 0.4044`. The probability of decreasing is `P(k ≤ 7) ≈ 0.4159`, so the count is slightly more likely to decrease than increase from this starting value.

**Question 6.** The 100 simulated trajectories should show random drift, with some reaching 0 or 20 copies by generation 15 and then remaining there.

**Question 7.** The first-generation histogram should be centered near 10 copies, with spread due to binomial sampling.

**Question 8.** The 15th-generation histogram should be more spread out, with visible mass at 0 and 20. With the provided random seed, about `38` of 100 simulations reached 0 or 20 by generation 15.

**Question 9a-e.** The transition matrix has shape `21 × 21` for states `0..20`. The first and last rows are absorbing states. The initial distribution has probability 1 at state 10. Multiplying by powers of the transition matrix gives theoretical distributions for later generations; these should broadly match the simulation histograms, with smoother probabilities than the finite simulation sample.

**Question 10.** Buri's data are consistent with genetic drift: over generations, replicate populations spread away from the initial allele count and many reach fixation or loss.

**Question 11.** A Wright-Fisher simulation with gene pool size 32, initial count 16, 19 generations, and 107 simulations should show the same qualitative pattern as Buri's experiment. Exact fixation counts differ because the simulation is random and because real experiments can deviate from model assumptions. In the provided data, the final generation has `58` populations fixed at 0 or 32; one seeded simulation produced `28` fixed populations.
