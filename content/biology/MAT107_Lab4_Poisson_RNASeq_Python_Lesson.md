# Lab 4: The Poisson Distribution and RNA-Seq

> **Python version of the original MAT/BIS 107 MATLAB lab.**  
> This lesson uses basic Python packages only: `math`, `random`/`numpy`, `pandas` when reading CSV files, and `matplotlib.pyplot` for plots. The original MATLAB labs asked students to submit a single script; this version is designed for a Jupyter notebook, where students run each code cell and write short explanations in markdown cells.

**Instructor note.** The code cells are intentionally readable rather than highly optimized. For class use, encourage students to run each cell, inspect intermediate variables, and answer the written questions in their own words.


## Lesson goals
Students will compare binomial and Poisson distributions, simulate a simplified RNA-Seq experiment, estimate relative abundance, and see why overdispersed biological count data may need a Negative Binomial model.

## Expanded background
The Poisson distribution models the number of events in a fixed interval when events are independent and occur at a constant average rate. It is often used for count data. RNA-Seq data are also counts: researchers sequence many RNA-derived fragments and count how many map to each gene.

RNA-Seq counts are affected by biological abundance, transcript length, primer binding, reverse-transcription efficiency, sequencing depth, and mapping. Longer transcripts provide more possible primer-binding positions, so raw counts must be adjusted by effective length before comparing gene abundance.

### Caveats
This is a simplified RNA-Seq model. Real RNA-Seq analysis involves library preparation bias, PCR amplification, sequencing errors, multimapping reads, normalization across samples, batch effects, biological replicates, and sophisticated statistical tools. The Negative Binomial fit here uses a simple method-of-moments estimate rather than specialized software.

```python
import math
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(107)

def poisson_pmf(k, lam):
    k = np.asarray(k)
    return np.exp(-lam) * (lam ** k) / np.array([math.factorial(int(x)) for x in k])

def binomial_pmf_array(k_values, n, p):
    return np.array([math.comb(n, int(k)) * (p ** int(k)) * ((1-p) ** (n-int(k))) for k in k_values])

def negative_binomial_pmf(k_values, r, p):
    # Negative binomial PMF for number of failures k before r successes.
    # Mean = r(1-p)/p; variance = r(1-p)/p^2. Uses log-gamma so r can be non-integer.
    vals = []
    for k in k_values:
        log_coef = math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
        vals.append(math.exp(log_coef + r * math.log(p) + k * math.log(1-p)))
    return np.array(vals)
```

## Student Questions and Code
The following sections are written as student-facing prompts. Students should run the relevant code cells, inspect the output, and write their own answers before looking at the instructor key.

## Part 1: Poisson distribution
### Question 1
Draw 1000 samples from a Poisson random variable with `lambda = 10`. What are the mean and variance?

### Question 2
Plot a binomial distribution with `n=10000`, `p=0.001` over `k=0..20`, and compare it to a Poisson distribution with `lambda = n*p = 10`. Do they agree?

```python
samples = rng.poisson(lam=10, size=1000)
print("sample mean:", samples.mean())
print("sample variance:", samples.var(ddof=1))

n = 10_000
p = 0.001
lam = n * p
points = np.arange(0, 21)
points_bino = binomial_pmf_array(points, n, p)
points_poiss = poisson_pmf(points, lam)

width = 0.4
plt.bar(points - width/2, points_bino, width=width, label='Binomial')
plt.bar(points + width/2, points_poiss, width=width, label='Poisson')
plt.xlabel('k')
plt.ylabel('Probability')
plt.title('Binomial and Poisson distributions')
plt.legend()
plt.show()
```

## Part 2: A simplified RNA-Seq simulation
We simulate five genes. Each gene has a transcript length `ti` and a number of RNA fragments `ni`. Primers bind possible sites, reverse transcriptase creates cDNA fragments, and only fragments with length at least 30 are counted.

### Question 3
Simulate one RNA-Seq experiment using:

`ti = [200, 300, 1000, 500, 450]`  
`ni = [5, 3, 5, 7, 10]`  
`np = 200` primers

```python
def rna_seq_sim(ti, ni, n_primers, primer_len=6, rt_drop_p=0.01, min_map_len=30, rng=None):
    # Simplified RNA-Seq simulation.
    # Returns counts per gene and a fragment array with columns: gene index, start position, length.
    if rng is None:
        rng = np.random.default_rng()
    ti = np.asarray(ti, dtype=int)
    ni = np.asarray(ni, dtype=float)
    G = len(ti)

    # If any stochastic ni values are zero, they contribute no binding sites.
    binding_sites = np.maximum(ti + 1 - primer_len, 0) * ni
    if binding_sites.sum() == 0:
        return np.zeros(G, dtype=int), np.empty((0, 3), dtype=int)
    probs = binding_sites / binding_sites.sum()

    bound_genes = rng.choice(np.arange(G), size=n_primers, replace=True, p=probs)
    frags = np.zeros((n_primers, 3), dtype=int)

    for i, gene in enumerate(bound_genes):
        t = ti[gene]
        start = rng.integers(0, t)  # 0-based binding position
        failures_before_drop = rng.geometric(rt_drop_p) - 1
        length = primer_len + failures_before_drop
        length = min(length, t - start)
        frags[i] = [gene, start, length]

    detected = frags[frags[:, 2] >= min_map_len]
    counts = np.bincount(detected[:, 0], minlength=G) if len(detected) else np.zeros(G, dtype=int)
    return counts, frags

ti = np.array([200, 300, 1000, 500, 450])
ni = np.array([5, 3, 5, 7, 10])
n_primers = 200
counts, frags = rna_seq_sim(ti, ni, n_primers, rng=rng)
print("counts:", counts)
print("first five fragments [gene, start, length]:")
print(frags[:5])
```

### Questions 4-5: Relative abundance
Because longer genes have more possible primer binding sites, raw counts should be adjusted by effective length `ti - 5`.

`relative abundance_i = (counts_i / (ti_i - 5)) / sum(counts_j / (ti_j - 5))`

Compute this adjusted relative abundance. Compare it to the true relative abundance `ni / sum(ni)`.

```python
effective_lengths = ti - 5
adjusted = counts / effective_lengths
relative_abundance = adjusted / adjusted.sum()
true_relative_abundance = ni / ni.sum()

print("estimated relative abundance:", np.round(relative_abundance, 3))
print("true relative abundance:     ", np.round(true_relative_abundance, 3))
```

## Part 3: Count distributions across experiments
### Question 6
Run 1000 experiments with the same parameters and save the counts for the first gene.

### Question 7
Fit a Poisson distribution using the sample mean and overlay it on the histogram. How well does it fit?

```python
n_sims = 1000
first_gene_counts = np.empty(n_sims, dtype=int)
for i in range(n_sims):
    counts, _ = rna_seq_sim(ti, ni, n_primers, rng=rng)
    first_gene_counts[i] = counts[0]

plt.hist(first_gene_counts, bins=np.arange(first_gene_counts.max()+2)-0.5, density=True, alpha=0.7, label='Simulated counts')
mu = first_gene_counts.mean()
ks = np.arange(0, first_gene_counts.max()+1)
plt.plot(ks, poisson_pmf(ks, mu), marker='o', label=f'Poisson fit, mean={mu:.2f}')
plt.xlabel('Counts for gene 1')
plt.ylabel('Density')
plt.title('Counts across repeated RNA-Seq simulations')
plt.legend()
plt.show()

print("mean:", mu)
print("variance:", first_gene_counts.var(ddof=1))
```

## Part 4: Biological variation and overdispersion
In real biological experiments, the underlying transcript abundances vary across samples. To simulate this, replace `ni` with a Poisson random sample around `ni` for each experiment.

### Question 8
Run 1000 simulations with variable `ni`.

### Question 9
Fit a Poisson distribution. Does it still fit well?

### Question 10
Fit a Negative Binomial distribution using a simple method-of-moments estimate. Does it capture the data better?

### Question 11
Why might a Negative Binomial be a suitable model for RNA-Seq counts?

```python
variable_counts = np.empty(n_sims, dtype=int)
for i in range(n_sims):
    ni_random = rng.poisson(ni)
    counts, _ = rna_seq_sim(ti, ni_random, n_primers, rng=rng)
    variable_counts[i] = counts[0]

mu = variable_counts.mean()
var = variable_counts.var(ddof=1)
ks = np.arange(0, variable_counts.max()+1)

plt.hist(variable_counts, bins=np.arange(variable_counts.max()+2)-0.5, density=True, alpha=0.7, label='Simulated counts')
plt.plot(ks, poisson_pmf(ks, mu), marker='o', label=f'Poisson mean={mu:.2f}')

if var > mu:
    r = mu**2 / (var - mu)
    p_nb = mu / var
    nb_fit = negative_binomial_pmf(ks, r, p_nb)
    plt.plot(ks, nb_fit, marker='s', label=f'NegBin MOM fit')
    print("Negative binomial r:", r)
    print("Negative binomial p:", p_nb)
else:
    print("Variance is not greater than mean; Negative Binomial overdispersion is not needed here.")

plt.xlabel('Counts for gene 1')
plt.ylabel('Density')
plt.title('Counts with variable biological abundance')
plt.legend()
plt.show()

print("mean:", mu)
print("variance:", var)
```

## Instructor Answer Key Summary

**Question 1.** The sample mean and variance should both be close to `λ = 10`. With the provided random seed, one run gives mean about `10.12` and sample variance about `10.96`.

**Question 2.** The binomial distribution with `n=10000`, `p=0.001` and the Poisson distribution with `λ=10` are nearly identical over `k=0..20`, because `n` is large and `p` is small.

**Question 3.** Counts vary by simulation. With the provided random seed, one run gives counts similar to `[8, 8, 51, 41, 40]`. Students should focus on the structure of the simulation and why longer transcripts receive more primer-binding opportunities.

**Questions 4-5.** Raw counts should be divided by effective transcript length before comparing relative abundance. With the seed used here, the adjusted estimate is approximately `[0.140, 0.093, 0.175, 0.284, 0.308]`, compared with the true relative abundance `[0.167, 0.100, 0.167, 0.233, 0.333]`. Differences are expected because of random sampling and the small number of primers.

**Question 6.** The first-gene counts across repeated experiments form a count distribution. With fixed `ni`, the mean is about `9` in the provided run.

**Question 7.** The Poisson fit should be reasonable when the underlying abundances are fixed. With the provided seed, the first-gene count mean is about `8.99` and variance about `8.25`.

**Question 8.** When `ni` varies across experiments, the count distribution becomes wider.

**Question 9.** A Poisson distribution is usually too narrow for the variable-abundance case because the variance is greater than the mean.

**Question 10.** A Negative Binomial fit better captures overdispersion when the variance exceeds the mean. With the provided seed, the mean is about `8.95` and the variance about `26.22`.

**Question 11.** A Negative Binomial can be suitable for RNA-Seq because RNA-Seq counts often include both technical sampling variation and biological variation across samples, producing overdispersion relative to a Poisson model.
