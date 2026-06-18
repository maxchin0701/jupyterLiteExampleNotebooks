# Lab 3: Sanger Sequencing

> **Python version of the original MAT/BIS 107 MATLAB lab.**  
> This lesson uses basic Python packages only: `math`, `random`/`numpy`, `pandas` when reading CSV files, and `matplotlib.pyplot` for plots. The original MATLAB labs asked students to submit a single script; this version is designed for a Jupyter notebook, where students run each code cell and write short explanations in markdown cells.

**Instructor note.** The code cells are intentionally readable rather than highly optimized. For class use, encourage students to run each cell, inspect intermediate variables, and answer the written questions in their own words.


## Lesson goals
Students will connect Sanger sequencing to geometric probability, explore how nucleotide chemistry affects fragment-length distributions, and use simulations to understand detection limits.

## Expanded background
Sanger sequencing uses normal nucleotides and chain-terminating nucleotides. During DNA synthesis, DNA polymerase extends a new strand complementary to a template. If a chain-terminating nucleotide is incorporated, extension stops. By measuring fragment lengths, researchers infer where that nucleotide occurred in the original sequence.

This lab studies one simplified pass of Sanger sequencing. For example, to find `T` positions in the original strand, the reaction contains normal `A` nucleotides plus defective `A` nucleotides. When the polymerase reaches a `T` in the template, it must add an `A`. If it adds a defective `A`, the fragment stops there. If it adds a normal `A`, synthesis continues.

Let `p` be the proportion of **good** nucleotides. Then `1-p` is the probability that the reaction stops at a given target position. The probability of stopping at the `k`-th target nucleotide is:

`P(stop at k) = p^(k-1) (1-p)`

### Caveats
This model focuses on one nucleotide type and assumes a constant probability `p`, independent events, unlimited nucleotide supply, and perfect fragment-size detection above a threshold. Real Sanger sequencing includes fluorescent labels, electropherogram peak quality, polymerase bias, secondary structure, noise, and declining read quality over long fragments.

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(107)

def geometric_stop_pmf(k, p_good):
    # Probability that sequencing stops at the k-th target base.
    # k starts at 1. p_good is the proportion of non-defective nucleotides.
    k = np.asarray(k)
    return (p_good ** (k - 1)) * (1 - p_good)

def simulate_sanger(n_experiments, n_targets, p_good, rng=None):
    # Simulate Sanger outcomes.
    # Returns values 1..n_targets for stopped fragments and n_targets+1 for read-through.
    if rng is None:
        rng = np.random.default_rng()
    # numpy geometric gives trial number of first success, where success is defective incorporation.
    stops = rng.geometric(1 - p_good, size=n_experiments)
    return np.minimum(stops, n_targets + 1)
```

## Student Questions and Code
The following sections are written as student-facing prompts. Students should run the relevant code cells, inspect the output, and write their own answers before looking at the instructor key.

## Part 1: The fragment-length distribution
### Question 1
Plot the probability distribution over values of `k` for `p = 0.5`, `0.7`, and `0.95`. What trends do you observe as `p` gets closer to 1?

### Question 2
Which `p` gives the highest probability of ending at the 10th `T`?

### Question 3
What happens as `k` gets large? How does this affect sequencing of very long DNA fragments?

```python
ks = np.arange(1, 11)
for p_good in [0.5, 0.7, 0.95]:
    plt.plot(ks, geometric_stop_pmf(ks, p_good), marker='o', label=f'p={p_good}')

plt.xlabel('k-th target nucleotide')
plt.ylabel('Probability')
plt.title('Probability of stopping at the k-th target nucleotide')
plt.xticks(ks)
plt.legend()
plt.show()

for p_good in [0.5, 0.7, 0.95]:
    print(f"p={p_good}: P(stop at 10th target) = {geometric_stop_pmf(10, p_good):.6f}")
```

## Part 2: Optimizing the proportion of good nucleotides
The expected number of experiments needed to detect the `k`-th target nucleotide is the inverse of the probability of that event:

`E[number of experiments] = 1 / (p^(k-1)(1-p))`

### Question 4
Plot the expected number of experiments needed to detect the `k`-th target for `k=1`, `k=2`, and `k=3`.

### Question 5
Describe the trends.

### Question 6
The optimal value for detecting the `k`-th target is `p = (k-1)/k`. Compute the optimal `p` for `k=1`, `k=2`, and `k=3`, and plot those points on your curves.

```python
p_values = np.linspace(0.001, 0.999, 1000)

for k in [1, 2, 3]:
    expected_experiments = 1 / geometric_stop_pmf(k, p_values)
    plt.plot(p_values, expected_experiments, label=f'k={k}')

# Add optimal points.
for k in [1, 2, 3]:
    p_opt = 0 if k == 1 else (k - 1) / k
    # Avoid p=0 exactly in the curve calculation, but the formula is valid for k=1.
    prob = geometric_stop_pmf(k, p_opt) if p_opt < 1 else np.nan
    exp_n = 1 / prob
    plt.scatter([p_opt], [exp_n])
    print(f"k={k}, optimal p={p_opt}, expected experiments={exp_n}")

plt.ylim(0, 20)
plt.xlabel('Proportion of good nucleotides, p')
plt.ylabel('Expected number of experiments')
plt.title('Expected experiments needed for detection')
plt.legend()
plt.show()
```

## Part 3: Simulating Sanger experiments
### Question 7
Simulate 10,000 Sanger experiments for a DNA strand with 30 target `T` positions using `p=0.8` and `p=0.9`. Plot the distribution of fragment lengths for each case. Which value seems better?

### Question 8
Repeat with the optimal `p` for detecting the 30th target.

### Question 9
Assume the detection limit is `D = 0.01`: a fragment length must make up at least 1% of the pool to be detected. Can all fragment lengths be detected? What challenge appears for DNA with more than about 100 target positions?

```python
n_targets = 30
n_experiments = 10_000

for p_good in [0.8, 0.9]:
    outcomes = simulate_sanger(n_experiments, n_targets, p_good, rng)
    plt.hist(outcomes, bins=np.arange(1, n_targets + 3) - 0.5, alpha=0.6, label=f'p={p_good}')

plt.xlabel('Fragment outcome: stop position, or 31 for read-through')
plt.ylabel('Number of experiments')
plt.title('Simulated Sanger fragment outcomes')
plt.legend()
plt.show()

p_opt = (n_targets - 1) / n_targets
outcomes_opt = simulate_sanger(n_experiments, n_targets, p_opt, rng)
plt.hist(outcomes_opt, bins=np.arange(1, n_targets + 3) - 0.5)
plt.xlabel('Fragment outcome: stop position, or 31 for read-through')
plt.ylabel('Number of experiments')
plt.title(f'Simulated outcomes using optimal p={p_opt:.3f}')
plt.show()

# Detection threshold check
D = 0.01
counts = np.bincount(outcomes_opt, minlength=n_targets + 2)[1:]
frequencies = counts / n_experiments
undetectable = np.where(frequencies < D)[0] + 1
print("Fragment outcomes below detection threshold:", undetectable)
print("Frequencies for first 10 outcomes:", frequencies[:10])
```

## Instructor Answer Key Summary

**Question 1.** As `p` gets closer to 1, the distribution shifts toward later target positions. When `p` is low, the reaction is more likely to terminate early.

**Question 2.** Among `p=0.5`, `p=0.7`, and `p=0.95`, `p=0.95` gives the highest probability of ending at the 10th target nucleotide. The probabilities are approximately `0.00098`, `0.01211`, and `0.03151`, respectively.

**Question 3.** For any fixed `p < 1`, the probability of stopping at the `k`-th target eventually decreases as `k` becomes large. This makes long fragments harder to detect unless many experiments are run and `p` is chosen carefully.

**Question 4.** The expected number of experiments is `1 / (p^(k-1)(1-p))`. The plotted curves show that early targets are best detected with lower `p`, while later targets require higher `p`.

**Question 5.** For `k=1`, increasing `p` makes detection harder because the first target is skipped more often. For larger `k`, very low `p` causes too much early termination, while very high `p` causes too many read-through events; an intermediate `p` is best.

**Question 6.** The optimal `p` values are `0` for `k=1`, `0.5` for `k=2`, and `2/3 ≈ 0.667` for `k=3`. The corresponding expected experiment counts are `1`, `4`, and `6.75`.

**Question 7.** For 30 target positions, `p=0.9` is generally better than `p=0.8` for seeing later fragments because it reduces early termination.

**Question 8.** The optimal value for detecting the 30th target is `(30-1)/30 = 29/30 ≈ 0.967`.

**Question 9.** Not all fragment lengths may be detectable at a 1% detection threshold. Long-target positions have low probabilities, so a large number of experiments may be needed. For long DNA molecules with many targets, optimizing one region can make other regions poorly represented.
