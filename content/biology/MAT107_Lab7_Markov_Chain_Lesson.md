# MAT/BIS 107 Lab 7 Lesson Plan: Limiting Behavior of Markov Chains

## Instructor overview

This lesson converts the original MATLAB-based Lab 7 into a Python/Jupyter lesson. Students use Markov chains in two contexts: a two-state weather model for Davis, California, and a four-state nucleotide transition model for DNA sequences. The original lab asks students to compute limiting distributions, simulate Markov chains, and use dinucleotide transition patterns to distinguish two genomes with the same overall nucleotide composition.

**Estimated time:** one 90–120 minute lab period, or two shorter class meetings.

**Files needed:** `Ecoli-k12-genome.fasta`, `Organism_A.fasta`, and `Organism_B.fasta` in the same directory as the notebook.

**Python packages used:** `numpy`, `math`, and `matplotlib.pyplot`.

## Learning goals

By the end of this lesson, students should be able to:

1. Represent a simple Markov chain as a transition matrix.
2. Use matrix powers and eigenvectors to estimate a limiting distribution.
3. Simulate a Markov chain from a transition matrix.
4. Convert a DNA sequence into a dinucleotide transition matrix.
5. Explain why two sequences can have similar single-nucleotide composition but different transition behavior.

## Background for instructors

A **Markov chain** is a model for a system that moves among a set of states. The key assumption is that the next state depends only on the current state, not the full history. This is often called the **memoryless** property. In this lab, the first system has two states: dry and wet. The second system has four states: A, T, G, and C.

A **transition matrix** stores the probability of moving from one state to another. In this Python version, rows represent the current state and columns represent the next state. Each row should sum to 1.

A **limiting distribution** describes the long-run fraction of time a chain spends in each state. For many finite Markov chains, especially those that are irreducible and aperiodic, the limiting distribution can be found either by taking a large matrix power or by finding the eigenvector associated with eigenvalue 1. In practice, numerical eigenvectors may need to be normalized so their entries sum to 1.

In the DNA section, students count adjacent nucleotide pairs. These are **dinucleotides**. A genome can have approximately equal proportions of A, T, G, and C, but still have non-random transitions. For example, repeated A stretches would increase the A→A transition probability even if the total number of A bases is unchanged.

## Important caveats

- The Davis weather model is intentionally simple. It ignores seasonality, temperature, fog, and multi-year variation.
- Simulations are random. Students should get similar qualitative results but not identical trajectories.
- A Markov chain model of DNA is not a full biological model of genome structure. It captures only local one-step dependencies.
- FASTA files often contain header lines beginning with `>`. These should not be counted as sequence.
- Eigenvectors are determined up to a scaling constant, so raw eigenvector values may not sum to 1 until normalized.

## Suggested lesson flow

1. Start with a quick conceptual discussion: What information is in a transition matrix?
2. Work through Questions 1–8 as a guided example on limiting distributions.
3. Let students simulate weather chains for Questions 9–11.
4. Introduce FASTA files and dinucleotide pairs before Question 12.
5. Close with the organism comparison in Questions 14–16.

## Student notebook content

The following cells are designed to be copied directly into a Jupyter notebook or used as the provided `.ipynb` file.

```python
import numpy as np
import matplotlib.pyplot as plt
```

### Helper functions

```python
def stationary_distribution(P):
    """Return the stationary distribution for a row-stochastic transition matrix."""
    values, vectors = np.linalg.eig(P.T)
    index = np.argmin(np.abs(values - 1))
    vec = np.real(vectors[:, index])
    vec = vec / np.sum(vec)
    return vec


def simulate_markov_chain(P, initial_state, steps, rng=None):
    """Simulate states 0, 1, ..., n-1 from a row-stochastic transition matrix."""
    if rng is None:
        rng = np.random.default_rng()
    states = np.zeros(steps, dtype=int)
    states[0] = initial_state
    for i in range(1, steps):
        current = states[i - 1]
        states[i] = rng.choice(np.arange(P.shape[0]), p=P[current])
    return states


def read_fasta(filename):
    """Read a FASTA file and return one uppercase sequence string."""
    pieces = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith(">"):
                pieces.append(line.upper())
    return "".join(pieces)


def dna_transition_matrix(sequence, order="ATGC"):
    """Count dinucleotide transitions and return a transition matrix."""
    index = {base: i for i, base in enumerate(order)}
    counts = np.zeros((len(order), len(order)), dtype=int)
    for first, second in zip(sequence[:-1], sequence[1:]):
        if first in index and second in index:
            counts[index[first], index[second]] += 1
    transition_matrix = counts / counts.sum(axis=1, keepdims=True)
    return transition_matrix, counts
```

### Questions 1–3: Davis weather transition matrix

**Question 1.** Encode the Davis weather transition matrix and save it as `P`. Use row 0 for dry and row 1 for wet.

```python
P = np.array([[0.90, 0.10],
              [0.48, 0.52]])
P
```

**Question 2.** Compute a large power of the transition matrix, using a power larger than 100. What does the resulting matrix look like?

```python
P_power = np.linalg.matrix_power(P, 100)
P_power
```

**Question 3.** Assume the sequence starts dry, so `pi0 = [1, 0]`. What distribution does the system approach? Repeat for a wet starting day, `pi0 = [0, 1]`. Is the distribution the same?

```python
pi_dry = np.array([1, 0])
pi_wet = np.array([0, 1])

print("Starting dry:", pi_dry @ P_power)
print("Starting wet:", pi_wet @ P_power)
```

### Questions 4–8: Eigenvectors and interpretation

**Question 4.** Compute the eigenvalues and eigenvectors of the transpose of `P`.

```python
eigenvalues, eigenvectors = np.linalg.eig(P.T)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

**Question 5.** Find the eigenvector associated with eigenvalue 1. What is this eigenvector before normalization?

```python
idx = np.argmin(np.abs(eigenvalues - 1))
raw_vec = np.real(eigenvectors[:, idx])
raw_vec
```

**Question 6.** Does this raw eigenvector sum to 1?

```python
raw_vec.sum()
```

**Question 7.** Normalize the eigenvector so it sums to 1. Compare it with Question 3.

```python
weather_limit = raw_vec / raw_vec.sum()
weather_limit
```

**Question 8.** The limiting distribution predicts the long-run fraction of dry and wet days. Do you think this captures Davis weather over a full year? How might seasonality affect the model? How could the model be improved?

Write your answer in a markdown cell or as a Python comment.

### Questions 9–11: Simulating weather chains

**Question 9.** Simulate 365 days of weather using the Davis transition matrix. Start with a dry day.

```python
rng = np.random.default_rng(107)
T = 365
weather = simulate_markov_chain(P, initial_state=0, steps=T, rng=rng)
weather[:25]
```

```python
plt.figure()
plt.plot(np.arange(1, T + 1), weather + 1)
plt.xlabel("Day")
plt.ylabel("State: 1 = dry, 2 = wet")
plt.title("Simulated Davis Weather Chain")
plt.yticks([1, 2], ["dry", "wet"])
plt.show()
```

**Question 10.** Compute and plot the proportion of wet days over time. Does it approach the wet-day prediction from the limiting distribution?

```python
wet_over_time = np.cumsum(weather == 1) / np.arange(1, T + 1)

plt.figure()
plt.plot(np.arange(1, T + 1), wet_over_time, label="Simulated wet proportion")
plt.axhline(weather_limit[1], linestyle="--", label="Limiting wet proportion")
plt.xlabel("Time (days)")
plt.ylabel("Proportion of wet days")
plt.ylim(0, 1)
plt.legend()
plt.show()
```

**Question 11a.** Random Town has transition matrix `[[0.5, 0.5], [0.5, 0.5]]`. What is the limiting distribution?

```python
P_random = np.array([[0.5, 0.5],
                     [0.5, 0.5]])
random_limit = stationary_distribution(P_random)
random_limit
```

**Question 11b.** Simulate Random Town and compare its convergence to Davis. Does it seem faster or slower?

```python
random_weather = simulate_markov_chain(P_random, initial_state=0, steps=T, rng=rng)
random_wet_over_time = np.cumsum(random_weather == 1) / np.arange(1, T + 1)

plt.figure()
plt.plot(np.arange(1, T + 1), wet_over_time, label="Davis")
plt.axhline(weather_limit[1], linestyle="--", label="Davis limit")
plt.plot(np.arange(1, T + 1), random_wet_over_time, label="Random Town")
plt.axhline(random_limit[1], linestyle="--", label="Random Town limit")
plt.xlabel("Time (days)")
plt.ylabel("Proportion of wet days")
plt.ylim(0, 1)
plt.legend()
plt.show()
```

### Questions 12–16: DNA transition matrices

**Question 12.** Read `Ecoli-k12-genome.fasta`, compute the dinucleotide transition matrix, and decide whether dinucleotide pairs are used uniformly. Which transitions are largest?

```python
ecoli_seq = read_fasta("Ecoli-k12-genome.fasta")
ecoli_transition, ecoli_counts = dna_transition_matrix(ecoli_seq)
np.round(ecoli_transition, 4)
```

```python
bases = list("ATGC")
for i, first in enumerate(bases):
    for j, second in enumerate(bases):
        print(f"{first}->{second}: {ecoli_transition[i, j]:.4f}")
```

**Question 13.** Find the limiting distribution of the E. coli transition matrix. Are nucleotides used about equally?

```python
ecoli_limit = stationary_distribution(ecoli_transition)
np.round(ecoli_limit, 4)
```

**Question 14.** Use the same code to analyze `Organism_A.fasta` and `Organism_B.fasta`. What are their transition matrices?

```python
seq_A = read_fasta("Organism_A.fasta")
seq_B = read_fasta("Organism_B.fasta")

trans_A, counts_A = dna_transition_matrix(seq_A)
trans_B, counts_B = dna_transition_matrix(seq_B)

print("Organism A")
print(np.round(trans_A, 4))
print("\nOrganism B")
print(np.round(trans_B, 4))
```

**Question 15.** Compute the limiting distribution for each organism. Confirm that the overall compositions are similar.

```python
limit_A = stationary_distribution(trans_A)
limit_B = stationary_distribution(trans_B)

print("Organism A limiting distribution:", np.round(limit_A, 4))
print("Organism B limiting distribution:", np.round(limit_B, 4))
```

**Question 16.** Do dinucleotide transition matrices let us distinguish the organisms? Which genome likely belongs to *E. atrepeatium*, the organism with long A and T repeat regions?

Write your answer in a markdown cell or as a Python comment.

## Instructor Answer Key Summary

- Q1: `P = [[0.90, 0.10], [0.48, 0.52]]`.
- Q2: `P^100` has nearly identical rows: approximately `[0.8276, 0.1724]`.
- Q3: Starting from dry or wet converges to approximately `[0.8276, 0.1724]`.
- Q4–Q7: The eigenvalue-1 eigenvector normalizes to `[0.8276, 0.1724]`.
- Q8: Answers vary. Strong answer: a one-matrix model ignores Davis seasonality; a better model could estimate separate transition matrices by season or month.
- Q9–Q10: Simulations vary, but the wet-day proportion should move toward about `0.1724` over time.
- Q11a: Random Town limiting distribution is `[0.5, 0.5]`.
- Q11b: Random Town often appears to converge faster because each day is independent of the previous day.
- Q12: E. coli dinucleotide transitions are not uniform. The largest transitions include `G->C`, `T->T`, `C->G`, and `A->A`.
- Q13: E. coli limiting distribution is close to uniform, approximately `[0.2462, 0.2459, 0.2537, 0.2542]` in A, T, G, C order.
- Q14: Organism A transition matrix is approximately `[[0.4513, 0.0503, 0.2490, 0.2494], [0.0525, 0.4392, 0.2525, 0.2558], [0.0981, 0.1008, 0.3996, 0.4015], [0.1003, 0.0991, 0.3987, 0.4019]]`. Organism B transition matrix is approximately `[[0.2459, 0.2506, 0.2550, 0.2486], [0.2528, 0.2527, 0.2457, 0.2488], [0.0988, 0.1008, 0.3998, 0.4005], [0.0984, 0.1014, 0.3990, 0.4011]]`.
- Q15: Both have similar long-run composition, about 14% A, 14% T, 36% G, and 36% C.
- Q16: Organism A likely belongs to *E. atrepeatium* because it has high `A->A` and `T->T` transition probabilities.
