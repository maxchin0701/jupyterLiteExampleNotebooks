---
title: mat107_sanger_sequencing_lesson.md

---

# Modeling Sanger Sequencing with Probability and Python

**Estimated time:** one class day, about 3–4 hours total  
**Computing format:** Jupyter Notebook, JupyterLite, Google Colab, or local Python  
**Python level:** beginner  
**Python Packages Used:** `math`, `random`, `collections.Counter`, and `matplotlib.pyplot` for graphs  

---

## Big Idea For The Day

Sanger sequencing is a laboratory method for reading DNA sequence information. It works because DNA polymerase builds a new DNA strand one nucleotide at a time, but sometimes the reaction includes special defective nucleotides that stop the new strand from growing.

In this lab, students do not run a wet-lab sequencing experiment. Instead, they build a mathematical and computational model of one part of the experiment. They ask:

> How does the mixture of normal and defective nucleotides affect which DNA fragment lengths we can detect?

This is a probability question. Each time the polymerase reaches a target base, it randomly chooses either a normal nucleotide or a defective nucleotide. If it chooses a normal nucleotide, the strand keeps growing. If it chooses a defective nucleotide, the strand stops. Because this choice is random, different experimental copies can stop at different positions.

We can see this visualized here. 

![sanger_sequencing](https://hackmd.io/_uploads/HJdRwoZMzl.jpg)

---

## Learning Goals

By the end of the lesson, students should be able to:

1. Explain, in simple words, why Sanger sequencing produces DNA fragments of different lengths.
2. Describe what the parameter `p` means in this model.
3. Use the formula `P(ending at the kth T) = p^(k-1)(1-p)` to calculate probabilities.
4. Explain why very small or very large values of `p` can cause problems.
5. Simulate many sequencing experiments using Python.
6. Use a histogram to connect a probability model to observed simulated data.
7. Explain why long DNA fragments can become difficult to detect in practice.

---

# Background

## 1. DNA replication as a copying process

DNA is usually double-stranded. Each strand is made from four nucleotide letters: A, T, C, and G. The two strands pair in a predictable way:

- A pairs with T
- T pairs with A
- C pairs with G
- G pairs with C

During replication, the two DNA strands separate. DNA polymerase reads one strand and builds the complementary strand. For example, if the original strand is:

```text
A C C T G T C C G
```

then the complementary strand is:

```text
T G G A C A G G C
```

This copying process is usually very reliable. But Sanger sequencing intentionally adds a small amount of defective nucleotides.

## 2. What makes a nucleotide “defective”?

In this simplified model, a defective nucleotide can be added to the growing DNA strand, but once it is added, the strand cannot grow any farther. The defective nucleotide acts like a stop sign.

For example, suppose we are trying to identify positions where the original strand contains `T`. To do that, the reaction includes a mixture of normal `A` nucleotides and defective `A` nucleotides. This works because DNA polymerase adds `A` across from `T`.

When the polymerase reaches a `T` on the original strand:

- If it adds a normal `A`, copying continues.
- If it adds a defective `A`, copying stops.

The length of the stopped fragment tells us where the stop happened. If the stopped fragment has length 4, then the defective nucleotide was added at position 4 of the new strand. That means the original strand had a `T` at that position.

## 3. Why repeated experiments produce different lengths

The reaction is repeated many times at the molecular level. Each copy of the DNA strand experiences its own random sequence of choices. One copy might stop at the first `T`. Another copy might pass the first `T` and stop at the second `T`. Another might pass many `T` positions before stopping.

So the observed fragment length is a **random variable**. It is not the same every time.

This is the central mathematical idea of the lab: Sanger sequencing has a built-in random process, and probability helps us predict which fragment lengths will be common or rare.

## 4. The meaning of `p`

In this lab, `p` means:

> the proportion of non-defective, or “good,” nucleotides in the reaction mixture.

So:

- `p = 0` means all target nucleotides are defective.
- `p = 1` means all target nucleotides are good.
- `p = 0.9` means 90% are good and 10% are defective.

The probability of choosing a defective nucleotide is:

```text
1 - p
```

## 5. Probability of stopping at the kth T

Suppose we are studying the locations of `T` in the original DNA strand. To stop at the first `T`, the polymerase must choose a defective `A` immediately.

```text
P(stop at 1st T) = 1 - p
```

To stop at the second `T`, the polymerase must first choose a good `A` at the first `T`, then choose a defective `A` at the second `T`.

```text
P(stop at 2nd T) = p(1 - p)
```

To stop at the third `T`, it must choose good nucleotides at the first two `T` positions, then a defective nucleotide at the third `T`.

```text
P(stop at 3rd T) = p * p * (1 - p) = p^2(1 - p)
```

In general:

```text
P(stop at kth T) = p^(k-1)(1 - p)
```

This is a version of the geometric distribution. Students do not need advanced probability to use it. They only need the idea that the experiment must “continue” `k-1` times and then “stop” once.

## 6. Why `p` cannot be too low or too high

The value of `p` controls how far the polymerase usually gets.

If `p` is too low, many nucleotides are defective. Most fragments stop very early. This makes it easy to detect early positions but hard to detect later positions.

If `p` is too high, most nucleotides are good. Many fragments do not stop until very late, or may not stop within the region we care about. This can also make some positions hard to detect.

So there is a tradeoff. A good experiment needs enough defective nucleotides to stop synthesis, but enough good nucleotides to allow the polymerase to reach later positions.

## 7. Expected number of experiments

If the probability of observing a certain event is small, we expect to repeat the experiment many times before seeing it. In this lab, we use:

```text
Expected number of experiments = 1 / probability of the event
```

For the kth `T`:

```text
Expected experiments to detect kth T = 1 / [p^(k-1)(1-p)]
```

This lets students connect probability to experimental effort. Rare fragment lengths require many experiments to observe.

## 8. Optimal `p`

The lab gives the formula:

```text
optimal p = (k - 1) / k
```

This is the value of `p` that minimizes the expected number of experiments needed to detect the kth `T`.

Examples:

| k | optimal p |
|---:|---:|
| 1 | 0 |
| 2 | 0.5 |
| 3 | 0.667 |
| 30 | 0.967 |

As the target position gets farther away, the optimal `p` gets closer to 1. That makes sense: to reach a late position, most nucleotides need to be good.

## 9. Practical limitation: detection threshold

Even if a fragment length can theoretically occur, it may be too rare to detect in practice. The lab uses a detection threshold `D`. If `D = 0.01`, then a fragment must make up at least 1% of all fragments to be detected.

For 10,000 simulated fragments:

```text
0.01 * 10000 = 100 fragments
```

So any fragment length that appears fewer than 100 times may be considered below the detection limit.

This is important for long DNA fragments. As DNA gets longer, the probability gets spread across many possible fragment lengths. Each individual length may become too rare to detect clearly.

---

# Student lesson

## Warm-up

Consider this DNA strand:

```text
A C C T G T C C G
```

1. What base pairs with A?
2. What base pairs with T?
3. What base pairs with C?
4. What base pairs with G?
5. What is the complementary strand?

**Instructor answer:**

```text
T G G A C A G G C
```

---

# Python Setup

Run this cell first.

```python
import math
import random
from collections import Counter
import matplotlib.pyplot as plt
```

This lesson uses regular Python lists and loops. It does not use MATLAB, NumPy, SciPy, or pandas.

---

# Helper Functions

Run this cell once. These functions will be used throughout the lab.

```python
def probability_stop_at_k(k, p):
    """
    Return the probability that the sequencing reaction stops at the kth T.

    k: which T we are asking about. The first T is k = 1.
    p: proportion of good nucleotides.
    """
    return (p ** (k - 1)) * (1 - p)


def expected_experiments(k, p):
    """
    Return the expected number of experiments needed to detect the kth T.
    This is 1 divided by the probability of stopping at the kth T.
    """
    probability = probability_stop_at_k(k, p)

    # If the probability is 0, the expected number would be infinite.
    if probability == 0:
        return math.inf

    return 1 / probability


def make_p_values(start=0.001, stop=0.999, step=0.001):
    """
    Make a list of p values without using NumPy.
    We avoid exactly 0 and exactly 1 for some plots because division by zero can occur.
    """
    values = []
    current = start
    while current <= stop:
        values.append(current)
        current += step
    return values


def simulate_one_experiment(p, Nk):
    """
    Simulate one Sanger sequencing experiment.

    p: proportion of good nucleotides.
    Nk: number of T positions in the DNA strand.

    Returns:
    - 1 through Nk if the experiment stops at that T
    - Nk + 1 if the experiment reads through the entire region without stopping
    """
    k = 1

    while k <= Nk:
        random_number = random.random()

        # A defective nucleotide is chosen with probability 1 - p.
        if random_number < (1 - p):
            return k

        # Otherwise, a good nucleotide was chosen and copying continues.
        k += 1

    # If we get here, the reaction never stopped within the Nk positions.
    return Nk + 1


def simulate_many_experiments(p, Nk, N_experiments):
    """
    Simulate many Sanger sequencing experiments.
    """
    outcomes = []
    for i in range(N_experiments):
        outcome = simulate_one_experiment(p, Nk)
        outcomes.append(outcome)
    return outcomes
```

---

# Part 1: Probability of stopping at the kth T

## Question 1

Plot the probability distribution over values of `k` for:

- `p = 0.5`
- `p = 0.7`
- `p = 0.95`

What trends do you observe as `p` gets closer to 1?

```python
k_values = list(range(1, 11))
p_values = [0.5, 0.7, 0.95]

for p in p_values:
    probabilities = []
    for k in k_values:
        probability = probability_stop_at_k(k, p)
        probabilities.append(probability)

    plt.plot(k_values, probabilities, marker="o", label=f"p = {p}")

plt.ylim(0, 0.6)
plt.xlabel("k")
plt.ylabel("Probability")
plt.title("Probability of Ending at the kth T")
plt.legend()
plt.show()
```

**Discussion prompt:**

As `p` gets closer to 1, the polymerase is more likely to keep going. This means early stops become less common, and later stops become more possible.

---

## Question 2

Which `p` from Question 1 gives the highest probability of an experiment ending at the 10th `T`?

```python
for p in p_values:
    probability = probability_stop_at_k(10, p)
    print("p =", p, "probability of stopping at 10th T =", round(probability, 4))
```

**Expected result:**

`p = 0.95` gives the highest probability of ending at the 10th `T` among these three values.

---

## Question 3

What trend do you observe as `k` gets large? How does this affect the sequencing of very large DNA fragments?

**Suggested answer:**

For a fixed value of `p`, the probability usually becomes smaller as `k` gets larger. This means it can be difficult to detect positions that are far away from the starting point. In real experiments, very long fragments may be rare or hard to detect.

---

# Part 2: Expected number of experiments

The expected number of experiments is:

```text
1 / probability
```

For this lab:

```text
Expected experiments to detect kth T = 1 / [p^(k-1)(1-p)]
```

## Question 4

Plot the expected number of experiments needed to detect the kth `T` over a range of `p` values for:

- `k = 1`
- `k = 2`
- `k = 3`

```python
p_grid = make_p_values()
k_values = [1, 2, 3]

for k in k_values:
    expected_values = []
    for p in p_grid:
        value = expected_experiments(k, p)
        expected_values.append(value)

    plt.plot(p_grid, expected_values, label=f"k = {k}")

plt.xlim(0, 1)
plt.ylim(0, 10)
plt.xlabel("Proportion of Good Nucleotides, p")
plt.ylabel("Expected Number of Experiments")
plt.title("Expected Experiments Needed to Detect kth T")
plt.legend()
plt.show()
```

---

## Question 5

For each value of `k`, describe how the expected number of experiments changes as `p` changes.

**Suggested answer:**

For `k = 1`, the expected number of experiments is smallest when `p` is close to 0. This is because if most nucleotides are defective, the reaction usually stops at the first `T`.

For larger `k`, the pattern changes. If `p` is too low, the reaction stops too early. If `p` is too high, the reaction may not stop often enough. For each `k`, there is a value of `p` that gives the lowest expected number of experiments.

---

# Part 3: Optimizing p

The original lab gives this formula for the optimal value of `p`:

```text
optimal p = (k - 1) / k
```

This formula finds the `p` that minimizes the expected number of experiments needed to detect the kth `T`.

## Question 6

Use the formula to calculate the optimal `p` for:

- `k = 1`
- `k = 2`
- `k = 3`

Then plot these points on top of the curves from Question 4.

```python
p_grid = make_p_values()
k_values = [1, 2, 3]

# Plot the expected experiment curves.
for k in k_values:
    expected_values = []
    for p in p_grid:
        expected_values.append(expected_experiments(k, p))

    plt.plot(p_grid, expected_values, label=f"k = {k}")

# Calculate and plot the optimal points.
for k in k_values:
    p_opt = (k - 1) / k
    n_opt = expected_experiments(k, p_opt)

    print("k =", k, "optimal p =", round(p_opt, 3), "expected experiments =", round(n_opt, 3))
    plt.plot(p_opt, n_opt, marker="*", markersize=12)

plt.xlim(0, 1)
plt.ylim(0, 10)
plt.xlabel("p")
plt.ylabel("Expected Number of Experiments")
plt.title("Optimal p Values")
plt.legend()
plt.show()
```

**Discussion prompt:**

The star points should appear at the lowest points of the curves. These are the values of `p` that make detection most efficient for each `k`.

---

# Part 4: Simulations

So far, we have used formulas. Now we will simulate many experiments and see whether the simulated data match the ideas from the formulas.

A simulated experiment works like this:

1. Start at the first `T`.
2. Randomly decide whether the nucleotide is good or defective.
3. If it is defective, stop and record that `k`.
4. If it is good, move to the next `T`.
5. Repeat until the reaction stops or reaches the end.

---

## Try one simulation

```python
p = 0.5
Nk = 30

single_outcome = simulate_one_experiment(p, Nk)
print(single_outcome)
```

If the output is a number from `1` to `30`, the experiment stopped at that `T`. If the output is `31`, the experiment read through all 30 `T` positions without stopping.

---

## Question 7

Simulate 10,000 Sanger experiments for a DNA strand with 30 `T` positions using:

- `p = 0.8`
- `p = 0.9`

Plot the distribution of fragment lengths for each case. Which value seems to work better?

```python
Nk = 30
N_experiments = 10000

p = 0.8
outcomes_08 = simulate_many_experiments(p, Nk, N_experiments)

plt.hist(outcomes_08, bins=range(1, Nk + 3), edgecolor="black")
plt.xlabel("Fragment Length / Stop Position")
plt.ylabel("Counts")
plt.title("Simulated Sanger Experiments, p = 0.8")
plt.show()

p = 0.9
outcomes_09 = simulate_many_experiments(p, Nk, N_experiments)

plt.hist(outcomes_09, bins=range(1, Nk + 3), edgecolor="black")
plt.xlabel("Fragment Length / Stop Position")
plt.ylabel("Counts")
plt.title("Simulated Sanger Experiments, p = 0.9")
plt.show()
```

**Suggested answer:**

`p = 0.9` usually gives better coverage of later fragment lengths than `p = 0.8`. With `p = 0.8`, many experiments stop early.

---

## Question 8

Repeat the simulation, but this time use the optimal `p` for a DNA strand with 30 `T` positions.

```python
Nk = 30
N_experiments = 10000
p_opt = (Nk - 1) / Nk

print("Optimal p for Nk = 30:", round(p_opt, 4))

outcomes_opt = simulate_many_experiments(p_opt, Nk, N_experiments)

plt.hist(outcomes_opt, bins=range(1, Nk + 3), edgecolor="black")
plt.xlabel("Fragment Length / Stop Position")
plt.ylabel("Counts")
plt.title("Simulated Sanger Experiments, Optimal p")
plt.show()
```

**Suggested answer:**

The distribution should be more spread out across the DNA region. Many experiments may also read through the entire region, which appears as `Nk + 1`, or `31` in this example.

---

## Question 9

Assume the detection limit is:

```text
D = 0.01
```

This means a fragment length must make up at least 1% of all fragments to be detected.

For 10,000 experiments, the count must be at least:

```text
0.01 * 10000 = 100
```

Can all fragment lengths in Question 8 be detected?

```python
D = 0.01
D_limit = N_experiments * D

counts = Counter(outcomes_opt)

print("Detection limit:", D_limit, "counts")
print()

for fragment_length in range(1, Nk + 2):
    count = counts[fragment_length]
    detectable = count >= D_limit
    print(fragment_length, "count =", count, "detectable =", detectable)

plt.hist(outcomes_opt, bins=range(1, Nk + 3), edgecolor="black")
plt.axhline(D_limit, linestyle="--", label="Detection limit")
plt.xlabel("Fragment Length / Stop Position")
plt.ylabel("Counts")
plt.title("Detection Limit for Simulated Fragments")
plt.legend()
plt.show()
```

**Suggested answer:**

For `Nk = 30`, many or all fragment lengths may be detectable, depending on the random simulation. However, as the DNA gets much longer, the fragments are spread across many more possible lengths. Then each individual length may appear too rarely to pass the detection limit.

---

# Optional extension: What happens when the DNA is longer?

Try changing `Nk` from 30 to 100.

```python
Nk = 100
N_experiments = 10000
p_opt = (Nk - 1) / Nk
D = 0.01
D_limit = N_experiments * D

outcomes_long = simulate_many_experiments(p_opt, Nk, N_experiments)
counts_long = Counter(outcomes_long)

number_detectable = 0
for fragment_length in range(1, Nk + 2):
    if counts_long[fragment_length] >= D_limit:
        number_detectable += 1

print("Nk =", Nk)
print("Optimal p =", round(p_opt, 4))
print("Detection limit =", D_limit, "counts")
print("Number of detectable fragment lengths =", number_detectable, "out of", Nk + 1)

plt.hist(outcomes_long, bins=range(1, Nk + 3), edgecolor="black")
plt.axhline(D_limit, linestyle="--", label="Detection limit")
plt.xlabel("Fragment Length / Stop Position")
plt.ylabel("Counts")
plt.title("Longer DNA Region: Nk = 100")
plt.legend()
plt.show()
```

**Discussion prompt:**

What changed when `Nk` increased from 30 to 100? Why does the detection limit become a bigger problem?

---

# Wrap-up questions

1. In this model, what does `p` represent?
2. Why does a defective nucleotide stop the reaction?
3. Why is stopping at the 10th `T` less likely than stopping at the 1st `T` for many values of `p`?
4. Why is `p = 0` bad for detecting later positions?
5. Why is `p = 1` bad for detecting any positions?
6. What does a histogram show us that a formula alone does not?
7. Why might long DNA molecules be harder to sequence clearly using this simplified model?

---

# Instructor Answer Key Summary

## Questions 1–3

As `p` increases, the reaction is more likely to continue past early `T` positions. The probability of stopping at later `T` positions becomes larger. For the 10th `T`, `p = 0.95` has the highest probability among `0.5`, `0.7`, and `0.95`. For fixed `p`, probabilities generally get small as `k` becomes large, making far-away positions harder to detect.

## Questions 4–6

The expected number of experiments is the inverse of the probability of stopping at the kth `T`. For `k = 1`, lower `p` is best because defective nucleotides cause immediate stopping. For larger `k`, too low a `p` causes early stopping, while too high a `p` makes stopping rare. The optimal value is:

```text
p = (k - 1) / k
```

The plotted optimal points should lie at the minima of the expected-experiment curves.

## Questions 7–9

For `Nk = 30`, `p = 0.9` gives better representation of later fragments than `p = 0.8`. The optimal `p = 29/30`, or about `0.967`, spreads fragments more evenly over the region and produces many read-through outcomes. With a detection threshold of `D = 0.01`, a fragment length must appear at least 100 times in 10,000 experiments. As DNA gets longer, each fragment length may become less frequent, causing more lengths to fall below the detection limit.

---

# Clarification Points

Don't confuse `k` with the physical DNA position. In this lab, `k` means the kth occurrence of the target base, not necessarily the kth base in the whole DNA sequence.

You may also confuse `p` with the probability of stopping. In this lab, `p` is the probability of choosing a good nucleotide. The probability of stopping is `1 - p`.

The simulation is useful because it connects a formula to something you can see. Each individual outcome is random, but the histogram from many outcomes has a predictable shape.

