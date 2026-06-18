# MAT/BIS 107 Lab 8 Lesson Plan: Modeling Cancer with Markov Chains

## Instructor overview

This lesson converts the original MATLAB-based Lab 8 into a Python/Jupyter lesson. Students explore a 50-state Markov chain model of metastatic cancer progression based on a transition matrix from Newton et al. The lesson includes interpreting a transition matrix, plotting transition probabilities from lung cancer, simulating cancer progression paths, and estimating mean first-passage times.

**Estimated time:** one 90–120 minute lab period.

**Files needed:** `cancer_model.csv` in the same directory as the notebook. The original file was provided as `cancer_model.xls`; for a lightweight Jupyter/Python workflow, this lesson uses a CSV version of the same transition matrix.

**Python packages used:** `pandas`, `numpy`, and `matplotlib.pyplot`.

## Learning goals

By the end of this lesson, students should be able to:

1. Interpret rows and columns of a transition matrix.
2. Connect a transition matrix to a biological movement model.
3. Simulate trajectories from a Markov chain.
4. Estimate first-passage times by simulation.
5. Compute mean first-passage times by solving a linear system.
6. Explain why model output should not be over-interpreted as a clinical prediction for an individual patient.

## Background for instructors

**Metastasis** is the spread of cancer cells from a primary site to other parts of the body. A Markov model treats each body site as a state and each transition as a probabilistic move from one site to another. This is a simplified mathematical model: it does not model cell biology directly, nor does it include treatment, patient-specific factors, time in days, or disease severity.

In the original study, the transition matrix has 50 body sites. In this lab, row 23 corresponds to lung as the current state. Since Python uses zero-based indexing, row 23 in MATLAB becomes row index 22 in Python. This indexing difference is one of the most important practical caveats for students.

A **first-passage time** is the number of steps until the chain first reaches a target state. A **mean first-passage time** averages this quantity across possible random trajectories. It can be estimated by repeated simulations or computed more directly using linear algebra.

## Important caveats

- This is an educational modeling exercise, not medical advice or a patient-level prediction tool.
- The model is based on autopsy-derived observations and estimated transition probabilities.
- A transition step is not a fixed number of days or months.
- Python indexing starts at 0; the lab's site numbers start at 1.
- Simulated trajectories are random. Use a random seed for reproducibility.
- Some columns are all zeros, meaning those sites were not observed as destinations in this model.

## Site index reference

The lung is site 23, regional lymph nodes are site 24, heart is site 17, and pancreas is site 28. In Python, subtract 1 to get the row/column index.

## Student notebook content

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

### Helper functions

```python
site_names = [
    "Adrenal", "Anus", "Appendix", "Bile Duct", "Bladder", "Bone", "Brain",
    "Branchial Cyst", "Breast", "Cervix", "Colon", "Diaphragm", "Duodenum",
    "Esophagus", "Eye", "Gallbladder", "Heart", "Kidney", "Large Intestine",
    "Larynx", "Lip", "Liver", "Lung", "Lymph Nodes (reg)", "Lymph Nodes (dist)",
    "Omentum", "Ovaries", "Pancreas", "Penis", "Pericardium", "Peritoneum",
    "Pharynx", "Pleura", "Prostate", "Rectum", "Retroperitoneum", "Salivary",
    "Skeletal Muscle", "Skin", "Small Intestine", "Spleen", "Stomach", "Testes",
    "Thyroid", "Tongue", "Tonsil", "Unknown", "Uterus", "Vagina", "Vulva"
]


def load_transition_matrix(filename="cancer_model.csv"):
    data = pd.read_csv(filename)
    P = data.to_numpy(dtype=float)
    # CSV conversion can introduce tiny floating-point row-sum errors.
    # Normalize each row so np.random.choice accepts it as probabilities.
    row_sums = P.sum(axis=1, keepdims=True)
    P = P / row_sums
    return P, list(data.columns)


def simulate_markov_chain(P, initial_site_number, steps, rng=None):
    """Simulate a Markov chain using 1-based site numbers for input and output."""
    if rng is None:
        rng = np.random.default_rng()
    states = np.zeros(steps, dtype=int)
    states[0] = initial_site_number - 1
    for i in range(1, steps):
        current = states[i - 1]
        states[i] = rng.choice(np.arange(P.shape[0]), p=P[current])
    return states + 1


def mean_first_passage_time(P, target_site_number):
    """Return vector of mean first-passage times to a target site from all starting sites."""
    target = target_site_number - 1
    Q = P.copy()
    Q[target, :] = 0
    Q[:, target] = 0
    return np.linalg.solve(np.eye(P.shape[0]) - Q, np.ones(P.shape[0]))


def first_passage_time(chain, target_site_number):
    hits = np.where(chain == target_site_number)[0]
    if len(hits) == 0:
        return np.nan
    return hits[0]
```

### Questions 1–2: Loading and checking the matrix

**Question 1.** Some sites were not observed as destinations for metastatic cancer in this model. What should the transition probability be for moving into one of those sites? What should those columns look like?

Write your answer in a markdown cell or as a Python comment.

**Question 2.** Load the transition matrix. Display one column corresponding to a site not connected in the graph. Confirm that it agrees with your answer to Question 1.

```python
P, columns = load_transition_matrix("cancer_model.csv")
print(P.shape)

# Site 2 is Anus. In Python, column index 1 corresponds to site 2.
print(P[:, 1])
print("Column sum:", P[:, 1].sum())
```

### Questions 3–4: Lung cancer transitions and limiting behavior

**Question 3.** Plot the probability distribution for transitions from the lung. Which site is most likely?

```python
lung_index = 23 - 1
lung_transitions = P[lung_index, :]

plt.figure(figsize=(12, 4))
plt.bar(np.arange(1, 51), lung_transitions)
plt.xlabel("Site number")
plt.ylabel("Transition probability")
plt.title("Transition probabilities from lung to other sites")
plt.xticks(np.arange(1, 51, 2))
plt.show()

most_likely_site = np.argmax(lung_transitions) + 1
print("Most likely site number:", most_likely_site)
print("Most likely site name:", site_names[most_likely_site - 1])
print("Probability:", lung_transitions[most_likely_site - 1])
```

**Question 4.** Compute the limiting distribution of metastasis if the primary site is the lungs. Use a large matrix power and start at the lung.

```python
initial = np.zeros(50)
initial[lung_index] = 1
limit_from_lung = initial @ np.linalg.matrix_power(P, 100)

for site_number, probability in enumerate(limit_from_lung, start=1):
    if probability > 0.001:
        print(f"{site_number:2d} {site_names[site_number - 1]:20s} {probability:.4f}")
```

### Questions 5–7: Simulation and first-passage time

**Question 5.** Generate some trajectories of cancer progression from the lung. Does the site from Question 3 appear at high frequency?

```python
rng = np.random.default_rng(107)
chain = simulate_markov_chain(P, initial_site_number=23, steps=100, rng=rng)
print(chain[:20])

counts = np.bincount(chain, minlength=51)[1:]
print("Count for regional lymph nodes, site 24:", counts[24 - 1])

plt.figure(figsize=(12, 4))
plt.bar(np.arange(1, 51), counts)
plt.xlabel("Site number")
plt.ylabel("Count in simulated chain")
plt.title("Sites visited in one simulated chain from lung")
plt.show()
```

**Question 6.** Generate 1000 simulations starting from the lungs with length 1000. Estimate the mean first-passage time to the heart, pancreas, and the most likely site from Question 3. Is the most likely site's mean first-passage time lower?

```python
rng = np.random.default_rng(107)
n_sims = 1000
chain_length = 1000

fp_heart = []
fp_pancreas = []
fp_lymph = []

for i in range(n_sims):
    chain = simulate_markov_chain(P, initial_site_number=23, steps=chain_length, rng=rng)
    fp_heart.append(first_passage_time(chain, 17))
    fp_pancreas.append(first_passage_time(chain, 28))
    fp_lymph.append(first_passage_time(chain, 24))

print("Mean first-passage time to heart:", np.nanmean(fp_heart))
print("Mean first-passage time to pancreas:", np.nanmean(fp_pancreas))
print("Mean first-passage time to regional lymph nodes:", np.nanmean(fp_lymph))
```

**Question 7.** Use the linear algebra approach to determine the mean first-passage times from the lungs to the pancreas and to the site from Question 3. How do the results compare to the simulations?

```python
T_heart = mean_first_passage_time(P, 17)
T_pancreas = mean_first_passage_time(P, 28)
T_lymph = mean_first_passage_time(P, 24)

print("Heart from lung:", T_heart[23 - 1])
print("Pancreas from lung:", T_pancreas[23 - 1])
print("Regional lymph nodes from lung:", T_lymph[23 - 1])
```

## Instructor Answer Key Summary

- Q1: The probability of transitioning into a site that was not observed as a colonized destination should be 0; the corresponding column should be all zeros.
- Q2: Column 2, Anus, is all zeros in the provided matrix.
- Q3: Site 24, regional lymph nodes, is the most likely transition from the lung.
- Q4: The limiting distribution from the lung has its largest probability at site 24. Other high-probability sites include site 25, site 22, site 23, and site 1 depending on rounding.
- Q5: In a typical 100-step simulation, regional lymph nodes should appear frequently, though counts vary by random seed.
- Q6: Simulation estimates vary. Expected approximate values are heart about `36`, pancreas about `26`, and regional lymph nodes about `5.5`.
- Q7: Linear algebra values should be close to `36.05` for heart, `26.63` for pancreas, and `5.62` for regional lymph nodes. These should agree with sufficiently many simulations.
