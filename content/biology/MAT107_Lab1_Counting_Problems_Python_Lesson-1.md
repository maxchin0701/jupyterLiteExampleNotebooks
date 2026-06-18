# Lab 1: Counting Problems

> **Python version of the original MAT/BIS 107 MATLAB lab.**  
> This lesson uses basic Python packages only: `math`, `random`/`numpy`, `pandas` when reading CSV files, and `matplotlib.pyplot` for plots. The original MATLAB labs asked students to submit a single script; this version is designed for a Jupyter notebook, where students run each code cell and write short explanations in markdown cells.

**Instructor note.** The code cells are intentionally readable rather than highly optimized. For class use, encourage students to run each cell, inspect intermediate variables, and answer the written questions in their own words.


## Lesson goals
By the end of this lab, students should be able to use Python as a calculator, use variables, compute powers and combinations, and explain how counting rules apply to DNA sequences, codons, and simplified sequence alignments.

## Background: why counting matters in biology
DNA sequences are strings made from four nucleotide symbols: `A`, `C`, `G`, and `T`. Even short DNA strings create huge numbers of possibilities because every position multiplies the number of possible sequences. A 3-nucleotide sequence has `4 × 4 × 4 = 64` possible sequences. A 12-nucleotide sequence has many more.

Counting also appears when DNA is translated into proteins. DNA is read in 3-letter codons. There are 64 possible codons, but only 20 common amino acids, so the genetic code is **degenerate**: several codons can encode the same amino acid. This means that different DNA sequences can produce the same protein sequence.

Counting also appears in sequence alignment. When comparing two sequences, possible differences include substitutions, insertions, and deletions. Even a simplified model can produce many possible alignments.

### Caveats
This lab uses simplified biology. Real protein-coding sequences include stop codons, reading frames, regulatory context, strand direction, and more details. The alignment-counting problem here does not score alignments or enforce all biological constraints; it is a counting exercise that helps introduce combinatorics.

```python
import math

# Python can be used as a calculator.
a = 5
b = 20
c = math.sqrt(a) * b
print(c)

# Useful commands for this lab:
print(4 ** 3)         # powers
print(math.comb(5,2)) # combinations, read as "5 choose 2
```

## Student Questions and Code
The following sections are written as student-facing prompts. Students should run the relevant code cells, inspect the output, and write their own answers before looking at the instructor key.

## Part 1: Counting DNA sequences
For a DNA sequence of length `L`, each position has 4 possible nucleotides. Therefore, the number of possible sequences is:

`4^L`

### Question 1
How many unique DNA sequences can be constructed with a length of 12?

```python
L = 12
n_sequences = 4 ** L
print(n_sequences)
```

## Part 2: Counting possible proteins and codons
Suppose a protein contains one methionine, one alanine, one glycine, and one leucine.

Codon counts for this lab:

| Amino acid | Number of possible codons |
|---|---:|
| Methionine | 1 |
| Alanine | 4 |
| Glycine | 4 |
| Leucine | 6 |

### Question 2
If you do not know the order of the four amino acids, how many ways can these four amino acids be arranged into a protein sequence?

### Question 3
Proteins usually start with methionine. If methionine must be first, how many protein sequences are possible?

### Question 4
Using your answer from Question 3, how many DNA sequences could produce one of those protein sequences?

### Question 5
What percentage of all length-12 DNA sequences would produce this protein composition under this simplified model?

```python
# Q2: all four amino acids are distinct in this problem.
arrangements_any_order = math.factorial(4)
print("Q2 arrangements:", arrangements_any_order)

# Q3: methionine fixed in the first position; arrange the other 3 amino acids.
arrangements_with_m_first = math.factorial(3)
print("Q3 arrangements with methionine first:", arrangements_with_m_first)

# Q4: for each amino acid order, multiply codon choices.
codon_choices_per_order = 1 * 4 * 4 * 6
dna_sequences_for_protein = arrangements_with_m_first * codon_choices_per_order
print("Q4 DNA sequences:", dna_sequences_for_protein)

# Q5: percentage of all length-12 DNA sequences.
percent = 100 * dna_sequences_for_protein / n_sequences
print("Q5 percent:", percent)
```

## Part 3: Counting simplified alignments
Suppose there are two DNA sequences:

`x = (x1, x2, x3, x4, x5)` and `y = (y1, y2, y3, y4)`

We will count possible alignments under a simplified model in which differences are substitutions, insertions, or deletions. If exactly `s` substitutions occur, we choose `s` positions from `x` and `s` positions from `y`. The remaining unmatched positions are treated as insertions or deletions.

For this simplified lab:

`number for a given s = C(len(x), s) × C(len(y), s)`

Then we sum over all possible values of `s`.

### Question 6
How many possible alignments exist for sequences of length 5 and 4?

### Question 7
How many possible alignments exist for two sequences if both sequences have length 15?

```python
def simplified_alignment_count(m, n):
    # Count simplified alignments for sequence lengths m and n.
    total = 0
    terms = []
    for s in range(min(m, n) + 1):
        term = math.comb(m, s) * math.comb(n, s)
        terms.append((s, term))
        total += term
    return total, terms

for m, n in [(5, 4), (15, 15)]:
    total, terms = simplified_alignment_count(m, n)
    print(f"Lengths {m} and {n}: total simplified alignments = {total}")
    print("terms by number of substitutions:", terms)
    print()
```

## Reflection questions
1. Why does the number of possible DNA sequences grow so quickly as sequence length increases?
2. Why can multiple DNA sequences encode the same protein?
3. What simplifying assumptions did we make in the alignment-counting section?

## Instructor Answer Key Summary

**Question 1.** A length-12 DNA sequence has `4^12 = 16,777,216` possible sequences.

**Question 2.** Four distinct amino acids can be arranged in `4! = 24` ways.

**Question 3.** If methionine must be first, the remaining three amino acids can be arranged in `3! = 6` ways.

**Question 4.** Each methionine-alanine-glycine-leucine order has `1 × 4 × 4 × 6 = 96` possible codon choices, so the total is `6 × 96 = 576` DNA sequences.

**Question 5.** The percentage is `100 × 576 / 16,777,216 ≈ 0.00343%`.

**Question 6.** The simplified alignment count for lengths 5 and 4 is `126`.

**Question 7.** The simplified alignment count for two sequences of length 15 is `155,117,520`.

**Reflection.** DNA sequence counts grow exponentially because every additional position multiplies the total by 4. Multiple DNA sequences can encode the same protein because the genetic code is degenerate. The alignment-counting section ignores scoring, gap penalties, biological plausibility, and many details used in real sequence alignment.
