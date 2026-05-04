# Bidirectional Associative Memory (BAM) from Scratch

> A **heteroassociative neural memory** that stores pattern pairs and recalls them in **both directions** — with crosstalk analysis and noise-correction experiments.

---

## Overview

Bidirectional Associative Memory (BAM) is a two-layer recurrent neural network that learns to associate input patterns with output patterns — and can recall in **either direction**. Given X, it recalls Y. Given Y, it recalls X.

This project implements BAM entirely in **NumPy** from first principles, covering the learning rule, iterative recall, crosstalk analysis, and error-correction evaluation.

Built for **COSC 4P80** (Neural Networks) at Brock University.

---

## How BAM Works

```
Layer A (X)  ←──────W──────→  Layer B (Y)
   [5 neurons]               [4 neurons]

Forward:  Ŷ = sign(Wᵀ · X)
Backward: X̂ = sign(W  · Y)

Training: W = Xᵀ · Y   (outer product / Hebbian rule)
```

Recall is **iterative** — the network alternates between layers until it converges to a fixed point (stable state) or detects a cycle.

---

## Experiments

### Part A — Bidirectional Recall
Stored 3 pattern pairs (X ∈ {-1,+1}⁵, Y ∈ {-1,+1}⁴) and verified one-step recall in both directions.

```
X → Ŷ  (forward):  all 3 pairs recalled perfectly 
Y → X̂  (backward): all 3 pairs recalled perfectly 
```

### Part B — Crosstalk Analysis
Computed the **crosstalk matrix** C where `C[i,j] = dot(Xᵢ, Xⱼ)` for all stored pattern pairs. High crosstalk values indicate interference between patterns — a predictor of recall failure as capacity increases.

```
Output: pairwise dot products, row sums (signed & absolute), global totals
```

### Part C — Incremental Capacity Testing
Added 1 required pair + 3 random pairs one at a time. After each addition:
- Retrained the weight matrix
- Tested one-step recall for all stored pairs (A→B and B→A)
- Recomputed the crosstalk matrix

Tracks how recall accuracy degrades as storage capacity approaches its limit.

### Part D — Error Correction (20-Run Table)
Simulated noisy inputs by **flipping ~20% of bits** in stored patterns, then ran iterative BAM recall to attempt correction. Results reported as a full table:

| Run | Original (O) | Mutated (M) | Hdist(O,M) | Corrected (C) | Hdist(O,C) |
|---|---|---|---|---|---|
| 1 | [-1 1 1 1 -1] | [-1 -1 1 1 -1] | 1 | [-1 1 1 1 -1] | 0 |
| … | … | … | … | … | … |

Hdist(O,C) = 0 means perfect recovery. This tests BAM's **noise tolerance** and **basin of attraction** for each stored pattern.

## Sample Output

### Weight Matrix W (computed as W = Aᵀ · B)
```
[[-1  1  1 -1]
 [ 1  3 -1  1]
 [ 1  3 -1  1]
 [-1  1  1  3]
 [-1 -3  1 -1]]
```

### Part A — Bidirectional Recall (one step)

**Forward X → Ŷ:**
```
X = [-1  1  1  1 -1]  →  Predicted Y = [ 1  1 -1  1]  Match: True  
X = [-1 -1 -1 -1  1]  →  Predicted Y = [-1 -1  1 -1]  Match: False 
X = [-1 -1 -1  1  1]  →  Predicted Y = [-1 -1  1  1]  Match: True  
```

**Backward Y → X̂:**
```
Y = [ 1  1 -1  1]   →  Predicted X = [-1  1  1  1 -1]  Match: True  
Y = [ 1 -1 -1 -1]   →  Predicted X = [-1 -1 -1 -1  1]  Match: True  
Y = [-1 -1  1  1]   →  Predicted X = [ 1 -1 -1  1  1]  Match: False 
```

### Part B — Crosstalk Matrix (base 3 pairs)
```
i\j   1    2    3   | sum  |sum|
 1  [ 0   -3   -1]  |  -4     4
 2  [-3    0    3]  |   0     6
 3  [-1    3    0]  |   2     4

Global (signed) = -2  |  Global (abs) = 14
```

### Part C — After Adding Pair #4 (4 total pairs)
```
One-step recall (A → B):  4/4 correct 
One-step recall (B → A):  3/4 correct
```
Crosstalk increases as more pairs are stored — demonstrating capacity limits.

### Part D — Error Correction Table (20 runs, ~20% bit flips)

| Run | Original (O) | Mutated (M) | Hdist(O,M) | Corrected (C) | Hdist(O,C) |
|---|---|---|---|---|---|
| 1 | [-1 1 1 1 -1] | [-1 1 -1 1 -1] | 1 | [-1 1 1 1 -1] | 0 |
| 17 | [-1 1 1 1 -1] | [-1 1 -1 1 -1] | 1 | [-1 1 1 1 -1] | 0 |
| 20 | [-1 1 1 1 -1] | [-1 1 1 -1 -1] | 1 | [-1 1 1 1 -1] | 0 |
| … | … | … | … | … | … |

> Full output available in `output.pdf`. Written report in `Report.pdf`.

---

```
bam.py
├── activate(v)           # Bipolar sign activation: +1 if v≥0, else -1
├── formula(X, Y)         # BAM learning rule: W = Xᵀ · Y
├── recall_forward(W, x)  # One-step X → Y
├── recall_backward(W, y) # One-step Y → X
├── recall(W, x, d)       # Iterative recall until convergence
└── correct_from_A(W, A)  # Iterative error correction from A side

metrics.py
├── hamming(u, v)         # Hamming distance between two bipolar vectors
└── print_crosstalk(X)    # Full crosstalk matrix with row/global sums

parts.py
├── run_partC(...)        # Incremental pair addition + recall + crosstalk
└── run_partD(...)        # 20-run mutation & error-correction table
```

---

## Project Structure

```
Assign_1/
├── main.py        # Runs all 4 parts (A, B, C, D)
├── bam.py         # BAM core: learning rule, recall, error correction
├── metrics.py     # Hamming distance, crosstalk matrix
├── parts.py       # Part C and D experiment runners
├── output.pdf     # Full program output
└── Report.pdf     # Written analysis and discussion
```

---

## Getting Started

### Prerequisites
```bash
pip install numpy
```

### Run All Parts
```bash
python main.py
```

This runs all four experiments sequentially and prints:
- Weight matrix W
- Part A: bidirectional recall verification
- Part B: crosstalk matrix
- Part C: incremental capacity table
- Part D: 20-run error-correction table

---

## Key Concepts Demonstrated

- **Hebbian learning** — weight matrix as sum of outer products
- **Heteroassociative memory** — mapping between two different pattern spaces
- **Bidirectional recall** — fixed-point iteration across two layers
- **Crosstalk / interference** — measuring pattern orthogonality and capacity limits
- **Noise tolerance** — Hamming-distance-based error correction evaluation
- **Basin of attraction** — how far a corrupted pattern can be from a stored one and still be recovered

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-from--scratch-lightblue?style=for-the-badge&logo=numpy&logoColor=white)

> No ML frameworks. Every recall step, activation, and convergence check is hand-coded.

---
