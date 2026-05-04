import numpy as np
from bam import formula, recall_backward, recall_forward
from metrics import print_crosstalk
from parts import run_partC, run_partD

# =====================================
# Part A : Checking recall is possible in both directions
# =====================================
X = np.array([
    [-1,  1,  1,  1, -1],
    [-1, -1, -1, -1,  1],
    [-1, -1, -1,  1,  1]
], dtype=np.int8)

Y = np.array([
    [ 1,  1, -1,  1],
    [ 1, -1, -1, -1],
    [-1, -1,  1,  1]
], dtype=np.int8)

# Train (Part A)
W = formula(X, Y)
print("Weight matrix W (W = A^T B):\n", W, "\n")

# Part A: one-step recall in both directions
print("Part A: Recalling Y from X (one step):\n")
for x, y in zip(X, Y):
    y_pred = recall_forward(W, x)
    print("X =", x, "\nTarget Y =", y, "\nPredicted Y =", y_pred,
          "\nMatch:", np.all(y_pred == y), "\n")

print("Part A: Recalling X from Y (one step):\n")
for x, y in zip(X, Y):
    x_pred = recall_backward(W, y)
    print("Y =", y, "\nTarget X =", x, "\nPredicted X =", x_pred,
          "\nMatch:", np.all(x_pred == x), "\n")

# =====================================
# Part B: calculating Crosstalk on A
# =====================================
print_crosstalk(X, title="Part B: Crosstalk on A (dot products, base 3 pairs)")

# =====================================
# Part C: Adding one given pair and 3 more random pairs one by one and checking crosstalk after each addition
# =====================================
extra_pairs = [
    (np.array([ 1, 1, 1, 1, 1]), np.array([-1, 1, 1, -1])),  # required pair first
    (np.array([ 1,-1, 1,-1, 1]), np.array([ 1, 1, 1,-1])),
    (np.array([-1, 1,-1, 1,-1]), np.array([-1, 1,-1,  1])),
    (np.array([ 1, 1,-1,-1, 1]), np.array([ 1,-1, 1,  1])),
]
run_partC(X, Y, extra_pairs, print_W=False)

# =====================================
# Part D: 20-run mutation and error correcting  table 
# =====================================
run_partD(X, Y, trials=20, mutate_ratio=0.20, seed=42)

if __name__ == "__main__":
    pass


