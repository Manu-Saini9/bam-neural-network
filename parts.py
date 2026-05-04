import numpy as np
from bam import activate, formula, recall_forward, recall_backward, correct_from_A
from metrics import print_crosstalk, hamming

# Part C: Adding given pair, and then adding 3 more pairs one by one; after each, retrain & report 
def run_partC(X_base, Y_base, extra_pairs, print_W=False):
    X_list = [np.asarray(r, dtype=int).copy() for r in X_base]
    Y_list = [np.asarray(r, dtype=int).copy() for r in Y_base]

    for step, (xa, ya) in enumerate(extra_pairs, start=1):
        X_list.append(np.asarray(xa, dtype=int))
        Y_list.append(np.asarray(ya, dtype=int))
        X = np.vstack(X_list)
        Y = np.vstack(Y_list)
        W = X.T @ Y

        print(f"After adding pair #{len(X_list)} (total pairs now = {len(X_list)})")
        if print_W:
            print("W =\n", W)

        # step X -> Y
        print("One-step recall (A -> B):")
        for x, y in zip(X, Y):
            yhat = activate(W.T @ x)
            print(f"A={x}  B̂={yhat}  Match={np.all(yhat == y)}")
        print()

        # step Y -> X
        print("One-step recall (B -> A):")
        for x, y in zip(X, Y):
            xhat = activate(W @ y)
            print(f"B={y}  Â={xhat}  Match={np.all(xhat == x)}")
        print()

        # Crosstalk among A after this addition
        print_crosstalk(X, title="Crosstalk on A (after this addition)")

# Part D: 20-run mutation table
def run_partD(X_base, Y_base, trials=20, mutate_ratio=0.20, seed=42):
    """
    Per lecture slides:
      - select stored A (Original O)
      -  flip about 20% of the bits -> Mutated M
      - iteratively correct from A side (A->B->A...) with bipolar activationfunction that forces every output value to be either +1 or -1
      - print the mutation and error correction table: Run, O, M, Hdist(O,M), Corrected C, Hdist(O,C)
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X_base, dtype=int)
    Y = np.asarray(Y_base, dtype=int)
    W = formula(X, Y)

    
    print("Part D: Error-Correction Table (20 runs)")
    print(f"{'Run':<4} {'Original (O)':<25} {'Mutated (M)':<25} {'Hdist(O,M)':<12} "
          f"{'Corrected (C)':<25} {'Hdist(O,C)':<12}")
    print("-" * 110)

    for run in range(1, trials + 1):
        # choose a stored A
        idx = int(rng.integers(0, X.shape[0]))
        O = X[idx]

        # flip ~20% of bits (at least 1)
        n = O.size
        k = max(1, int(round(mutate_ratio * n)))
        flip_idx = rng.choice(n, size=k, replace=False)
        M = O.copy()
        M[flip_idx] *= -1

        # iterative correction from A
        C, iters, status = correct_from_A(W, M, max_iters=50)

        # distances
        dOM = hamming(O, M)
        dOC = hamming(O, C)

        # print one clean row
        print(f"{run:<4} {str(O):<25} {str(M):<25} {dOM:<12} {str(C):<25} {dOC:<12}")
    print()
