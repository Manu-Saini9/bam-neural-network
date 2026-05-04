import numpy as np

# Utilities 
def hamming(u, v):
    u = np.array(u).ravel()
    v = np.array(v).ravel()
    return int(np.sum(u != v))

#  Part B: Calculating Crosstalk using pairwise dot products among A vectors 
def print_crosstalk(X, title="Part B: Crosstalk on A"):
    """
   per lecture slides:
    Builds C where C[i,j] = dot(X_i, X_j) for i != j, 0 on diagonal.
    Prints each row with signed sum( that is sum of all the values with sign ) and |abs| sum, plus global totals.

    """

    X = np.array(X, dtype=int)
    p = X.shape[0]
    C = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(p):
            if i == j:
                C[i, j] = 0
            else:
                C[i, j] = int(np.dot(X[i], X[j]))

    print(title)
 
    print("i\\j  " + "".join(f"{j+1:6d}" for j in range(p)) + f"{'   sum':>10}{'   sum|.|':>12}")
    global_signed = 0
    global_abs = 0
    for i in range(p):
        row = C[i]
        row_sum = int(np.sum(row))
        row_abs = int(np.sum(np.abs(row)))
        global_signed += row_sum
        global_abs += row_abs
        print(f"{i+1:3d} |" + "".join(f"{v:6d}" for v in row) + f"{row_sum:10d}{row_abs:12d}")
    print(f"Global (signed) = {global_signed}")
    print(f"Global (abs)    = {global_abs}\n")
    return C
