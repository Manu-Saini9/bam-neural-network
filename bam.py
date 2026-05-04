import numpy as np

# Activation function 
def activate(v):
    return np.where(v >= 0, 1, -1)

# BAM learning rule (matrix form W = A^T B) 
def formula(X, Y):
    return X.T.dot(Y)

# Iterative recall until its get converged
def recall(W, x, d='out'):
    end_of_recall = False
    y_pred = None
    x_eval = None
    while not end_of_recall:
        # X->Y or Y->X step
        y_pred = activate(W.T.dot(x) if d == 'out' else W.dot(x))
        # update the driving side
        x_eval = activate(W.dot(y_pred) if d == 'out' else W.T.dot(y_pred))
        # convergence check
        x, end_of_recall = x_eval, np.all(np.equal(x, x_eval))
    # for d='out' this returns Y; for d='in' this returns X
    return y_pred


def recall_forward(W, x):  # X -> Y
    return activate(W.T.dot(x))

def recall_backward(W, y):   # Y -> X
    return activate(W.dot(y))

# Part D helper method that try to fix a noisy a pattern by running it through BAM
def correct_from_A(W, A_init, max_iters=50):

      # start with a copy of A so we don’t change the original
    A = np.array(A_init, dtype=int).copy()
    seen = set()
    for t in range(1, max_iters + 1):
        B = activate(W.T @ A)      # A -> B
        A_new = activate(W @ B)    # B -> A

        if np.array_equal(A_new, A):
            return A_new, t, "fixed-point"

        key = tuple(A_new.tolist())
        if key in seen:
            return A_new, t, "cycle"  # small cycle detected

        seen.add(key)
        A = A_new

    return A, max_iters, "no convergence"
