import torch


def sensingH(x, gdmd):
    [b, L, M, N] = x.shape
    S = gdmd.shape[0]
    y = torch.zeros(b, S, M, N + L - 1).to(x.device)

    y_noshift = x[:, :, None] * gdmd
    for k in range(L):
        y[..., k:N + k] += + y_noshift[:, k]

    return y


def sensingHt(y, gdmd):
    [_, M, N] = gdmd.shape
    L = y.shape[-1] - N + 1

    y1 = torch.zeros(y.shape[0], L, M, N).to(y.device)
    for k in range(L):
        y1[:, k] += torch.sum(y[..., k:N + k] * gdmd, dim=1)

    return y1


def computeP(f, L):
    [S, M, N] = f.shape
    P = torch.zeros(S, S, M, N + L - 1).to(f.device)
    for t in range(S):
        for s in range(t, S):
            # This loop corresponds to the formula: sum_{l=1}^L S^Z_{l-1}(f_t).*S^Z_{l-1}(f_s)
            for l in range(L):
                P[t, s, :, l:N + l] = P[t, s, :, l:N + l] + f[t] * f[s]

            # Here we use the fact that the matrix is "self-adjoint", so we do not need
            # to compute lower diagonal images
            P[s, t] = P[t, s]

    return P


def computeQ(f, L):
    [_, M, N] = f.shape
    Q = torch.zeros(L, L, M, N).to(f.device)
    # we first evaluate the first row of Q's, so k=1 below
    for l in range(L):
        Q[0, l, :, l:N] = torch.sum(f[..., :N - l] * f[..., l:N], dim=0)

    # We now compute remaining indices using the symmetry
    for k in range(L):
        for l in range(k, L):
            # This line uses the Teoplitz structure to get the upper
            # diagonal elements
            Q[k, l] = Q[0, l - k]
            # And this line the formula Q_{l,k}=S_{k-l}[Q_{k,l}],
            # to get the lower diagonal elements
            Q[l, k, :, :N - l + k] = Q[k, l, :, l - k:N]

    return Q


def IMVM(P, y):
    return torch.sum(P * y[:, None], dim=2)


def IMVMS(Q, d):
    # this is an image - matrix vector multiplication with shifts
    [L, _, M, N] = Q.shape
    y = torch.zeros(d.shape[0], L, M, N).to(Q.device)
    for k in range(L):
        for l in range(k + 1):
            y[:, k, :, :N + l - k] += Q[k, l, :, :N + l - k] * d[:, l, :, k - l:N]

        for l in range(k + 1, L):
            y[:, k, :, l - k:N] += Q[k, l, :, l - k:N] * d[:, l, :, :N - l + k]

    return y


def ComputePinv(P, rho):
    [S, _, M, NplusLminus1] = P.shape
    e = torch.ones(M, NplusLminus1)
    E = torch.zeros_like(P).to(P.device)

    for t in range(S):
        E[t, t] = e

    R = rho * E + P
    A = rho * E + P
    T = E

    # we first work the lower diagonal rows, just as in standard gaussian elimination
    for t in range(S):
        Rtt = R[t, t].clone()

        # this divides the t'th row with r_tt
        for s in range(S):
            R[t, s] = R[t, s] / Rtt
            T[t, s] = T[t, s] / Rtt

        # this withdraws a multiple of the t'th row from the remaining
        for u in range(t + 1, S):
            # this is the multiple in question
            Rut = R[u, t].clone()
            # and here comes the elimination step
            for s in range(S):
                R[u, s] = R[u, s] - Rut * R[t, s]
                T[u, s] = T[u, s] - Rut * T[t, s]

    # this piece does the final "upper triangular" elimination
    for s in reversed(range(1, S)):
        for t in reversed(range(s)):
            for s1 in range(S):
                T[t, s1] = T[t, s1] - R[t, s] * T[s, s1]

            R[t, s] = R[t, s] - R[t, s] * R[s, s]

    return T, A


def ComputeQinv(Q, rho):
    [L, _, M, N] = Q.shape
    e = torch.ones(M, N)
    E = torch.zeros_like(Q).to(Q.device)

    for t in range(L):
        E[t, t] = e

    R = rho * E + Q
    A = rho * E + Q
    T = E

    for t in range(L):
        Rtt = R[t, t].clone()

        # this divides the tth row with r_tt
        for s in range(L):
            R[t, s] = R[t, s] / Rtt
            T[t, s] = T[t, s] / Rtt

        # this withdraws a suitable multiple of the tth row from the remaining
        for t1 in range(t + 1, L):
            Rt1t = R[t1, t].clone()
            for s in range(L):
                R[t1, s, :, :N + t - t1] -= Rt1t[:, :N + t - t1] * R[t, s, :, t1 - t: N]
                T[t1, s, :, :N + t - t1] -= Rt1t[:, :N + t - t1] * T[t, s, :, t1 - t: N]

    # this piece does the final "upper triangular" elimination
    for s in reversed(range(1, L)):
        for t in reversed(range(s)):
            for s1 in range(L):
                T[t, s1, :, s - t: N] -= R[t, s, :, s - t: N] * T[s, s1, :, : N + t - s]

    return T, A
