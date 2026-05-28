import numpy as np


def local_cost_matrix(x, y):
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if y.ndim == 1:
        y = y[:, np.newaxis]
    diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]
    C = np.sqrt(np.sum(diff ** 2, axis=-1))
    return C


def dtw(x, y, band_ratio=None):
    C = local_cost_matrix(x, y)
    N, M = C.shape
    D = np.full((N, M), np.inf)
    D[0, 0] = C[0, 0]

    if band_ratio is None:
        band = max(N, M)
    else:
        band = max(int(band_ratio * max(N, M)), abs(N - M) + 1)

    for m in range(1, min(M, band + 1)):
        D[0, m] = D[0, m - 1] + C[0, m]
    for n in range(1, min(N, band + 1)):
        D[n, 0] = D[n - 1, 0] + C[n, 0]

    for n in range(1, N):
        m_lo = max(1, n - band)
        m_hi = min(M, n + band + 1)
        for m in range(m_lo, m_hi):
            D[n, m] = min(D[n - 1, m - 1],
                          D[n - 1, m],
                          D[n, m - 1]) + C[n, m]

    dtw_distance = D[N - 1, M - 1]

    path = [(N - 1, M - 1)]
    n, m = N - 1, M - 1
    while n > 0 or m > 0:
        if n == 0:
            m -= 1
        elif m == 0:
            n -= 1
        else:
            candidates = {
                (n - 1, m - 1): D[n - 1, m - 1],
                (n - 1, m): D[n - 1, m],
                (n, m - 1): D[n, m - 1],
            }
            n, m = min(candidates, key=candidates.get)
        path.append((n, m))

    path.reverse()
    return dtw_distance, D, path
