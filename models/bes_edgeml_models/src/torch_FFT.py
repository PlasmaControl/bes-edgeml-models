import torch


def FFT(x: torch.Tensor):
    """A vectorized, non-recursive version of the Cooley-Tukey FFT"""

    N = torch.Tensor([x.shape[0]])
    pi = torch.Tensor([3.141592653589793])

    # N_min here is equivalent to the stopping condition above,
    # and should be a power of 2
    N_min = torch.min(torch.tensor([N, 32]))

    # Perform an O[N^2] DFT on all length-N_min sub-problems at once
    n = torch.arange(N_min)
    k = n[:, None]
    M = torch.exp(-2j * pi * n * k / N_min)
    x = torch.complex(x, torch.zeros_like(x))
    X = torch.matmul(M, x.reshape((N_min.to(torch.int32).item(), -1)))

    # build-up each level of the recursive calculation all at once
    while X.shape[1] < N:
        X_even = X[:, :X.shape[1] // 2]
        X_odd = X[:, X.shape[1] // 2:]
        factor = torch.exp(-1j * pi * torch.arange(X.shape[0]) / X.shape[0])[:, None]
        X = torch.vstack([X_even + factor * X_odd, X_even - factor * X_odd])

    return X.ravel()


def rFFT(x: torch.Tensor, dim: int = 2):

    t = x.transpose(dim, -1)
    s = t.shape
    t = t.reshape(-1, x.shape[dim])

    z = []
    for i in t:
        z.append(FFT(i))

    fft = torch.tensor(z)
    rfft = fft.reshape(s).transpose(dim, -1)

    return rfft



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = torch.randn((64, 1, 64, 8, 8))
    # y = torch.sin(x) + torch.sin(2*x) + torch.sin(3 * x)
    my_fft = rFFT(x, dim=2)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(x.squeeze())
    ax2.plot(my_fft)
    plt.show()
