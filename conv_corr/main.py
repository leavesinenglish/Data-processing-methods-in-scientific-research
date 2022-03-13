import numpy as np
import matplotlib.pyplot as mpl


def conv1(f: np.ndarray, g: np.ndarray):
    return np.fft.ifft(np.fft.fft(f, f.size + g.size - 1) * np.fft.fft(g, f.size + g.size - 1), f.size + g.size - 1)


def corr1(f: np.ndarray, g: np.ndarray):
    return np.fft.ifft(np.fft.fft(f, f.size + g.size - 1) * np.fft.fft(g, f.size + g.size - 1).conj(),
                       f.size + g.size - 1)


def conv2(f: np.ndarray, g: np.ndarray):
    size = f.size + g.size - 1
    res = np.empty(size)
    for n in range(size):
        sum = 0
        for m in range(f.size):
            if min(f.size, g.size) > n - m >= 0:
                sum += f[m] * g[n - m]
        res[n] = sum
    return res


def corr2(f: np.ndarray, g: np.ndarray):
    size = f.size + g.size - 1
    f = np.concatenate((np.zeros(size - f.size), f))
    g = np.concatenate((g, np.zeros(size - g.size)))
    res = np.empty(size)
    for n in range(size):
        sum = 0
        for m in range(f.size):
            if min(f.size, g.size) > m - n >= 0:
                sum += f[m] * g[m - n]
        res[n] = sum
    return res



def main():
    graph1 = np.array([1 if i <= 250+30 else 0 for i in range(501)])
    graph2 = np.array([1 if i <= 250 else 0 for i in range(501)])

    #graph1 = np.array([np.sin((np.pi * i + 30) / 100) for i in range(501)])
    #graph2 = np.array([np.sin(np.pi * i / 100) for i in range(501)])

    # graph1 = np.array([np.random.rand() for i in range(101)])
    # graph2 = np.array([np.random.rand() for i in range(101)])

    # graph1 = np.array([100 - 2 * i if i > 50 else -100 - 2 * i for i in range(101)] * 3)
    # graph2 = np.array([100 - 2 * i if i > 70 else -100 - 2 * i for i in range(101)] * 3)

    mpl.figure()
    mpl.subplot(3, 1, 1)
    mpl.plot(graph1)
    mpl.plot(graph2, '--')
    mpl.ylabel('f(t)')

    mpl.subplot(3, 1, 2)
    mpl.plot(np.real(conv2(graph1, graph2)))
    mpl.plot(np.real(np.convolve(graph1, graph2)), '--')
    mpl.ylabel('conv')

    mpl.subplot(3, 1, 3)
    mpl.plot(np.real(corr2(graph1, graph2)))
    mpl.plot(np.real(np.correlate(graph1, graph2, 'full')), '--')
    mpl.ylabel('corr')

    mpl.show()


main()