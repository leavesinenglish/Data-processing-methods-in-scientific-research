import numpy as num
from math import sin
from numpy.random import uniform
import matplotlib.pyplot as mpl


def Fourier(f: num.ndarray) -> num.ndarray:
    N = f.shape[0]
    n = num.arange(N)
    k = n.reshape((N, 1))
    M = num.exp(-2j * num.pi * k * n / N)
    return num.dot(M, f)


def Fourier_inverse(F: num.ndarray) -> num.ndarray:
    N = F.shape[0]
    n = num.arange(N)
    k = n.reshape((N, 1))
    M = num.exp(2j * num.pi * k * n / N)
    return num.dot(M, F)


def main():
    sinus = num.array([sin(2 * num.pi * i / 100) for i in range(501)])
    sum_sin = num.array([sin(2 * num.pi * i / 100) + 6 * sin((25 * num.pi * i / 100)) for i in range(501)])
    sin_noise = num.array([sin(2 * num.pi * i / 10) + 2 * (num.random.rand() - 0.5) for i in range(501)])
    meandr = num.array([1 if i <= 250 else -1 for i in range(501)])
    sum_sin_noise = num.array(
        [sin(2 * num.pi * i / 100) + 6 * sin(5 * num.pi * i / 100) + 2 * (num.random.rand()) for i in range(501)])

    mpl.figure()
    mpl.subplot(3, 3, 1)
    mpl.plot(sinus)

    mpl.subplot(3, 3, 2)
    mpl.plot(sum_sin)

    mpl.subplot(3, 3, 3)
    mpl.plot(meandr)

    mpl.subplot(3, 3, 4)
    mpl.plot(abs(Fourier(sinus)))
    # mpl.plot(num.fft.fft(sinus), "--")

    mpl.subplot(3, 3, 5)
    mpl.plot(abs(Fourier(sum_sin)))
    # mpl.plot(num.fft.fft(sum_sin), "--")

    mpl.subplot(3, 3, 6)
    mpl.plot(abs(Fourier(meandr)))
    # mpl.plot(num.fft.fft(meandr), "--")

    mpl.subplot(3,3,7)
    mpl.plot(sin_noise)

    mpl.subplot(3, 3, 8)
    mpl.plot(abs(Fourier(sin_noise)))

    mpl.show()


main()
