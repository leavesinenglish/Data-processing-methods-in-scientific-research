import numpy as np
import matplotlib.pyplot as mpl
import scipy.signal as scp
from fractions import Fraction
from decimal import Decimal


################################FILTERS################################################
def ampl_l(signal, A, B):
    return np.array([0 if A <= i <= B else 1 for i in range(len(signal))])


def fft_ampl_window(signal, N, A, B):
    a, b = np.array_split(np.fft.ifft(ampl_l(signal, A, B)), 2)
    sinc = np.concatenate([b, a])
    sinc = sinc * np.array(
        [1 if len(signal) / 2 - N / 2 <= i <= len(signal) / 2 + N / 2 else 0 for i in range(len(signal))])
    a, b = np.array_split(sinc, 2)
    return np.concatenate([b, a])


def filter_l(signal, N, A, B):
    h = np.fft.fft(fft_ampl_window(signal, N, A, B))
    return np.real(np.fft.ifft(h * np.fft.fft(signal)))


#############################RESAMPLING##################################################
def resampling(signal, new_size):
    K = Fraction(new_size, signal.size).numerator
    M = Fraction(new_size, signal.size).denominator
    interpolated = signal
    if K != 1:
        interpolated = np.zeros(signal.size * K)
        for i in range(signal.size * K):
            if i % K == 0:
                interpolated[i] = signal[i // K]
        interpolated = K * filter_l(interpolated, interpolated.size, interpolated.size / (2 * K),
                                    interpolated.size * (1 - 1 / (2 * K)))
    decimated = interpolated
    if M != 1:
        decimated = np.zeros(int(interpolated.size / M))
        for i in range(interpolated.size):
            if i % M == 0:
                decimated[i // M] = interpolated[i]
    return decimated


#########################RESAMPLING#WITH#FOURIER###########################################
def fourier_resampling(signal, m):
    n = signal.size
    fourier = np.fft.fft(signal)
    if m > n:
        result_fft = np.concatenate([fourier[:int(fourier.size / 2)], np.zeros(m - n), fourier[int(fourier.size / 2):]])
    elif m < n:
        result_fft = np.concatenate([fourier[:int(fourier.size / 2) - n + m], fourier[int(fourier.size / 2) - n + m:]])
    return np.fft.ifft(result_fft*m/n)


def test():
    size = 1000
    new_size = 2000
    # signal = np.sin(x)
    # signal = np.array([5*np.sin(np.pi * i/500) + 0.5*(np.random.rand() - 0.5) for i in range(50)])
    signal = np.array([np.sin(np.pi * i / 5) + 3 * np.sin(np.pi * i / 100) + np.random.rand() for i in range(1000)])

    fig = mpl.figure()
    mpl.rcParams["figure.dpi"] = 300

    mpl.subplot(5, 1, 1)
    mpl.plot(signal)  # , '.')
    mpl.ylabel('signal')

    mpl.subplot(5, 1, 2)
    mpl.plot(resampling(signal, new_size))  # , '.')
    mpl.ylabel('resampled')

    mpl.subplot(5, 1, 3)
    mpl.plot(np.abs(np.fft.fft(signal)))
    mpl.ylabel('spectrum')

    mpl.subplot(5, 1, 4)
    mpl.plot(np.abs(np.fft.fft(resampling(signal, new_size))))
    mpl.ylabel('resampled spectrum')

    mpl.subplot(5, 1, 5)
    mpl.plot(fourier_resampling(signal, new_size))
    mpl.ylabel('resampled with fourier')

    mpl.show()


test()
