import numpy as np
import matplotlib.pyplot as mpl


def moving_average_filter(f: np.ndarray, N):
    filtered = np.zeros(f.size)
    for n in range(f.size):
        for k in range(N):
            filtered[n] += f[n - k]
    return filtered / N


def h_maf(N, signal):
    return np.array([1 / N if i <= N else 0 for i in range(signal.size)])


def test_maf():
    # signal = np.array([np.sin(np.pi * i / 100) for i in range(1000)])
    signal = np.array([np.sin(np.pi * i / 10) + 2 * (np.random.rand() - 0.5) for i in range(1000)])
    # signal = np.array([np.sin(np.pi * i / 100) + 3 * np.sin(6 * np.pi * i / 100) for i in range(1000)])

    fig = mpl.figure()
    fig.set_figheight(9)
    fig.set_figwidth(10)

    mpl.subplot(6, 1, 1)
    mpl.plot(signal)
    mpl.ylabel('Сигнал')

    mpl.subplot(6, 1, 2)
    mpl.plot(moving_average_filter(signal, 10))
    mpl.ylabel('Фильтр')

    mpl.subplot(6, 1, 3)
    mpl.plot(np.abs(np.fft.fft(h_maf(10, signal))))
    mpl.ylabel('АЧХ фильтра')

    mpl.subplot(6, 1, 4)
    mpl.plot(np.abs(np.fft.fft(moving_average_filter(signal, 10))))
    mpl.ylabel('АЧХ отфильтрованного\n'
               ' сигнала')

    mpl.subplot(6, 1, 5)
    mpl.plot(np.arctan(np.imag((np.fft.fft(h_maf(10, signal)))) / np.real(
        np.fft.fft(h_maf(10, signal)))))
    mpl.ylabel('ФЧХ фильтра')

    mpl.subplot(6, 1, 6)
    mpl.plot(np.abs(np.fft.fft(signal)))

    mpl.show()


########################################################################################################################

def ampl(signal):
    return np.array([1 if i <= len(signal) / 4 or i >= 3 * len(signal) / 4 else 0 for i in range(len(signal))])


def phase(signal):
    return np.array([4 * np.pi * i / len(signal) if i <= len(signal) / 4 else 4 * np.pi * (i - len(signal)) / len(
        signal) if 3 * len(signal) / 4 <= i <= len(signal) else 0 for i in range(len(signal))])


def fft_ampl_window(signal, N):
    a, b = np.split(np.fft.fft(ampl(signal)), 2)
    sinc = np.concatenate([b, a])
    return sinc * np.array([1 if len(signal) / 2 - N <= i <= len(signal) / 2 + N else 0 for i in range(len(signal))])


def filter(signal, N):
    h = np.fft.ifft(fft_ampl_window(signal, N))
    return np.real(np.fft.ifft(h * np.fft.fft(signal)))


def test_lf():
    # signal = np.array([np.sin(np.pi * i / 100) for i in range(1000)])
    # signal = np.array([np.sin(np.pi * i / 100) + 2 * (np.random.rand() - 0.5) for i in range(1000)])
    signal = np.array([np.sin(np.pi * i / 100) + 3 * np.sin(6 * np.pi * i / 100) for i in range(1000)])

    fig = mpl.figure()
    fig.set_figheight(9)
    fig.set_figwidth(10)

    mpl.subplot(6, 1, 1)
    mpl.plot(signal)
    mpl.ylabel('signal')

    mpl.subplot(6, 1, 2)
    mpl.plot(ampl(signal))
    mpl.ylabel('A')

    mpl.subplot(6, 1, 3)
    mpl.plot(phase(signal))
    mpl.ylabel('Ph')

    mpl.subplot(6, 1, 4)
    mpl.plot(np.abs(fft_ampl_window(signal, 100)))
    mpl.ylabel('sinc')

    mpl.subplot(6, 1, 5)
    mpl.plot(filter(signal, 100))
    mpl.ylabel('filtered')

    mpl.subplot(6, 1, 6)
    mpl.plot(np.abs(np.fft.fft(filter(signal, 100))))
    mpl.ylabel('fft_filtered')

    mpl.show()


########################################################################################################################

def ampl(signal):
    return np.array([1 if i <= len(signal) / 4 or i >= 3 * len(signal) / 4 else 0 for i in range(len(signal))])


def phase(signal):
    return np.array([4 * np.pi * i / len(signal) if i <= len(signal) / 4 else 4 * np.pi * (i - len(signal)) / len(
        signal) if 3 * len(signal) / 4 <= i <= len(signal) else 0 for i in range(len(signal))])


def fft_ampl_window(signal, N):
    a, b = np.split(np.fft.fft(ampl(signal)), 2)
    sinc = np.concatenate([b, a])
    return sinc * np.array([1 if len(signal) / 2 - N <= i <= len(signal) / 2 + N else 0 for i in range(len(signal))])


def filter(signal, N):
    h = np.fft.ifft(fft_ampl_window(signal, N))
    return np.real(np.fft.ifft(h * np.fft.fft(signal)))


def test_hf():
    # signal = np.array([np.sin(np.pi * i / 100) for i in range(1000)])
    # signal = np.array([np.sin(np.pi * i / 100) + 2 * (np.random.rand() - 0.5) for i in range(1000)])
    signal = np.array([np.sin(np.pi * i / 100) + 3 * np.sin(6 * np.pi * i / 100) for i in range(1000)])

    fig = mpl.figure()
    fig.set_figheight(9)
    fig.set_figwidth(10)

    mpl.subplot(6, 1, 1)
    mpl.plot(signal)
    mpl.ylabel('Сигнал')


    mpl.show()


def main():
    test_hf()


main()
