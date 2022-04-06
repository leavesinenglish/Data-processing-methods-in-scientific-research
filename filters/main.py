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
    signal = np.array([np.sin(np.pi * i / 100) + 2 * (np.random.rand() - 0.5) for i in range(1000)])
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
def ampl_l(signal, A, B):
    #return np.array([1 if i <= len(signal) / 128 or i >= 127 * len(signal) / 128 else 0 for i in range(len(signal))])
    # A = len(signal)/4
    # B = 3*len(signal)/4
    return np.array([0 if A <= i <= B else (np.cos(np.pi*i/A)+1j*np.sin((i-len(signal))*np.pi/A)) for i in range(len(signal))])


def fft_ampl_window(signal, N, A,B):
    a, b = np.split(np.fft.ifft(ampl_l(signal, A, B)), 2)
    sinc = np.concatenate([b, a])
    sinc = sinc * np.array(
        [1 if len(signal) / 2 - N <= i <= len(signal) / 2 + N else 0 for i in range(len(signal))])
    a, b = np.split(sinc, 2)
    return np.concatenate([b, a])


def filter_l(signal, N, A, B):
    h = np.fft.fft(fft_ampl_window(signal, N, A,B))
    return np.real(np.fft.ifft(h * np.fft.fft(signal)))


def test_lf():
    size = 1000
    # signal = np.array([np.sin(np.pi * i / 100) for i in range(1000)])
    signal = np.array([1 + 0.5 * np.random.rand() if ((i <= size / 8) or (
            size / 2 <= i <= 5 * size / 8) or size / 4 <= i <= 3 * size / 8 or 3 * size / 4 <= i <= 7 * size / 8)
                       else 0.5 * np.random.rand() for i in range(size)])
    # signal = np.array([np.sin(np.pi * i / 100) + 3 * np.sin(6 * np.pi * i / 100) for i in range(1000)])

    fig = mpl.figure()
    fig.set_figheight(15)
    fig.set_figwidth(15)

    mpl.subplot(7, 1, 1)
    mpl.plot(signal)
    mpl.ylabel('signal')

    mpl.subplot(7, 1, 2)
    mpl.plot(np.abs(ampl_l(signal, len(signal)/4, 3*len(signal)/4)))
    mpl.ylabel('A')

    mpl.subplot(7, 1, 3)
    mpl.plot(np.angle(ampl_l(signal, len(signal)/4, 3*len(signal)/4)))
    mpl.ylabel('Ph')

    mpl.subplot(7, 1, 4)
    mpl.plot(np.abs(fft_ampl_window(signal, 100, len(signal)/4, 3*len(signal)/4)))
    mpl.ylabel('sinc')

    mpl.subplot(7, 1, 5)
    mpl.plot(filter_l(signal, 100, len(signal)/4, 3*len(signal)/4))
    mpl.ylabel('filtered')

    mpl.subplot(7, 1, 6)
    mpl.plot(np.abs(np.fft.fft(filter_l(signal, 100, len(signal)/4, 3*len(signal)/4))))
    mpl.ylabel('fft_filtered')

    mpl.subplot(7, 1, 7)
    mpl.plot(np.abs(np.fft.fft(signal)))
    mpl.ylabel('fft_unfiltered')

    mpl.show()

########################################################################################################################
def ampl_h(signal, A, B):
    # A = len(signal) / 8
    # B = 7 * len(signal) / 8
    return np.array([(np.cos(np.pi*i/A)+1j*np.sin((i-len(signal))*np.pi/A)) if A <= i < B else 0 for i in range(len(signal))])


def fft_ampl_window_h(signal, N, A, B):
    a, b = np.split(np.fft.ifft(ampl_h(signal, A, B)), 2)
    sinc = np.concatenate([b, a])
    sinc = sinc * np.array(
        [1 if len(signal) / 2 - N <= i <= len(signal) / 2 + 2 * N else 0 for i in range(len(signal))])
    a, b = np.split(sinc, 2)
    return np.concatenate([b, a])


def filter_h(signal, N, A, B):
    h = np.fft.fft(fft_ampl_window_h(signal, N, A, B))
    return np.real(np.fft.ifft(h * np.fft.fft(signal)))


def filter_h_2(signal, N, A, B):
    return signal - filter_l(signal, N, A, B)


def test_hf():
    # signal = np.array([np.sin(np.pi * i / 100) for i in range(1000)])
    # signal = np.array([np.sin(np.pi * i / 100) + np.random.rand() - 0.5 for i in range(1000)])
    # signal = np.array([np.sin(np.pi * i / 5) + 3 * np.sin(np.pi * i / 100) for i in range(1000)])
    signal = np.array([np.sin((np.pi * i) / 2) + 2 * np.sin((np.pi * i) / 50) for i in range(1000)])

    fig = mpl.figure()
    fig.set_figheight(13)
    fig.set_figwidth(13)

    mpl.subplot(7, 1, 1)
    mpl.plot(signal)
    mpl.ylabel('signal')

    mpl.subplot(7, 1, 2)
    mpl.plot(np.abs(ampl_h(signal, len(signal)/8, 7*len(signal)/8)))
    mpl.ylabel('A')

    mpl.subplot(7, 1, 3)
    mpl.plot(np.angle(ampl_h(signal, len(signal)/8, 7*len(signal)/8)))
    mpl.ylabel('Ph')

    mpl.subplot(7, 1, 4)
    mpl.plot(np.real(fft_ampl_window_h(signal, 100, len(signal)/8, 7*len(signal)/8)))
    mpl.ylabel('sinc')

    mpl.subplot(7, 1, 5)
    mpl.plot(filter_h_2(signal, 100, len(signal)/8, 7*len(signal)/8))
    mpl.ylabel('filtered')

    mpl.subplot(7, 1, 6)
    mpl.plot(np.abs(np.fft.fft(signal, 100)))
    mpl.ylabel('fft_unfiltered')

    mpl.subplot(7, 1, 7)
    mpl.plot(np.abs(np.fft.fft(filter_h_2(signal, 100, len(signal)/8, 7*len(signal)/8))))
    mpl.ylabel('fft_filtered')

    mpl.show()

########################################################################################################################
def notch_filter(signal, N, A, B, A1, B1):
    h = np.fft.fft(fft_ampl_window_h(signal, N, A, B)) + np.fft.fft(fft_ampl_window(signal, N, A1, B1))
    return np.real(np.fft.ifft(h * np.concatenate([np.fft.fft(signal), np.zeros(h.size - signal.size)])))


def stripe_filter(signal, N, A, B, A1, B1):
    h = np.fft.fft(fft_ampl_window_h(signal, N, A,B)) * np.fft.fft(fft_ampl_window(signal, N, A1, B1))
    return np.real(np.fft.ifft(h * np.concatenate([np.fft.fft(signal), np.zeros(h.size - signal.size)])))


def test_notch_stripe():
    #signal = np.array([np.sin(np.pi * i / 100) for i in range(1000)])
    signal = np.array([5*np.sin(np.pi * i / 15) + 2*(np.random.rand() - 0.5) for i in range(1000)])
    #signal = np.array([np.sin((np.pi * i) / 3) + np.sin((np.pi * i) / 100) for i in range(1000)])
    signal_1 = signal

    fig = mpl.figure()
    fig.set_figheight(9)
    fig.set_figwidth(10)

    mpl.subplot(6, 1, 1)
    mpl.plot(signal)
    mpl.ylabel('signal')

    mpl.subplot(6, 1, 2)
    mpl.plot(np.abs(np.fft.fft(signal)))
    mpl.ylabel('fft_signal')

    mpl.subplot(6, 1, 3)
    mpl.plot(np.abs(np.fft.fft(fft_ampl_window_h(signal, 100, len(signal)*0.01, 0.99*len(signal))) * np.fft.fft(fft_ampl_window(signal, 100, len(signal)*0.1, 0.9*len(signal)))))
    #mpl.plot(np.abs(fft_ampl_window(signal, 100,)))
    mpl.ylabel('h_stripe')

    mpl.subplot(6, 1, 4)
    mpl.plot(stripe_filter(signal, 100, len(signal)*0.01, 0.99*len(signal), len(signal)*0.1, 0.9*len(signal)))
    mpl.ylabel('stripe')

    mpl.subplot(6, 1, 5)
    mpl.plot(np.abs(np.fft.fft(fft_ampl_window_h(signal, 100, len(signal)*0.1, 0.9*len(signal))) + np.fft.fft(fft_ampl_window(signal, 100, len(signal)*0.01, len(signal)*0.99))))
    mpl.ylabel('h_notch')

    mpl.subplot(6, 1, 6)
    mpl.plot(notch_filter(signal, 100, len(signal)*0.1, 0.9*len(signal), len(signal)*0.01, 0.99*len(signal)))
    mpl.ylabel('notch')

    mpl.show()


def main():
    # test_maf()
    # test_lf()
    #test_hf()
    test_notch_stripe()


main()
