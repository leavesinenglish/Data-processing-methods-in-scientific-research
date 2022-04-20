import numpy as np
import matplotlib.pyplot as mpl
import scipy.signal as scp

#+ 3*np.sin(2*np.pi*5*t) +random
def gaus(x, sigma=1, mu=0):
    return np.array([np.exp(-(i - mu) ** 2 / (sigma ** 2)) for i in x])


def lff(x, x0=3):
    return np.array(
        [complex(1., abs(i) * 2 * np.pi / x0) if -(x0 + 0.01) < i < (x0 + 0.01) else complex(0., 0.) for i in x])


def low_freq_filter(signal, freq, t):
    df = 1. / t
    N = len(signal)
    xf = np.fft.fftfreq(N) * N * df
    ideal_lff_F = lff(xf, freq)
    ideal_lff_F_c = ideal_lff_F
    g = gaus(np.concatenate((xf[int(N / 2):], xf[:int(N / 2)])), freq)
    ideal_lff = np.fft.ifft(ideal_lff_F_c)
    ideal_lff_real = ideal_lff
    ideal_lff_real = np.concatenate((ideal_lff_real[int(N / 2):], ideal_lff_real[:int(N / 2)]))
    ideal_lff_real_w = ideal_lff_real[int(N / 2 - 25):int(N / 2) + 25]
    graph_ideal_lff_real_w = np.concatenate(
        (np.zeros((int(N / 2) - 25)), ideal_lff_real_w, np.zeros((int(N / 2) - 25))))
    ideal_lff_real_wg = graph_ideal_lff_real_w * g
    return np.convolve(signal, ideal_lff_real_wg[int(N / 2) - 25:int(N / 2) + 25])[25:N + 25].real


def demod(signal, a, f0, L, size):
    t = np.linspace(0, L, size)
    cos = a * np.cos(2 * np.pi * f0 * t)
    sin = a * np.sin(2 * np.pi * f0 * t)
    Re = signal * sin
    Im = signal * cos
    Re = low_freq_filter(Re, f0, L)
    Im = low_freq_filter(Im, f0, L)
    return np.array([complex(Re[i], Im[i]) for i in range(len(Re))])


def test():
    fig = mpl.figure()
    fig.set_figheight(10)
    fig.set_figwidth(10)
    mpl.rcParams["figure.dpi"] = 300

    f0 = 20
    a = 1
    L = 10
    size = 1000
    t = np.linspace(0, L, size)
    random = np.array([np.random.rand() - 0.5 for _ in range(size)])
    signal = np.sin(2 * np.pi * 5 * t) + 2*np.sin(2*np.pi*35*t) + random

    mpl.subplot(5, 1, 1)
    mpl.plot(t, signal)
    mpl.title("signal")

    mpl.subplot(5, 1, 2)
    mpl.plot(np.abs(np.fft.fft(signal)))
    mpl.title("fft signal")

    mpl.subplot(5, 1, 3)
    res = demod(signal, a, f0, L, size)
    mpl.plot(t, np.real(res))
    mpl.title("real demodulated signal")

    mpl.subplot(5, 1, 4)
    mpl.plot(t, np.imag(res))
    mpl.title("imag demodulated signal")

    mpl.subplot(5, 1, 5)
    mpl.plot(np.abs(np.fft.fft(res)))
    print("peak frequency = "+str(np.argmax(np.abs(np.fft.fft(res)))/L))
    fourier = np.abs(np.fft.fft(res))
    res2 = np.array([fourier[i] if i != np.argmax(fourier) else 0 for i in range(fourier.size)])
    print("peak frequency = "+str(np.argmax(res2)/L))
    mpl.title("fft demodulated")

    mpl.subplots_adjust(hspace=0.5)
    mpl.show()


test()

# посчитать частоту демодулированного
# cуперпозиция синусов чтобы получить несимметричный сигнал
