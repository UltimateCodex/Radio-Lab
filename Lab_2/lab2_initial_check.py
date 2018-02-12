import argparse

import matplotlib.pyplot as plt
import numpy as np

# parser = argparse.ArgumentParser(description='Get files to FFT')
#
# parser.add_argument('-data_file', type=str, help='file to be FFTed')
# parser.add_argument('-N', type=int, help='divisor used for sample frequency')
# parser.add_argument('-dual', type=bool, help='dual mode or nah')
# parser.add_argument('-volt_range', type=int, help='maximum in voltage range')
#
# args = parser.parse_args()
# N = parser.N
# dual = parser.dual
# data_taken = args.data_file


def scaleToReal(analog_values, volt_range=[-1, 1]):
    type_info = np.iinfo(np.int16)

    x1 = volt_range[0]
    x2 = volt_range[1]

    a1 = type_info.min
    a2 = type_info.max

    real_values = []
    real2_values = []
    for i in analog_values:
        real_values.append(float(x2 - x1) * float(i - a1) / float(a2 - a1) + x1)
    return np.asarray(real_values)


def getPowerSpectra(input_voltage):
    power, freqs = np.abs(getFFT(input_voltage)) ** 2
    return power


def getFFT(input_voltage):
    fourier = np.fft.fft(scaleToReal(input_voltage))
    freqs = np.fft.fftfreq(len(fourier), 1 / (v_samp)) / 1e6
    return fourier, freqs

if __name__ == '__main__':
    N = 1
    data_taken = "usb_1421MHz.npy"
    dual = True
    sample_size = len(data_taken)

    if dual:
        dual_imag = np.load(data_taken)[int(sample_size / 2):]
        dual_real = np.load(data_taken)[:int(sample_size / 2)]
        data = np.zeros((sample_size), dtype=np.complex)
    for i in range(N):
        data[i] = np.complex(dual_real[i], dual_imag[i])
    else:
        data = np.load(data_taken)

    v_samp = 62.5e6 / N

    fourier, freqs = getFFT(data)

    power = getPowerSpectra(data)

    plt.plot(np.fft.fftshift(freqs), np.fft.fftshift(power))
    plt.xlabel("Frequency (MHz)", fontsize=20)
    plt.ylabel("$(volt-second)$", fontsize=20)
    plt.show()


