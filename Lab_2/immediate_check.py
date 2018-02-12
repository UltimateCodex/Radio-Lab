import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Get files to FFT')

parser.add_argument('-data_file', type=str, help='file to be FFTed')
parser.add_argument('-N', type=int, help='divisor used for sample frequency')
parser.add_argument('-dual', type=bool, help='dual mode or nah')
parser.add_argument('-volt_range', type=str, help='maximum in voltage range')
parser.add_argument('-title', type=str, help='Title of plot')


args = parser.parse_args()

filename = args.data_file
divisor = args.N
dual = args.dual
data_taken = args.data_file
volt_range = args.volt_range
title = args.title

def scaleToReal(analog_values, volt_range=[-1, 1]):
    type_info = np.iinfo(np.int16)

    x1 = volt_range[0]
    x2 = volt_range[1]

    a1 = type_info.min
    a2 = type_info.max

    real_values = np.zeros(len(analog_values), dtype=np.complex)

    for i in range(len(analog_values)):
        real_values[i] = (float(x2 - x1) * float(analog_values[i] - a1) / float(a2 - a1) + x1)
    return np.asarray(real_values)


# def scaleToReal(analog_values):
#     return analog_values/16384

def getPowerSpectra(input_voltage):
    fourier = np.fft.fft(input_voltage)
    freqs = np.fft.fftfreq(len(fourier), 1 / (v_samp)) / 1e6
    power = np.abs(fourier) ** 2
    return power, freqs

if dual:
    data = np.load(filename)
    sample_size = int(len(data) / 2)

    dual_imag = scaleToReal(np.load(filename), [-int(volt_range[:-2]), int(volt_range[:-2])])[int(sample_size):]
    dual_real = scaleToReal(np.load(filename), [-int(volt_range[:-2]), int(volt_range[:-2])])[:int(sample_size)]

    data = np.zeros(sample_size, dtype=np.complex)

    for i in range(int(sample_size)):
        data[i] = np.complex(dual_real[i], dual_imag[i])

else:
    data = np.load(filename)
    sample_size = len(data)

v_samp = 62.5e6 / divisor

power, freqs = getPowerSpectra(data)

plt.plot(np.fft.fftshift(freqs), np.fft.fftshift(power))
plt.xlabel("Frequency (MHz)", fontsize=20)
plt.ylabel("Power $(volt-second)^2$", fontsize=20)
plt.title(title)
plt.savefig("lab2_" + title + ".pdf")
plt.close()

