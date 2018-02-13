import matplotlib.pyplot as plt
import numpy as np


def scaleToReal(analog_values, volt_range=[-1, 1]):
    type_info = np.iinfo(np.int16)

    x1 = volt_range[0]
    x2 = volt_range[1]

    a1 = type_info.min
    a2 = type_info.max

    real_values = []

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
    N = [1]
    volt_range = ['100mV']

    for divisor in N:
        for volts in volt_range:
            filename = "initial_test_" + volts + "_" + str(divisor) + ".npy"
            dual = True

            if dual:
                data = np.load(filename)
                sample_size = int(len(data) / 2)

                dual_imag = scaleToReal(np.load(filename), [-int(volts[:-2]), int(volts[:-2])])[int(sample_size):]
                dual_real = scaleToReal(np.load(filename), [-int(volts[:-2]), int(volts[:-2])])[:int(sample_size)]
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

    plt.semilogy(np.fft.fftshift(freqs), np.fft.fftshift(power))
    plt.xlabel("Frequency (MHz)", fontsize=20)
    plt.ylabel("Power $(volt-second)^2$", fontsize=20)
    plt.title("Divisor = " + str(divisor) + ", Voltage Max = " + volts)
    plt.savefig(str(divisor) + "_" + str(volts) + "_" + "no_test" + ".pdf")
    plt.show()
