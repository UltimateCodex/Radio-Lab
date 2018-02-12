import ugradio
import numpy as np
volt_range = '100mV'
sampling_range = [1, 2, 3]

for sample in sampling_range: 
	current_data = ugradio.pico.capture_data(volt_range, divisor=sample, dual_mode=True, nsamples=16000, nblocks=1)
	np.save("initial_test_" + str(volt_range) + "_" + str(sample) + ".npy", current_data)


