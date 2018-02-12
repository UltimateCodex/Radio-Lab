import ugradio
import numpy as np
possible_volt_ranges = ['50mV', '100mV', '200mV', '500mV', '1V', '2V']
for volt_range in possible_volt_ranges: 
	current_data = ugradio.pico.capture_data(volt_range, divisor=1, dual_mode=True, nsamples=16000, nblocks=1)
	np.save("initial_test_" + str(volt_range) + ".npy", current_data)


