import ugradio
import numpy as np
import time
import argparse

'Collects data from pico sampler and writes time of collection to filename'


parser = argparse.ArgumentParser(description='Get files to FFT')

parser.add_argument('-N', default = 16000, type=int, help='number of samples to collect')
parser.add_argument('-name', default = 'data', type=str, help='name of file')
parser.add_argument('-n', default = 1, type=int, help='number of blocks to collect')
parser.add_argument('-d', default = 2, type=int, help='divisor used for sample frequency')
parser.add_argument('-dual', default = False, type=bool, help='dual mode or nah')
parser.add_argument('-volt_range', default = '100mv', type=str, help='maximum in voltage range')


args = parser.parse_args()

n_samples = args.N
n_blocks = args.n
sample = args.d
dual = args.dual
volt_range = args.volt_range
filename = args.name


#t = time.localtime()
#filename = 'data_' + str(t.tm_mon) + ':' + str(t.tm_mday) + ':' + str(t.tm_min) + ':' + str(t.tm_sec) + str(sample) + '.npy'

current_data = ugradio.pico.capture_data(volt_range, divisor=sample, dual_mode=True, nsamples=n_samples, nblocks=n_blocks)
np.save(filename, current_data)


