import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math

"""
@summary: {0,1} stream to {+1,-1} stream
@param data : np.array()
@return: np.array()
"""
def center_data(data):
	
	match_offset = (data==0).nonzero()[0]
	data[match_offset] = -1
	return data

"""
@summary: {+x,-x} stream to {+1,-1} stream
@param data : np.array()
@return: np.array()
"""
def normalize_data(data):
	
	max = data.max()
	return data/max


"""
@summary: {+x,-x} stream to {+1,-1} stream
@param data : np.array()
@return: np.array()
"""
def normalize_complex(data):
	return data/np.abs(data).max()



"""
@summary: Samples repetition interpolation (no filtering)
@param stream : np.array()
@param ratio  : int
@return: np.array()
"""
def hard_interpolate(stream, ratio, alpha=0):	

	if alpha == 0:
		ones = np.array([[1]]*ratio)
		mult_matrix = stream*ones
		out = np.reshape(mult_matrix,(1, len(stream)*ratio), 'F') 
		out = out[0]
	else:
		out = []
		error = 0.0
		
		for i in range(len(stream)):
			out = out + [stream[i]]*int(ratio+round(error))
			if error > 0.5:
				error = 0
			error += alpha
	
	return np.array(out)

"""
@summary: Frequency shift of a complex input signal
@param data        : complex np.array()
@param freq_offset : frequency offset in Hz (int)
@param fs          : sample frequency in Hz (int)
@return: complex np.array()
"""
def freq_shift(data, freq_offset, fs):
	
	time = np.arange(0,len(data), dtype = float) * (1/fs)
	data_shift = np.exp(2*np.pi*freq_offset*time*1j)

	return data_shift * data


"""
FIXME : Use fs to get correct frequency deviation

@summary: Compute frequency demodulation of a raw IQ vector
@param data        : complex np.array()
@param fs          : sample frequency in Hz (int)
@return: float np.array()
"""
def freq_demod(data, fs):
	
	# Get signal phase
	raw_angle = np.angle(data)
	angle = np.unwrap(raw_angle)
	
	# derive to get frequency:
	freq_data = (angle[1:len(angle):2]-angle[0:len(angle)-1:2])

	return freq_data


"""
@summary: linear frequency modulation of input vector 
@param data        : complex np.array() assuming vector in [-1..+1] range
@param deviation   : frequency deviation in Hz
@param carrier     : carrier frequency in Hz
@param fs          : sample frequency in Hz (int)
@return: float np.array()
"""
def freq_mod(data, deviation, carrier, fs):
	
	time = np.arange(0,len(data),dtype=np.float)/fs
	frequency = carrier + (data*deviation)
	data_out = np.exp(2*np.pi*frequency*time*1j)

	return np.array(data_out)


"""
@summary: remove DC offset on a complex signal from all signal mean. 
@param data : complex np.array()
@param window_size : on the whole vector if -1, or sliding buffer of length window_size 
@return: complex np.array()
"""
def remove_dc_offset(data, window_size=-1):

	if window_size == -1:
		if np.iscomplex(data[0]):
			q_av = np.average(data.imag)
			i_av = np.average(data.real)
			
			data.imag = data.imag - q_av
			data.real = data.real - i_av
		else:
			av = np.average(data)
			data = data - av

	else:
		for k in range(0, len(data) - window_size):
			if np.iscomplex(data[0]):
				q_av = np.average(data.imag[k:k+window_size])
				i_av = np.average(data.real[k:k+window_size])
				
				data.imag[k] = data.imag[k] - q_av
				data.real[k] = data.real[k] - i_av
			else:
				av = np.average(data[k:k+window_size])
				data[k] = data[k] - av
			
	return data


"""
@summary: Hard decision of +/-1 of float signal 
@param data : float np.array()
@param neg : True/False neagate decision 
@return: float {-1;+1} np.array()
"""
def hard_decision(data, neg=False):
	
	offsets = (data < 0).nonzero()[0]
	data[offsets] = -1.0
	offsets = (data >= 0).nonzero()[0]
	data[offsets] = 1.0

	if neg == True:
		data = data*(-1)

	return data


"""
@summary: filtered signal y(k) = a.x(k) + b.y(k-1)
@param data : float np.array()
@param alpha : float
@param beta : float 
@return: float np.array()
"""
def smooth_filter(data, alpha, beta):
	y=[]
	y.append(alpha*data[0]);
	for x in data[1:]:
		y.append(alpha*x + beta*(y[len(y)-1]));
	
	return np.array(y)


"""
@summary: filtered signal y(k) = a.x(k) + b.y(k-1)
@param data : float np.array()
@param alpha : float
@param beta : float 
@return: float np.array()
"""
def sliding_average(data, size):
	y=[]
	for k in range(len(data)-size):  
		y.append(np.average(data[k:k+size]))	
	return np.array(y)



"""
@summary: Generates a clock signal with a rate given by 'period' samples
		  and of duration 'length'. 'alpha' is a non integer clock drift
		  correction
@param period : int (FIXME : for the moment must be even)
@param length : int must be a multiple of 'period'
@param alpha : float {0..0.5} time drift correction for non round period regarding to sample rate
@return: float np.array()
"""
def generate_clk(period, length, alpha, max_clk=1, min_clk=-1):
	
	clk = []
	itterations = int(length / period)
	error = 0.0
	
	for __ in range(itterations):
		clk = clk + [max_clk]*int(period/2)+[min_clk]*int(period/2+round(error))
		if error > 0.5:
			error = 0
		error += alpha
	
	return np.array(clk)


"""
@summary: Look for Clock  Sinc signal, and return the 
          positions in the frame of the sequence 
          Clock SYNC frame = 0xFFFE
@param data         : float np.array() stream
@param clock_period : number of samples per clock period
@param min_sync_len : minimum number of clock periods in the sync
@param max_sync_len : maximum number of clock periods in the sync
@return: float np.array()
"""
def clock_sync_frames(data, clock_period, min_sync_len, max_sinc_len):
	
	frame_start = []		
	before_sample = data[0]
	edge_count = 0
	edge_offset = 0
	prev_edge_offset = 0 
	current_offset = 1

	for sample in data[1:]:
		# Track rising edge :
		if (before_sample < 0) and (sample > 0):
			prev_edge_offset = edge_offset
			edge_offset = current_offset
			distance = edge_offset - prev_edge_offset
			if distance > int(clock_period * 1.2) and (edge_count >= min_sync_len) and min_sync_len <= max_sinc_len :
				# End Sync Bit detected :
				frame_start.append(current_offset-int(clock_period*0.5))
				edge_count = 0

			if abs(distance - clock_period) < 2:
				# Rising clock edge case :
				edge_count += 1
			else:
				# Wrong Alarm :
				edge_count = 0
				
		current_offset +=1
		before_sample = sample
		
	return frame_start


"""
@summary: correlates the training sequence signal to the data stream and return
          positions in the frame of the sequence 
@param data      : float np.array() stream
@param sync_seq  : float np.array() training sequence stream
@param threshold : float peak detection threshold
@return: float np.array()
"""
def sync_frames(data, sync_seq, threshold, ):
	
	corr_out = np.abs(np.correlate(data, sync_seq ))
	matches = (corr_out > threshold).nonzero()[0]
		
	first_match = matches[0]
	aggrega = []
	frame_start = []
	vec_len = len(matches)
	sync_len = len(sync_seq)
	
	for i in range(vec_len):
		offset = matches[i]
		if ((offset-first_match) < sync_len)&(i< vec_len - 1):
			aggrega.append(offset)
		else:
			max_found = np.max(corr_out[aggrega])
			match_offset = (corr_out[aggrega]==max_found).nonzero()[0]
			frame_start.append(aggrega[match_offset])
			first_match = offset
			aggrega = []
	
	return frame_start


	
	