import struct
import numpy as np

"""
FIXME : use correct python api instead of ugly code below...

@summary:  MSB first serializer
@param streams : two dimentional np.array([[]])
@param msb_first : True/False
@return: byte stream and negated byte stream for each input line and bit reverse  result
"""
from numpy import float32
def bit_string_to_byte(streams):
	
	byte_stream = []
	
	for stream in streams:
		bit_array = np.array(stream)
		byte_stream.append(np.packbits(bit_array))

	return byte_stream

"""
@summary:  MSbit first serializer
@param data : string input b''
@return: bit np.array()
"""
def serialize(data):

	data_a = np.fromstring(data, dtype=np.uint8)
	bit_stream =  np.unpackbits(data_a)
	return bit_stream.astype(np.float32)


"""
@summary:  gqrx raw IQ file 32 bits float little endian stream
@param data     : complex np.array()
@param file_out : file name string
@return: 
"""
def cplx_float_to_float_file(data, file_out):

	f_data = open(file_out, 'wb')

	for cplx in data:
		data = f_data.write(struct.pack("<f",cplx.real)); 		
		data = f_data.write(struct.pack("<f",cplx.imag)); 

	f_data.close()


"""
@summary:  Create Hackrf raw IQ file 8 bits signed stream
@param data     : {-1/+1} float complex np.array()
@param file_out : file name string
@return: 
"""
def cplx_float_to_byte_file(data, file_out):

	f_data = open(file_out, 'wb')

	# Normalize on signed Byte range :
	data = data * 127

	for cplx in data:
		data = f_data.write(struct.pack("b",cplx.real)); 		
		data = f_data.write(struct.pack("b",cplx.imag)); 

	f_data.close()


"""
@summary:  Create raw IQ file unsigned 8 bits signed stream
@param data     : {-1/+1} float complex np.array()
@param file_out : file name string
@return: 
"""
def cplx_float_to_unsigned_byte_file(data, file_out):

	f_data = open(file_out, 'wb')

	# Normalize on signed Byte range :
	data = (data+1+1j) * 127

	for cplx in data:
		data = f_data.write(struct.pack("B",cplx.real)); 		
		data = f_data.write(struct.pack("B",cplx.imag)); 

	f_data.close()

"""
@summary:  bits numpy array saved to text file
@param data     : bit arrays [[]]
@param file_out : file name string
@return: 
"""
def bit_stream_to_file(data, file_out):

	f_data = open(file_out, 'w')
	for frame in data:
		for bit in frame:
			if bit == 1:
				f_data.write('1')
			else:
				f_data.write('0') 	 	
		f_data.write('\n')
	f_data.close()


"""
@summary:  creates raw 16bits little endian stream file from float array
@param data     : float np.array()
@param file_out : file name string
@return: 
"""
def float_to_int16_file(data, file_out):

	f_data = open(file_out, 'wb')

	for sample in data:
		bin_spl = struct.pack(">h",int(sample))
		data = f_data.write(bin_spl); 		

	f_data.close()
	

"""
@summary: gqrx raw IQ file 32 bits float little endian stream to float cplx array
@param data     : float np.array()
@param swap_iq  : True/False
@return: complex np.array() 
"""
def bin_stream_to_cplx_float(data, swap_iq):
	i = 0
	out_data = []
	if swap_iq :
		for i in range(0,len(data),8):
			i_data = struct.unpack("<f",data[i:i+4])[0]
			q_data = struct.unpack("<f",data[i+4:i+8])[0]
			out_data.append(q_data+1j*i_data)
	else:
		for i in range(0,len(data),8):
			i_data = struct.unpack("<f",data[i:i+4])[0]
			q_data = struct.unpack("<f",data[i+4:i+8])[0]
			out_data.append(i_data+1j*q_data)

		
	return np.array(out_data)

