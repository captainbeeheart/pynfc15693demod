import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math


SAMPLE_FREQ        = 2e6
RF_PULSE           = int(math.ceil(9.44e-6 * SAMPLE_FREQ))   # Short pulse duration in Samples
SYMBOL_LENGTH      = 8 * RF_PULSE                            # RF symbol length (2 bits for 1 out of 4 coding)

SYNC_LENGTH = 5 * RF_PULSE
TOLERENCE   = 6

PULSE_00    = 2 * RF_PULSE
PULSE_01    = 4 * RF_PULSE
PULSE_10    = 6 * RF_PULSE
PULSE_11    = 8 * RF_PULSE
EOF_PULSE   = 3 * RF_PULSE




def set_sample_freq(sample_freq):
    global SAMPLE_FREQ
    global RF_PULSE
    global SYNC_LENGTH
    global PULSE_00
    global PULSE_01
    global PULSE_10
    global PULSE_11
    global EOF_PULSE
    SAMPLE_FREQ = sample_freq
    RF_PULSE           = int(math.ceil(9.44e-6 * SAMPLE_FREQ))  
    SYNC_LENGTH = 5 * RF_PULSE
    PULSE_00    = 2 * RF_PULSE
    PULSE_01    = 4 * RF_PULSE
    PULSE_10    = 6 * RF_PULSE
    PULSE_11    = 8 * RF_PULSE
    EOF_PULSE   = 3 * RF_PULSE
"""
get_sof(signal)
Returns a vector of indexes where a SOF frame has been detected
Each index dorresponds to the first rising edge pulse after the SOF

...---+        +--------------------+        +----------------x
      |        |                    |        |                x
      |  <1T>  |      <4T>          |  <1T>  |  <2T>          x 
      +--------+                    +--------+                x---...
                                                              ^
                                                              |
                                                              +- Points here
"""
def get_sof(signal, threshold = 0.4):
    frame_start = []
    before_sample = signal[0]
    edge_count = 0
    edge_offset = 0
    prev_edge_offset = 0 
    current_offset = 1

    for sample in signal[1:]:
        # Track rising edge :
        if (before_sample < threshold) and (sample > threshold):
            prev_edge_offset = edge_offset
            edge_offset = current_offset
            distance = edge_offset - prev_edge_offset
            if (distance > (SYNC_LENGTH-TOLERENCE)) and (distance < (SYNC_LENGTH+TOLERENCE)):
                # End Sync Bit detected :
                # Add offset to point end of SOF 
                frame_start.append(current_offset + 2*RF_PULSE )
                edge_count += 1
                
        current_offset +=1
        before_sample = sample
        
    return frame_start


"""
Compute Frame CRC from VICC
"""
POLYNOMIAL   = 0x8408
PRESET_VALUE = 0xFFFF
CHECK_VALUE  = 0xF0B8

def crc_core(data):
    current_crc_value = PRESET_VALUE
    for byte in data:
        current_crc_value = (current_crc_value ^ byte)& 0xFFFF
        for j in range(8):
            if (current_crc_value & 0x0001):
                current_crc_value = ((current_crc_value >> 1) ^ POLYNOMIAL)& 0xFFFF
            else:
                current_crc_value = (current_crc_value >> 1)&0xFFFF
    return current_crc_value


def check_crc(data):
    result = crc_core(data)
    if result == CHECK_VALUE:
        # print("CRC OK")
        return True
    else:
        # print("CRC Error")
        return False


def calc_crc(data):
    result = 0xFFFF - crc_core(data)
    print("CRC = %x" %result)
    return result


"""
Reorganize bits in the correct order, Msbit first as LsBits are transmit first 2 by 2
"""
def reorder_bits(data):
    numBitsFloor = math.floor(len(data)/8)
    numBitsCeil = math.ceil(len(data)/8)

    if numBitsFloor-numBitsCeil == 0:
        data1   = data.reshape(int(len(data)/2), 2)
        dataOut = data1.reshape(len(data1),-1)[::-1,:]
        dataOut = dataOut.reshape(1, len(data))
        dataOut = np.packbits(dataOut)
        dataOut = dataOut.reshape(1, len(dataOut))[:,::-1][0]
    else:
        return None

    return dataOut



def decode_frame(signal, threshold = 0.4):
    data = []        
    before_sample = signal[0]
    edge_offset = 0
    prev_edge_offset = 0 
    current_offset = 1

    while current_offset < len(signal):
        sample = signal[current_offset]
        # Track rising edge :
        if (before_sample < threshold) and (sample > threshold):
            prev_edge_offset = edge_offset
            edge_offset = current_offset
            distance = edge_offset - prev_edge_offset
            if (distance > (PULSE_00-TOLERENCE)) and (distance < (PULSE_00+TOLERENCE)):
                data += [0,0]
                current_offset += 6*RF_PULSE
                edge_offset = current_offset

            elif (distance > (PULSE_01-TOLERENCE)) and (distance < (PULSE_01+TOLERENCE)):
                data += [0,1]
                current_offset += 4*RF_PULSE
                edge_offset = current_offset

            elif (distance > (PULSE_10-TOLERENCE)) and (distance < (PULSE_10+TOLERENCE)):
                data += [1,0]
                current_offset += 2*RF_PULSE
                edge_offset = current_offset

            elif (distance > (PULSE_11-TOLERENCE)) and (distance < (PULSE_11+TOLERENCE)):
                data += [1,1]
                current_offset += 1
                edge_offset = current_offset

            elif (distance > (EOF_PULSE-TOLERENCE)) and (distance < (EOF_PULSE+TOLERENCE)):
                # print("EOF found")
                break
            else:
                pass
                # print("Decoding Error...")

        else:
            current_offset += 1

        before_sample = sample
        

    data = np.array(data)
    if len(data) > 0 :

        data = reorder_bits(data)
        if data is not None:
            if len(data) > 4 :
                if check_crc(data):
                    #print("   %r" %(list(map(hex,data))))
                    return data
                else:
                    print("CRC error")
                    return None

    return None


if __name__ == "__main__":

    # CRC Check: Must be equal to 0xBAE3
    calc_crc([1,2,3,4])
    calc_crc([0x22, 0x20, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0x04, 0xE0, 0x0B])
    check_crc([0x22, 0x20, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0x04, 0xE0, 0x0B, 0xE3, 0xBA])
    check_crc([0x22, 0x20, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0x04, 0xE0, 0x0B, 0xE3, 0xBB])

    