import matplotlib.pyplot as plt
import numpy as np
import binascii
import argparse
import math
from sig_tool import transforms
from sig_tool import sig
from sig_tool import ncf15693
from multiprocessing import Pool

# Tune number of parallel processing:
NB_PROC = 4

CHUNK_SIZE = 5000000
FRAME_SIZE = 312496

def parseInputs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, help="Input file to process")
    parser.add_argument("-sf", "--sample_freq", type=int, help="Sampes freq in Hz")
    return parser.parse_args()



def printData(data_list):
    for data in data_list:
        if len(data) > 0 :
            print("   %r" %(list(map(hex,data[1]))))



def getSynchOffsets(data, sample_offset):
    iqData = transforms.bin_stream_to_cplx_float(data,0)
    # Get amplitude and normalise data:
    iqAmp = np.absolute(iqData)
    iqNorm = sig.normalize_data(iqAmp)

    # Filter to smooth transitions:
    iqNorm = sig.sliding_average(iqNorm[:len(iqNorm-8)],6)

    # print("Get SYNC frames...")
    # Offset is in raw data bytes
    offsetList = ncf15693.get_sof(iqNorm)
    if offsetList is not None:
        offsetList = np.multiply(offsetList, 8)
        offsetList = np.add(offsetList, sample_offset)
    return offsetList



def decodeFrame(frame_chunk):
    
    iqData = transforms.bin_stream_to_cplx_float(frame_chunk,0)
    # Get amplitude and normalise data:
    iqAmp = np.absolute(iqData)
    iqNorm = sig.normalize_data(iqAmp)
    # Filter to smooth transitions:
    # print("Filtering data...")
    iqNorm = sig.sliding_average(iqNorm[:len(iqNorm-8)],6)
    data = ncf15693.decode_frame(iqNorm) 
    return data



def getChunkFrames(params):
    # Look for SOF frame start in the signal:
    # not optimal as parsing the file twice to avoid to trunc frames...
    samples = params.samples
    chunk_offset = params.chunk_offset
    chunk_number = params.chunk_len

    frame_offsets = np.array([])
    for k in range(chunk_offset, chunk_offset+chunk_number):
        print("Processing chunk #%d..." %k)
        chunk0_frames = getSynchOffsets(samples[k*CHUNK_SIZE:(k+1)*CHUNK_SIZE], k*CHUNK_SIZE)
        if chunk0_frames is not None:
            frame_offsets =  np.concatenate((frame_offsets, chunk0_frames))

        # Chech Half chunk to get split frames over 2 chunks
        if k > 0 :
            print("Processing chunk #%d boundary..." %k)
            chunk1_frames = getSynchOffsets(samples[(k*CHUNK_SIZE)-FRAME_SIZE:(k*CHUNK_SIZE)+FRAME_SIZE], (k*CHUNK_SIZE)-FRAME_SIZE)
            if chunk1_frames is not None:
                frame_offsets = np.concatenate((frame_offsets, chunk1_frames))

    frame_offsets = np.unique(frame_offsets)
    print("ChunkId %d, found %d" %(chunk_offset, len(frame_offsets)))
    return frame_offsets



def spawnSofProcess(params):
    frame_offset = []
    with Pool(processes = NB_PROC) as pool:
        frame_offset = frame_offset + pool.map(getChunkFrames, params)

    # Flatten resutl
    offset_list = np.array((1))
    for k in frame_offset:
        offset_list = np.append(offset_list,k)
    
    return np.unique(offset_list)



def decodeFrameProcess(frame_offsets):
    data_frames = []
    for offset in frame_offsets:
        print("decoding frame offset %d" %offset)
        data = decodeFrame(samples[int(offset): int(offset) + FRAME_SIZE])
        if data is not None:
            data_frames.append([offset, data])
    return data_frames


def spawnDecodeFrameProcess(params):
    decoded_data = []
    with Pool(processes = NB_PROC) as pool:
        data_frames = pool.map(decodeFrameProcess, params)
        if data_frames is not []:
            decoded_data = decoded_data + data_frames

    # Flatten resutl
    data = []
    for k in decoded_data:
        data = data +k

    return data


class process_param():
    def __init__(self, samples, chunk_offset, chunk_len):
        self.samples = samples
        self.chunk_offset = chunk_offset
        self.chunk_len = chunk_len


def getChunkList(chunkNbr, nb_proc):
    nbr_process = math.floor(chunkNbr  / nb_proc)
    chunk_split = [nbr_process]*(nb_proc-1)+[chunkNbr - ((nb_proc-1)*math.floor(chunkNbr/nb_proc))]
    return chunk_split



if __name__ == "__main__":
    
    args = parseInputs()
    frames_fd = open(args.file, 'rb') 	 	
   
    ncf15693.set_sample_freq(args.sample_freq)

    samples = frames_fd.read()
    iqLen = math.floor(len(samples)/8)*8
    print("Raw file loaded: %d bytes" %iqLen)

    chunkNbr = math.floor(iqLen / CHUNK_SIZE)
    print("Number of chunks: %d" %chunkNbr)

    # Split te processing on multiple threads regarding to NB_PROC variable:
    chunk_split = getChunkList(chunkNbr, NB_PROC)

    print("Split in %d processes, each %r chunks" %(NB_PROC, chunk_split))

    k = 0
    params = []

    # Set input parameters
    for chunk_len in chunk_split:
        params.append(process_param(samples, k, chunk_len))
        k += chunk_len

    # Parallel processing of SOF search
    frame_offsets = spawnSofProcess(params)

    print('Got %d raw SOF frames' %len(frame_offsets))
    print('%r ' %(frame_offsets))

    # Split the processing on multiple threads regarding to NB_PROC variable:
    chunk_split = getChunkList(len(frame_offsets), NB_PROC)

    k = 0
    params = []

    # Set input parameters
    for chunk_len in chunk_split:
        params.append(frame_offsets[k:k+chunk_len])
        k += chunk_len

    # Parallel processing for frame decoding
    data_frames = spawnDecodeFrameProcess(params)

    printData(data_frames)
        

    # plt.figure(1)
    # plt.plot(iqNorm, 'r')
    # plt.show()