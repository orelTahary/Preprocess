import numpy as np
NUM_BYTES_IN_FILE = 2**24
BLOCK_SIZE = 2**16
NUM_BLOCKS_IN_FILE = NUM_BYTES_IN_FILE / BLOCK_SIZE
EMPTY_BYTES_00 =  np.array([0, 0, 0, 0, 0, 0, 0, 0], 'uint8')
EMPTY_BYTES_FF = np.array([255, 255, 255, 255, 255, 255, 255, 255], 'uint8')
