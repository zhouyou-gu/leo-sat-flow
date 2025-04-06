import os
import multiprocessing
import psutil

# Number of logical cores (threads)
print("Logical cores (threads):", os.cpu_count())
print("Logical cores (threads):", multiprocessing.cpu_count())
print("Logical cores (threads):", psutil.cpu_count(logical=True))

# Number of physical cores
print("Physical cores:", psutil.cpu_count(logical=False))


import vispy

print(vispy.sys_info())

from numba import cuda

# Print summary of detected CUDA devices
cuda.detect()

# Get the current device
device = cuda.get_current_device()

# Print out device details
print("Device Name:", device.name)
print("Compute Capability:", device.compute_capability)
print("Max Threads per Block:", device.MAX_THREADS_PER_BLOCK)
meminfo = cuda.current_context().get_memory_info()
print("Total Memory (bytes):", meminfo.total)
print("Free Memory (bytes):", meminfo.free)

import numpy as np
import cupy as cp
a = np.zeros((10,10))
d = cp.asarray(a)
d += 1.  ## fails


import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device())