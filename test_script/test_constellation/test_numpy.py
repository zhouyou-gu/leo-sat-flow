import os
import multiprocessing
import psutil

# Number of logical cores (threads)
print("Logical cores (threads):", os.cpu_count())
print("Logical cores (threads):", multiprocessing.cpu_count())
print("Logical cores (threads):", psutil.cpu_count(logical=True))

# Number of physical cores
print("Physical cores:", psutil.cpu_count(logical=False))