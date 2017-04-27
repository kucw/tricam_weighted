from pynvml import *
import time
import json
import psutil

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(handle)
#print json.dumps({'total_memory':info.total/1048576000.0, 'free_memory':info.free/1048576000.0, 'used_memory':info.used/1048576000.0})
util = nvmlDeviceGetUtilizationRates(handle)
gpu_util = util.gpu
gpu_mem_util = util.memory

cpu_util = psutil.cpu_percent(interval=0.1)
cpu_mem_util = psutil.virtual_memory()[2]

print json.dumps({'gpu_utilization':gpu_util, 'gpu_memory_utilization':gpu_mem_util, 'cpu_utilization': cpu_util, 'cpu_memory_utilization': cpu_mem_util})
