import platform
import psutil
import GPUtil

# Function to get GPU information
def get_gpu_info():
    try:
        gpu = GPUtil.getGPUs()[0]
        return {
            'GPU Name': gpu.name,
            'GPU Driver': gpu.driver,
            'GPU Memory Total (MB)': gpu.memoryTotal,
            'GPU Memory Free (MB)': gpu.memoryFree,
            'GPU Memory Used (MB)': gpu.memoryUsed,
            'GPU Utilization (%)': gpu.load * 100
        }
    except Exception as e:
        return {'Error': str(e)}

# Function to get CPU information
def get_cpu_info():
    return {
        'Number of CPU Cores': psutil.cpu_count(logical=False),
        'Number of Logical CPUs': psutil.cpu_count(logical=True),
        'CPU Usage (%)': psutil.cpu_percent(interval=1)
    }

# Function to get memory (RAM) and disk information
def get_memory_disk_info():
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    return {
        'Memory Total (MB)': memory.total,
        'Memory Used (MB)': memory.used,
        'Memory Free (MB)': memory.free,
        'Disk Total (GB)': disk.total / (2**30),  # Convert bytes to GB
        'Disk Used (GB)': disk.used / (2**30),
        'Disk Free (GB)': disk.free / (2**30)
    }

# Get system information
system_info = {
    'System': platform.system(),
    'Node': platform.node(),
    'Release': platform.release(),
    'Version': platform.version(),
    'Machine': platform.machine(),
    'Processor': platform.processor(),
    **get_gpu_info(),
    **get_cpu_info(),
    **get_memory_disk_info()
}

# Print system information
for key, value in system_info.items():
    print(f'{key}: {value}')
