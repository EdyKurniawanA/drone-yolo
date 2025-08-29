import GPUtil

gpus = GPUtil.getGPUs()
if not gpus:
    print("No GPU detected by GPUtil")
else:
    for gpu in gpus:
        print(f"GPU: {gpu.name}")
        print(f"  Load: {gpu.load*100:.1f}%")
        print(f"  Free Memory: {gpu.memoryFree}MB")
        print(f"  Used Memory: {gpu.memoryUsed}MB")
        print(f"  Total Memory: {gpu.memoryTotal}MB")
        print(f"  Temperature: {gpu.temperature} °C")
