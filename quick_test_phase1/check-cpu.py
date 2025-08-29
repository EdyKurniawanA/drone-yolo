import psutil

# CPU usage %
print("CPU Usage:", psutil.cpu_percent(interval=1), "%")

# RAM usage
mem = psutil.virtual_memory()
print("RAM Used:", mem.used // (1024**2), "MB of", mem.total // (1024**2), "MB")

# Disk usage
disk = psutil.disk_usage("/")
print("Disk Used:", disk.used // (1024**3), "GB of", disk.total // (1024**3), "GB")

# Battery status (if laptop)
if psutil.sensors_battery():
    battery = psutil.sensors_battery()
    print(
        f"Battery: {battery.percent}% {'Charging' if battery.power_plugged else 'Not Charging'}"
    )
