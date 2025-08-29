import pynmea2

sample = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47"
msg = pynmea2.parse(sample)
print(msg.latitude, msg.longitude)
