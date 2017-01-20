import math
from random import random
from matplotlib import pyplot as plt
import numpy as np


Fs = 8000
carrier_frequency = 200
data_size = 10
samples = Fs / carrier_frequency * data_size / 2
samples_per_symbol = Fs / carrier_frequency


data = np.arange(data_size)

# Data generation
for n in range(0, data_size):
    if random() > 0.5:
        data[n] = 1
    else:
        data[n] = 0

print data

# Split data into I/Q
I_data = np.arange(data_size / 2)
Q_data = np.arange(data_size / 2)


for n in range(0, data_size, 2):
    I_data[n / 2] = data[n]
for n in range(1, data_size, 2):
    Q_data[n / 2] = data[n]


# Generate I/Q carrier
I_carrier = np.arange(samples, dtype=np.float_)
Q_carrier = np.arange(samples, dtype=np.float_)
for n in range(samples):
    I_carrier[n] = math.sin(2 * math.pi * carrier_frequency * n / Fs)
    Q_carrier[n] = math.cos(2 * math.pi * carrier_frequency * n / Fs)

plt.figure(1)
plt.plot(np.arange(samples), I_carrier, np.arange(samples), Q_carrier)
plt.title('I carrier and Q carrier')
plt.draw()

# Modulate carrier with I/Q data
I_modulated = np.arange(samples, dtype=np.float_)
for n in range(len(I_data)):
    sign = 1 if I_data[n] == 1 else -1

    start = n * samples_per_symbol
    end = (n + 1) * samples_per_symbol

    I_modulated[start: end] = sign * I_carrier[start: end]

Q_modulated = np.arange(samples, dtype=np.float_)
for n in range(len(Q_data)):
    sign = 1 if Q_data[n] == 1 else -1

    start = n * samples_per_symbol
    end = (n + 1) * samples_per_symbol

    Q_modulated[start: end] = Q_carrier[start: end] * sign


# Add the I/Q data in time domain
transmit_signal = I_modulated + Q_modulated
plt.figure(2)
plt.plot(np.arange(samples), I_modulated, np.arange(samples), Q_modulated, np.arange(samples), transmit_signal)
plt.title('Transmit signal')
plt.draw()

# Send signal
receive_signal = transmit_signal

# Separate I/Q data and demodulate

I_demodulated = np.arange(len(I_data))
for n in range(len(I_data)):
    start = n * samples_per_symbol
    end = (n + 1) * samples_per_symbol

    temp = np.sum(I_modulated[start: end] * I_carrier[start: end])
    I_demodulated[n] = 0 if temp < 0 else 1


Q_demodulated = np.arange(len(Q_data))
for n in range(len(I_data)):
    start = n * samples_per_symbol
    end = (n + 1) * samples_per_symbol

    temp = np.sum(Q_modulated[start: end] * Q_carrier[start: end])
    Q_demodulated[n] = 0 if temp < 0 else 1

print "I demodulated:", I_demodulated, "actual data:", I_data
print "Q demodulated:", Q_demodulated, "actual data:", Q_data

plt.show()

