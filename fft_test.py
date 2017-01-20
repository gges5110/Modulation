from matplotlib import pyplot as plt
import numpy as np

samples = 801
t = np.arange(samples)
signal = np.sin(t * np.pi / 20)

plt.figure(1)
plt.plot(signal)
plt.draw()

freq = np.fft.fftfreq(t.shape[-1])
sp = np.fft.fft(signal)
plt.figure(2)
plt.plot(freq, sp, 'ro')
plt.draw()

plt.show()
