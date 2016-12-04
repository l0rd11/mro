import pyaudio
import scipy.io.wavfile as wav
import numpy as np
from matplotlib import pyplot as plt

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(np.fromstring(data, dtype=np.int16))

print("* done recording")
numpydata = np.hstack(frames)
stream.stop_stream()
stream.close()
p.terminate()
plt.plot(numpydata)
plt.show()


wav.write('out.wav',RATE,numpydata)
