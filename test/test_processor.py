# @title Install and import dependencies

# %tensorflow_version 2.x
# pip install -qU ddsp

# Ignore a bunch of deprecation warnings
import warnings

warnings.filterwarnings("ignore")

import ddsp
import ddsp.training
from ddsp.colab.colab_utils import play, specplot, DEFAULT_SAMPLE_RATE
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# IMPORTS MINE:
import sounddevice as sd
import time as stime


sample_rate = DEFAULT_SAMPLE_RATE  # 16000



# --------------------------------------------------------------------------------------------------------------------------------
### my Functions:
def playsound(inputsound):
    sd.play(inputsound[0], DEFAULT_SAMPLE_RATE)
    stime.sleep(len(inputsound[0])/DEFAULT_SAMPLE_RATE)
    sd.stop()


# --------------------------------------------------------------------------------------------------------------------------------

n_frames = 1000
hop_size = 64
n_samples = n_frames * hop_size

# Create a synthesizer object.
additive_synth = ddsp.synths.Additive(
    n_samples=n_samples, sample_rate=sample_rate, name="additive_synth"
)

# --------------------------------------------------------------------------------------------------------------------------------

# Generate some arbitrary inputs.

# Amplitude [batch, n_frames, 1].
# Make amplitude linearly decay over time.
amps = np.linspace(1.0, -3.0, n_frames)
amps = amps[np.newaxis, :, np.newaxis]

# Harmonic Distribution [batch, n_frames, n_harmonics].
# Make harmonics decrease linearly with frequency.
n_harmonics = 30
harmonic_distribution = (
    np.linspace(-2.0, 2.0, n_frames)[:, np.newaxis]
    + np.linspace(3.0, -3.0, n_harmonics)[np.newaxis, :]
)
harmonic_distribution = harmonic_distribution[np.newaxis, :, :]

# Fundamental frequency in Hz [batch, n_frames, 1].
f0_hz = 440.0 * np.ones([1, n_frames, 1], dtype=np.float32)

# --------------------------------------------------------------------------------------------------------------------------------

# Plot it!
time = np.linspace(0, n_samples / sample_rate, n_frames)

plot_0 = plt.figure(figsize=(18, 4))
plt.subplot(131)
plt.plot(time, amps[0, :, 0])
plt.xticks([0, 1, 2, 3, 4])
plt.title("Amplitude")

plt.subplot(132)
plt.plot(time, harmonic_distribution[0, :, :])
plt.xticks([0, 1, 2, 3, 4])
plt.title("Harmonic Distribution")

plt.subplot(133)
plt.plot(time, f0_hz[0, :, 0])
plt.xticks([0, 1, 2, 3, 4])
_ = plt.title("Fundamental Frequency")
plot_0.show()

# --------------------------------------------------------------------------------------------------------------------------------
controls = additive_synth.get_controls(amps, harmonic_distribution, f0_hz)
print(controls.keys())


# Now let's see what they look like...
time = np.linspace(0, n_samples / sample_rate, n_frames)

plot_1 = plt.figure(figsize=(18, 4))
plt.subplot(131)
plt.plot(time, controls["amplitudes"][0, :, 0])
plt.xticks([0, 1, 2, 3, 4])
plt.title("Amplitude")

plt.subplot(132)
plt.plot(time, controls["harmonic_distribution"][0, :, :])
plt.xticks([0, 1, 2, 3, 4])
plt.title("Harmonic Distribution")

plt.subplot(133)
plt.plot(time, controls["f0_hz"][0, :, 0])
plt.xticks([0, 1, 2, 3, 4])
_ = plt.title("Fundamental Frequency")

plot_1.show()
# --------------------------------------------------------------------------------------------------------------------------------


x = tf.linspace(-10.0, 10.0, 1000)
y = ddsp.core.exp_sigmoid(x)

plot_2 = plt.figure(figsize=(18, 4))
plt.subplot(121)
plt.plot(x, y)

plt.subplot(122)
_ = plt.semilogy(x, y)
plot_2.show()

# --------------------------------------------------------------------------------------------------------------------------------
audio = additive_synth.get_signal(**controls)
playsound(audio)


# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------

end = input("Enter to close")
