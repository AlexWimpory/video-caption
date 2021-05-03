import matplotlib.pyplot as plt
import matplotlib.style as ms
import numpy as np

"""
Use Librosa to plot the:
* Time domain representation
* Mel power spectrogram
* MFCCs
* MFCC deltas
* MFCC delta-deltas
"""

ms.use('seaborn-muted')
import librosa.display

filename = 'gun'
y, sr = librosa.load(f'data\\{filename}.wav')

# Display the mel-scaled power (energy-squared) spectrogram
S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

# Convert to log scale (dB) useing the peak power as reference.
log_S = librosa.amplitude_to_db(S, ref=np.max)

# Extract the Mel-frequency cepstral coefficients (MFCCs)
mfcc = librosa.feature.mfcc(y=y, sr=sr)
print(mfcc.shape)

# Padding first and second deltas
delta_mfcc = librosa.feature.delta(mfcc)
delta2_mfcc = librosa.feature.delta(mfcc, order=2)

plt.figure(figsize=(12, 14))
plt.suptitle(f'{filename}', fontsize=24)

plt.subplot(5, 1, 1)
librosa.display.waveplot(y, sr=sr)
plt.title('Waveform', fontsize=16)

# Display the spectrogram on a mel scale
# sample rate and hop length parameters are used to render the time axis
plt.subplot(5, 1, 2)
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
plt.title('Mel Power Spectrogram', fontsize=16)
plt.colorbar(format='%+02.0f dB', pad=0.01)

# Plot MFCC as well as first and second delta
plt.subplot(5, 1, 3)
librosa.display.specshow(mfcc, x_axis='time')
plt.title('MFCCs', fontsize=16)
plt.ylabel('MFCC')
plt.colorbar(pad=0.01)

plt.subplot(5, 1, 4)
librosa.display.specshow(delta_mfcc, x_axis='time')
plt.ylabel('MFCC-$\Delta$')
plt.colorbar(pad=0.01)

plt.subplot(5, 1, 5)
librosa.display.specshow(delta2_mfcc, sr=sr, x_axis='time')
plt.ylabel('MFCC-$\Delta^2$')
plt.colorbar(pad=0.01)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
plt.savefig(f'plots/{filename}.png')

# Display graphs in Pycharm
plt.show()

# Stacking these 3 tables together into one matrix
M = np.vstack([mfcc, delta_mfcc, delta2_mfcc])


