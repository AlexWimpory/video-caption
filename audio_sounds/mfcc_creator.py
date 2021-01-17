import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment


def create_mfcc(path, sr=None, offset=0, duration=None):
    """Create an mfcc from the passed path"""
    wave = AudioSegment.from_wav(path)
    mono = wave.channels != 1
    if duration:
        y, sr = librosa.load(path, sr=sr, mono=mono, offset=offset, res_type='kaiser_fast')
    else:
        y, sr = librosa.load(path, sr=sr, mono=mono, offset=offset, duration=duration, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return mfcc


def mfcc_mean(mfcc):
    """MFCC not a reasonable input for model so convert matrix to vector"""
    return np.mean(mfcc.T, axis=0)


def visualise(mfcc, title):
    """Visualise an mfcc as a graph"""
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfcc, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title=title)
    plt.show()


if __name__ == '__main__':
    feature = create_mfcc('data/10_silence.wav')
    feature_mean = mfcc_mean(feature)
    visualise(feature, 'Silence')
