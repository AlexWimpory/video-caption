import librosa
import librosa.display
import matplotlib.pyplot as plt


def create_mfcc(path, sr=None, mono=False, offset=0, duration=None):
    """Create an mfcc from the passed path"""
    if duration:
        y, sr = librosa.load(path, sr=sr, mono=mono, offset=offset)
    else:
        y, sr = librosa.load(path, sr=sr, mono=mono, offset=offset, duration=duration)
    return librosa.feature.mfcc(y=y, sr=sr)


def visualise(mfccs, title):
    """Visualise an mfcc as a graph"""
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title=title)
    plt.show()


if __name__ == '__main__':
    thunder = create_mfcc('data/birds.wav')
    visualise(thunder, 'Birds')