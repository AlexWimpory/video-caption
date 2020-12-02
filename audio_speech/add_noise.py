import math
import numpy as np
import librosa
import subprocess
from scipy.io.wavfile import write

'''
Signal to noise ratio (SNR) can be defined as
SNR = 20*log(RMS_signal/RMS_noise)
Where:  RMS_signal is the RMS value of signal
        RMS_noise is that of noise.
        Log is the logarithm of 10
****Additive White Gaussian Noise (AWGN)****
 - This kind of noise can be added (arithmetic element-wise addition) to the signal
 - Mean value is zero (randomly sampled from a Gaussian distribution with mean value of zero)
 - Contains all the frequency components in an equal manner
****Real World Noise****
 - An audio file which can be overlapped the signal as noise
 - Frequency components will depend on the sound used
'''


def get_white_noise(signal, snr):
    """Given a signal and desired SNR, this gives the required AWGN
    that should be added to the signal to get the desired SNR in dB"""
    RMS_s = math.sqrt(np.mean(signal ** 2))
    RMS_n = math.sqrt(RMS_s ** 2 / (pow(10, snr / 10)))
    STD_n = RMS_n
    noise = np.random.normal(0, STD_n, signal.shape[0])
    return noise


def get_noise_from_sound(signal, noise, snr):
    """Given a signal, noise (audio) and desired SNR,
    this gives the noise (scaled version of noise input) that gives the desired SNR"""
    RMS_s = math.sqrt(np.mean(signal ** 2))
    RMS_n = math.sqrt(RMS_s ** 2 / (pow(10, snr / 10)))
    RMS_n_current = math.sqrt(np.mean(noise ** 2))
    noise = noise * (RMS_n / RMS_n_current)
    return noise


def to_polar(complex_ar):
    """convert complex np array to polar arrays (2 apprays; abs and angle)"""
    return np.abs(complex_ar), np.angle(complex_ar)


def add_awgn(signal_file, snr):
    """Add AWGN to a .wav file"""
    signal, sr = librosa.load(signal_file)
    signal = np.interp(signal, (signal.min(), signal.max()), (-1, 1))
    noise = get_white_noise(signal, snr=snr)
    signal_noise = signal + noise
    write('data/speech_noisy_broken.wav', sr, signal_noise.astype(np.float32))
    subprocess.call(['ffmpeg', '-y', '-i', 'data/speech_noisy_broken.wav', '-ar', '44100', '-ac', '1',
                     '-acodec', 'pcm_s16le', 'data/speech_noisy_fixed.wav'])


def add_real_world_noise(signal_file, noise_file, snr):
    """Combines 2 .wav files with the noise file being modified by the SNR"""
    signal, sr = librosa.load(signal_file)
    signal = np.interp(signal, (signal.min(), signal.max()), (-1, 1))
    noise, sr = librosa.load(noise_file)
    noise = np.interp(noise, (noise.min(), noise.max()), (-1, 1))
    # crop noise if its longer than signal
    if len(noise) > len(signal):
        noise = noise[0:len(signal)]
    noise = get_noise_from_sound(signal, noise, snr=snr)
    signal_noise = signal + noise
    write('data/speech_noisy_broken.wav', sr, signal_noise.astype(np.float32))
    subprocess.call(['ffmpeg', '-y', '-i', 'data/speech_noisy_broken.wav', '-ar', '44100', '-ac', '1',
                     '-acodec', 'pcm_s16le', 'data/speech_noisy_fixed.wav'])


if __name__ == '__main__':
    add_awgn('D:\\Audio Speech\\LibriSpeech\\train-clean-100\\19\\198\\19-198-0001.wav', 10)
    # add_real_world_noise('data/19-198-0001.wav', 'data/Welcome.wav', 10)
