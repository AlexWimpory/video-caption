import math
import os
import random
import tempfile
import numpy as np
import librosa
import subprocess
from scipy.io.wavfile import write
from audio_utils.utils.file_utils import split_base_and_extension

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


def add_awgn(signal_file, snr, output_path, length):
    """Add AWGN to a .wav file"""
    split_signal_file = split_base_and_extension(signal_file)
    temp_broken_file_name = os.path.join(output_path, 'temp_broken' + split_signal_file[1])
    temp_file_name = tempfile.mktemp(dir=output_path, prefix=split_signal_file[0], suffix=split_signal_file[1])
    signal, sr = librosa.load(signal_file)
    signal = np.interp(signal, (signal.min(), signal.max()), (-1, 1))
    noise = get_white_noise(signal, snr=snr)
    signal_noise = signal + noise
    write(temp_broken_file_name, sr, signal_noise.astype(np.float32))
    subprocess.call(
        ['ffmpeg', '-y', '-ss', '00:00:00', '-t', str(length), '-i', temp_broken_file_name,
         '-ar', '44100', '-ac', '1', '-acodec', 'pcm_s16le', temp_file_name])
    return temp_file_name


if __name__ == '__main__':
    with open('../../random_data/processing/data/silence.csv', 'w') as fout:
        for i in range(0, 800):
            random_snr = random.randint(60, 100)
            length = random.randint(2, 6)
            file_name = add_awgn('D:\\Audio Features\\UrbanSound8K\\UrbanSound8K\\audio\\fold11\\10_silence.wav',
                                 random_snr,
                                 'D:\\Audio Features\\UrbanSound8K\\UrbanSound8K\\audio\\fold11', length)
            ground_truth_file_name = os.path.splitext(os.path.basename(file_name))[0]
            fout.write(f"{ground_truth_file_name},['silence']\n")
