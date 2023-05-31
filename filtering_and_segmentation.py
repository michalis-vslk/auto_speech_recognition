from __future__ import print_function
import numpy as np
import librosa
import librosa.display
from scipy.signal import find_peaks

import plotting


# The function used in order to filter the original audio file
def filtering(signal, sr):
    # ---Input--- #
    # signal: The original audio signal
    # sr: The signal rate
    # ---Output--- #
    # foreground_audio: The filtered audio signal

    # Emphasizing the signal through pre-emphasis, in order to recover the foreground audio more accurately later on
    signal_emphasized = librosa.effects.preemphasis(signal)
    spectrogram_full, phase = librosa.magphase(librosa.stft(signal_emphasized))
    duration = librosa.get_duration(y=signal,sr=sr)
    idx = slice(*librosa.time_to_frames([0, duration], sr=sr))

    #plotting.plot_spectrogram(spectrogram_full, idx, phase)
    spectrogram_filter = librosa.decompose.nn_filter(spectrogram_full,
                                                     aggregate=np.nanmedian,
                                                     metric='cosine',
                                                     width=10)
                                                     #width=int(librosa.time_to_frames(2, sr=sr)))

    # The output of the filter shouldn't be greater than the input
    # if we assume signals are additive.  Taking the pointwise minimum
    # with the input spectrum forces this.
    spectrogram_filter = np.minimum(spectrogram_full, spectrogram_filter)

    margin_i, margin_v = 2, 10
    power = 2.1

    mask_i = librosa.util.softmask(spectrogram_filter,
                                   margin_i * (spectrogram_full - spectrogram_filter),
                                   power=power)

    mask_v = librosa.util.softmask(spectrogram_full - spectrogram_filter,
                                   margin_v * spectrogram_filter,
                                   power=power)

    # Once we have the masks, we simply multiply them with the input spectrum
    # to separate the components

    s_foreground = mask_v * spectrogram_full
    s_background = mask_i * spectrogram_full

    #plotting.plot_foreground_background_comparison(spectrogram_full, s_background, s_foreground,idx, sr,phase)

    temp = librosa.istft(s_foreground*phase)
    foreground_audio = librosa.effects.deemphasis(temp)
    return foreground_audio


# Segmenting the audio signal into words
def segmentation(foreground_audio):
    # ---Input--- #
    # foreground_audio: The filtered audio signal
    # ---Output--- #
    # word_matrix: The matrix containing each word's signal

    endpoints = librosa.effects.split(foreground_audio)

    words = []
    for start, end in endpoints:
        word = foreground_audio[start:end]
        words.append(word)

    # Saving the words in a matrix
    word_matrix = []
    for word in words:
        if 6500 < len(word) < 80000:
            word_matrix.append(word)

    return word_matrix


# Calculating the fundamental frequency of the speaker through all the isolated words
def fundamental_frequency_calculator(word_matrix, sr):
    # ---Input--- #
    # word_matrix: The matrix containing each word's signal
    # sr: The signal rate
    # ---Output--- #
    # avg_fundamental_freq: The fundamental frequency of the speaker

    avg_fundamental_freq = 0
    for i in range(len(word_matrix)):
        word_array = word_matrix[i]
        # Calculating autocorrelation of the word segment
        autocorrelation = librosa.autocorrelate(word_array)

        peaks, _ = find_peaks(autocorrelation)
        # Finding the second-highest peak (excluding the lag 0 peak)
        if len(peaks) > 1:
            period = peaks[np.argmax(autocorrelation[peaks[1:]])]
            fundamental_frequency = sr / period

            # If it's above average speech frequency, we use a cap of 500 Hz
            if fundamental_frequency > 500:
                fundamental_frequency = 500
            print("Word {}: Fundamental Frequency = {} Hz".format(i + 1, fundamental_frequency))
            avg_fundamental_freq += fundamental_frequency
        else:
            print("Word {}: Fundamental Frequency not found.".format(i + 1))

    avg_fundamental_freq = avg_fundamental_freq / len(word_matrix)

    return avg_fundamental_freq
