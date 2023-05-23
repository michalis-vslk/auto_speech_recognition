
import os

from utils import *

root = ""
while True:
    # Get the sound file path from the user
    file_path = input(TXT_INPUT_FILE)

    # Check if file exists
    if os.path.exists(file_path) is False:
        print(TXT_FILE_NOT_FOUND + str(file_path))
        print()
        continue  # Return to the start of the loop

    # Check if file is mp3 or wav
    root, extension = os.path.splitext(file_path)
    if extension not in AUDIO_WAV_EXTENSION:
        print(TXT_FILE_WRONG_EXTENSION + str(file_path))
        print()
        continue  # Return to the start of the loop

    break

# if plots directory doesn't exists, create is so we save our plots.
if os.path.exists(DIRECTORY_PLOTS) is False:
    os.mkdir(DIRECTORY_PLOTS)

# Load the file from path, then get the signal and sample rate.
signal, sr = librosa.load(file_path, sr=DEFAULT_SAMPLE_RATE)

# === Start Pre-Processing ===
pre_proceed_signal = pre_processing(signal, os.path.basename(root))

print("Finding digits...")

# === Start digit segmentation ===
samples = digits_segmentation(pre_proceed_signal)

# === Feature extraction & word recognition ===
digits_array = valid_digits(pre_proceed_signal, samples)

# === Get training samples in signal form ===
dataset_training_signals = get_training_samples_signal()

# === Display words a list of words found ===
recognized_digits = recognition(digits=digits_array,
                                signal_data=pre_proceed_signal,
                                dataset=dataset_training_signals)

print(TXT_DIGITS_FOUND.format(len(digits_array)))
# Prints the list that contains all the words found and separates each word
# with a ", " excluding the last one.
print()
print(TXT_DIGITS_RECOGNIZED)
print(", ".join([str(i) for i in digits_array]))


from termcolor import colored


# region Lang
# Console
TXT_INPUT_FILE = "Enter sound file path:\n"
TXT_FILE_NOT_FOUND = colored("File not found at:\n", "red")
TXT_FILE_WRONG_EXTENSION = colored("Wrong file extension. Please try again!", "red")
TXT_PRE_PROCESSING_STATISTICS = "PRE-PROCESSING STATISTICS:"
TXT_AUDIO_ORIGINAL_DURATION_FORMAT = "- Original Audio Duration: {} sec."
TXT_AUDIO_FILTERED_DURATION_FORMAT = "- Audio (Filtered) Duration: {} sec."
TXT_ORIGINAL_AUDIO_SAMPLE_RATE = "- Sample Rate: {}"
TXT_ZCR_AVERAGE = "- Average ZCR: {}"
TXT_DIGITS_FOUND = "[!] Total Digits Found: {}"
TXT_DIGITS_RECOGNIZED = "Digits Recognized:"
TXT_LINE = "==============================================="
# Plot
TXT_AMPLITUDE = "Amplitude"
TXT_TIME = "Time (s)"
TXT_FREQUENCY = "Frequency (Hz)"
TXT_ORIGINAL_SIGNAL = "Original Signal"
TXT_PRE_EMPHASIZED_SIGNAL = "Pre-Emphasized Signal"
TXT_DECIBELS = "Decibels (dB)"
TXT_MEL = "Mel Scale (Mel)"
TXT_ORIGINAL = "Original"
TXT_FILTERED = "Filtered"
TXT_STE = "STE"
TXT_ZERO_CROSSING_RATE = "Zero-Crossing Rate"
TXT_SHORT_TIME_ENERGY = "Short-Time Energy"
# endregion


# region Variables
DIRECTORY_PLOTS = ".\\data\\plots"
# Remove signal part if dB is less than 40
TOP_DB = 40
DEFAULT_SAMPLE_RATE = 16000
AUDIO_WAV_EXTENSION = ".wav"
# window length in sec. Default is 0.03.
WINDOW_LENGTH = 0.03
# step between successive windows in sec. Default is 0.01.
WINDOW_HOP = 0.01
FRAME_LENGTH = round(WINDOW_LENGTH * DEFAULT_SAMPLE_RATE)
DATASET_SPLIT_LABELS = ["s1", "s2", "s3"]
# endregion





import librosa.display
import noisereduce as nr
import scipy.signal as sg
import soundfile as sf

from plots import *


def pre_processing(signal_data, file_name):
    # === Pre-Emphasis ===
    # Parameters:
    #   signal_data: A nparray with the original signal.
    #   file_name: A string that contains the file name.

    signal_emphasized = librosa.effects.preemphasis(signal_data)

    # === Filtering ===
    # Remove the background noise from the audio file.
    signal_reduced_noise = remove_noise(signal_data)

    # Remove the silent parts of the audio that are less than 40dB
    signal_filtered, _ = librosa.effects.trim(signal_reduced_noise, TOP_DB)

    signal_zcr = librosa.feature.zero_crossing_rate(signal_filtered)
    zcr_average = np.mean(signal_zcr)

    signal_short_time_energy = calculate_short_time_energy(signal_filtered)

    # Show plots
    show_plot_emphasized(signal_data, signal_emphasized)
    show_plots_compare_two_signals(signal_data, signal_reduced_noise)
    show_plot_zcr(signal_zcr)
    show_plot_short_time_energy(signal_filtered, signal_short_time_energy)

    # Exporting the filtered audio file.
    filtered_file_path = ".\\data\\samples\\" + file_name + "_filtered.wav"
    sf.write(filtered_file_path, signal_filtered, DEFAULT_SAMPLE_RATE)

    # Print statistics
    print(TXT_LINE, "\n")
    print(TXT_PRE_PROCESSING_STATISTICS)
    print(TXT_ORIGINAL_AUDIO_SAMPLE_RATE.format(DEFAULT_SAMPLE_RATE))
    print(TXT_AUDIO_ORIGINAL_DURATION_FORMAT.format(
        round(librosa.get_duration(signal_data, sr=DEFAULT_SAMPLE_RATE), 2))
    )
    print(TXT_AUDIO_FILTERED_DURATION_FORMAT.format(
        round(librosa.get_duration(signal_filtered, sr=DEFAULT_SAMPLE_RATE), 2))
    )
    print(TXT_ZCR_AVERAGE.format(zcr_average), "\n")
    print(TXT_LINE)

    return signal_filtered


def remove_noise(signal_data):
    # Parameters:
    #   signal_data: A nparray with the original signal.

    reduced_noise = nr.reduce_noise(audio_clip=signal_data,
                                    noise_clip=signal_data)

    return reduced_noise


def calculate_short_time_energy(signal_data):
    # Parameters:
    #   signal_data: A nparray with the original signal.

    signal = np.array(signal_data, dtype=float)
    win = sg.get_window("hamming", 301)

    if isinstance(win, str):
        win = sg.get_window(win, max(1, len(signal) // 8))
    win = win / len(win)

    signal_short_time_energy = sg.convolve(signal ** 2, win ** 2, mode="same")

    return signal_short_time_energy


def digits_segmentation(signal_nparray):
    # Parameters:
    #   signal_data: A nparray with the filtered signal.

    # We reverse the signal nparray.
    signal_reverse = signal_nparray[::-1]

    frames = librosa.onset.onset_detect(signal_nparray, sr=DEFAULT_SAMPLE_RATE, hop_length=FRAME_LENGTH)
    times = librosa.frames_to_time(frames, sr=DEFAULT_SAMPLE_RATE, hop_length=FRAME_LENGTH)
    samples = librosa.frames_to_samples(frames, FRAME_LENGTH)

    frames_reverse = librosa.onset.onset_detect(signal_reverse, sr=DEFAULT_SAMPLE_RATE, hop_length=FRAME_LENGTH)
    times_reverse = librosa.frames_to_time(frames_reverse, sr=DEFAULT_SAMPLE_RATE, hop_length=FRAME_LENGTH)

    for i in range(0, len(times_reverse) - 1):
        times_reverse[i] = WINDOW_LENGTH - times_reverse[i]
        i += 1

    times_reverse = sorted(times_reverse)

    i = 0
    while i < len(times_reverse) - 1:
        if times_reverse[i + 1] - times_reverse[i] < 1:
            times_reverse = np.delete(times_reverse, i)
            i -= 1
        i += 1

    i = 0
    while i < len(times) - 1:
        if times[i + 1] - times[i] < 1:
            times = np.delete(times, i + 1)
            frames = np.delete(frames, i + 1)
            samples = np.delete(samples, i + 1)
            i = i - 1
        i = i + 1

    merged_times = [*times, *times_reverse]
    merged_times = sorted(merged_times)

    samples = librosa.time_to_samples(merged_times, sr=DEFAULT_SAMPLE_RATE)

    return samples


def valid_digits(signal_data, samples):
    # Parameters:
    #   signal_data: An nparray with the signal.
    #   samples: An ndarray that contains integers.

    count_digits = 0
    digit = {}

    for i in range(0, len(samples), 2):
        if len(samples) % 2 == 1 and i == len(samples) - 1:
            digit[count_digits] = signal_data[samples[i - 1]:samples[i]]
        else:
            digit[count_digits] = signal_data[samples[i]:samples[i + 1]]
        count_digits += 1

    return digit


def recognition(digits, signal_data, dataset):
    # === Recognition of Digits ===
    # Parameters:
    #   digits: An array containing integer digits.
    #   signal_data: A nparray with the original signal for comparison.
    #   dataset: An array with all training signals.

    # Init an array that will contain our recognized digits in string.
    recognized_digits_array = []
    for digit in digits:
        cost_matrix_new = []
        mfccs = []

        mfcc_digit = librosa.feature.mfcc(y=digit,
                                          S=signal_data,
                                          sr=DEFAULT_SAMPLE_RATE,
                                          hop_length=FRAME_LENGTH,
                                          n_mfcc=13)
        mfcc_digit_mag = librosa.amplitude_to_db(abs(mfcc_digit))

        # 0-9 from training set
        for i in range(len(dataset)):
            # We basically filter the training dataset as well.
            dataset[i] = filter_dataset_signal(dataset[i].astype(np.float))

            # MFCC for each digit from the training set
            mfcc = librosa.feature.mfcc(y=dataset[i],
                                        S=signal_data,
                                        sr=DEFAULT_SAMPLE_RATE,
                                        hop_length=80,
                                        n_mfcc=13)

            # logarithm of the features ADDED
            mfcc_mag = librosa.amplitude_to_db(abs(mfcc))

            # apply dtw
            cost_matrix, wp = librosa.sequence.dtw(X=mfcc_digit_mag, Y=mfcc_mag)

            # make a list with minimum cost of each digit
            cost_matrix_new.append(cost_matrix[-1, -1])
            mfccs.append(mfcc_mag)

        # index of MINIMUM COST
        index_min_cost = cost_matrix_new.index(min(cost_matrix_new))

        recognized_digits_array.append(DATASET_SPLIT_LABELS[index_min_cost])

        for i in dataset:
            show_mel_spectrogram(dataset[i], DATASET_SPLIT_LABELS[index_min_cost])

    return recognized_digits_array


def get_training_samples_signal():
    # Initialize an array to append the signals of the training samples.
    training_samples_signals = {}

    index = 0
    # Loop between a range of 0-9, 0 in range(10) is 0 to 9 in python.
    for i in range(10):
        # Loop between the labels, s1 means sample1 and so on.
        for name in DATASET_SPLIT_LABELS:
            # Load the signal and add it to our array.
            training_samples_signals[index], _ = librosa.load(".\\data\\training\\"
                                                              + str(i)
                                                              + "_"
                                                              + name
                                                              + AUDIO_WAV_EXTENSION,
                                                              sr=DEFAULT_SAMPLE_RATE)

    index += 1

    return training_samples_signals


def filter_dataset_signal(signal_data):
    # === Filtering ===
    # Parameters:
    #   signal_data: A nparray with the signal.

    # Remove the background noise from the audio file.
    signal_reduced_noise = remove_noise(signal_data)

    # Remove the silent parts of the audio that are less than 40dB
    signal_filtered, _ = librosa.effects.trim(signal_reduced_noise, TOP_DB)

    return signal_filtered



import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt

from constants import *


def show_plots_compare_two_signals(signal_data, signal_data_reduced):
    fig, ax = plt.subplots(nrows=2, sharex="all", sharey="all", constrained_layout=True)

    ax[0].set(title="Original Audio Waveform Graph", xlabel=TXT_TIME, ylabel=TXT_AMPLITUDE)
    ax[1].set(title="Audio Waveform Graph", xlabel=TXT_TIME, ylabel=TXT_AMPLITUDE)

    # Apply grid
    ax[0].grid()
    ax[1].grid()

    librosa.display.waveshow(signal_data, sr=DEFAULT_SAMPLE_RATE, ax=ax[0], label=TXT_ORIGINAL)
    librosa.display.waveshow(signal_data, sr=DEFAULT_SAMPLE_RATE, ax=ax[1], label=TXT_ORIGINAL)
    librosa.display.waveshow(signal_data_reduced, sr=DEFAULT_SAMPLE_RATE, ax=ax[1], label=TXT_FILTERED)

    # Set legend
    ax[1].legend()

    # Show plot
    plt.show()

    # Save plot to directory
    fig.savefig(".\\data\\plots\\original_and_filtered_audio.png")


def show_plot_emphasized(signal_data_orig, signal_data_emphasized):
    s_orig = librosa.amplitude_to_db(np.abs(librosa.stft(signal_data_orig)), ref=np.max, top_db=None)
    s_pre_emphasized = librosa.amplitude_to_db(np.abs(librosa.stft(signal_data_emphasized)), ref=np.max, top_db=None)

    fig, ax = plt.subplots(nrows=2, sharex="all", sharey="all", constrained_layout=True)
    librosa.display.specshow(s_orig, y_axis='log', x_axis='time', ax=ax[0])

    img = librosa.display.specshow(s_pre_emphasized, y_axis='log', x_axis='time', ax=ax[1])
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    ax[0].label_outer()

    # Set title
    ax[0].set(title=TXT_ORIGINAL_SIGNAL, xlabel=TXT_TIME, ylabel=TXT_FREQUENCY)
    ax[1].set(title=TXT_PRE_EMPHASIZED_SIGNAL, xlabel=TXT_TIME, ylabel=TXT_FREQUENCY)

    # Show plot
    plt.show()

    # Save plot to directory
    fig.savefig(".\\data\\plots\\original_and_pre_emphasis.png")


def show_plot_zcr(signal_data_zcr):
    plt.plot(signal_data_zcr[0])

    # Set title
    plt.title(TXT_ZERO_CROSSING_RATE)
    # Apply grid
    plt.grid()

    # Save plot to directory
    plt.savefig(".\\data\\plots\\zero_crossing_rate.png")

    # Zooming in
    plt.figure(figsize=(14, 5))

    # Show plot
    plt.show()


def show_plot_short_time_energy(signal_data_original, signal_data_ste):
    time = np.arange(len(signal_data_original)) * (1.0 / DEFAULT_SAMPLE_RATE)

    plt.figure()
    plt.plot(time, signal_data_ste, 'm', linewidth=2)
    plt.legend([TXT_ORIGINAL, TXT_STE])
    plt.title(TXT_SHORT_TIME_ENERGY)
    plt.xlabel(TXT_TIME)

    # Save plot to directory
    plt.savefig(".\\data\\plots\\short_time_energy.png")

    # Show the plot
    plt.show()


def show_mel_spectrogram(signal_nparray, num):
    # Calculating the Short-Time Fourier Transform of signal
    spectrogram = librosa.stft(signal_nparray)
    # Using the mel-scale instead of raw frequency
    spectrogram_mag, _ = librosa.magphase(spectrogram)
    mel_scale_spectrogram = librosa.feature.melspectrogram(S=spectrogram_mag,
                                                           sr=DEFAULT_SAMPLE_RATE)
    # use the decibel scale to get the final Mel Spectrogram
    mel_spectrogram = librosa.amplitude_to_db(mel_scale_spectrogram, ref=np.min)
    librosa.display.specshow(mel_spectrogram,
                             sr=DEFAULT_SAMPLE_RATE,
                             x_axis='time',
                             y_axis='mel')
    plt.colorbar(format="%+2.0f dB")

    # Zooming in
    plt.figure(figsize=(14, 5))

    plt.show()


