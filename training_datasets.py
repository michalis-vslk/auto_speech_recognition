import numpy as np
import librosa.display
import filtering_and_segmentation
import os


def training():
    t_datasets = []
    # assigning the proper directory
    directory = 'training dataset'

    # iterating over files in that directory
    for filename in os.scandir(directory):
        if filename.is_file():
            signal, sr = librosa.load(filename.path)
            foreground_audio = filtering_and_segmentation.filtering(signal, sr)
            word_matrix = filtering_and_segmentation.segmentation(foreground_audio)
            t_datasets.append(word_matrix[0])

    return t_datasets

# Compute spectrogram for the second signal
    # signal_2, sr_2 = librosa.load('3_s1.wav')
    # spectrogram_2 = np.abs(librosa.stft(signal_2))


def cost_calculator(word_matrix, t_dataset_matrix):
    cost_matrix_new = np.empty((len(word_matrix), len(t_dataset_matrix)))
    # cost_matrix_new = [None] * len(word_matrix) * len(t_dataset_matrix)
    min_cost_indexes = []
    # training_digits_coefficients = []
    for i in range(len(word_matrix)):
        spectrogram_1 = np.abs(librosa.stft(word_matrix[i], win_length=512, hop_length=256))
        spec1_mag = librosa.amplitude_to_db(spectrogram_1)
        for j in range(len(t_dataset_matrix)):
            # we increased the window length of the sort fourier transform because it works better
            # we check the amplitude in dB of the spectrogram/coefficients
            # we save the magnitudes of each spectrogram's coefficients, without the phase(hence the abs)
            spectrogram_2 = np.abs(librosa.stft(t_dataset_matrix[j], win_length=512, hop_length=256))
            spec2_mag = librosa.amplitude_to_db(spectrogram_2)

            # the best match for the word has the least cost (which is in the bottom right corner), we checked for the other wav files
            cost_matrix, wp = librosa.sequence.dtw(X=spec1_mag, Y=spec2_mag)
            # make a list with minimum cost of each digit
            cost_matrix_new[i][j] = cost_matrix[-1, -1]
            '''# we save the digit that is most similar to our word
            training_digits_coefficients.append(
                spec2_mag)'''
        # index of MINIMUM COST
        # index_min_cost = cost_matrix_new.index(min(cost_matrix_new[i]))
        index_min_cost = np.argmin(cost_matrix_new[i])
        min_cost_indexes.append(index_min_cost)

    print(min_cost_indexes)
    print(cost_matrix_new)
    return cost_matrix_new

'''spectrogram_2 = np.abs(librosa.stft(word_matrix2[0], win_length=512, hop_length=256))  
# we increaced the window length of the sort fourier transform because it works better?!

# we check the amplitude in dp of the spectogram/coefficients(not sure if its correct


# we save the magnitudes of each spectogram's coefficients,without the phase(hence the abs)
spec2_mag=librosa.amplitude_to_db(spectrogram_2)'''