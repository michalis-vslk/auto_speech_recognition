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
    ct = np.empty((len(word_matrix), 10))
    # cost_matrix_new = [None] * len(word_matrix) * len(t_dataset_matrix)
    min_cost_indexes = []
    max_cos_indexes = []
    average_cos_similarity = np.empty((len(word_matrix), len(t_dataset_matrix)))
    # training_digits_coefficients = []

    for i in range(len(word_matrix)):
        spectrogram_1 = np.abs(librosa.stft(word_matrix[i], win_length=128, hop_length=64))  #128 64
        spec1_mag = librosa.amplitude_to_db(spectrogram_1)
        for j in range(len(t_dataset_matrix)):
            # we increased the window length of the sort fourier transform because it works better
            # we check the amplitude in dB of the spectrogram/coefficients
            # we save the magnitudes of each spectrogram's coefficients, without the phase(hence the abs)
            spectrogram_2 = np.abs(librosa.stft(t_dataset_matrix[j], win_length=128, hop_length=64)) #128 64
            spec2_mag = librosa.amplitude_to_db(spectrogram_2)
            # the best match for the word has the least cost (which is in the bottom right corner), we checked for the other wav files
            cost_matrix, wp = librosa.sequence.dtw(X=spec1_mag, Y=spec2_mag)
            # make a list with minimum cost of each digit
            cost_matrix_new[i][j] = cost_matrix[-1, -1]

            window_size = 128  # Adjust the window size as needed #128 64
            hop_length = 64
            spectrogram_1_windows = librosa.util.frame(spec1_mag, frame_length=window_size, hop_length=hop_length)
            spectrogram_2_windows = librosa.util.frame(spec2_mag, frame_length=window_size, hop_length=hop_length)
            cosine_similarities = []
            for k in range(spectrogram_1_windows.shape[1]):
                vector_1 = spectrogram_1_windows[:, k].flatten()
                vector_2 = spectrogram_2_windows[:, k].flatten()
                if len(vector_2) > len(vector_1):
                    size_diff = vector_2.size - vector_1.size
                    vector_1 = np.concatenate((vector_1, np.zeros(size_diff)))
                else:
                    size_diff = vector_1.size - vector_2.size
                    vector_2 = np.concatenate((vector_2, np.zeros(size_diff)))
                cos_similarity = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
                cosine_similarities.append(cos_similarity)

            # Aggregate the cosine similarities
            average_cos_similarity[i][j] = np.nanmean(cosine_similarities)
            #print("Average Cosine Similarity:", average_cos_similarity[i])



        # index of MINIMUM COST
        #1st way
        # index_min_cost = np.argmin(cost_matrix_new[i])
        # min_cost_indexes.append(index_min_cost)
        #2nd way
        '''
        for k in range(10):
            start_idx = k * 7
            end_idx = (k + 1) * 7
            ct[i][k] = np.mean(cost_matrix_new[i][start_idx:end_idx])

        index_min_cost = np.argmin(ct[i])
        min_cost_indexes.append(index_min_cost)'''

        #3rd way
        for k in range(10):
            start_idx = k * 7
            end_idx = (k + 1) * 7
            ct[i][k] = min(cost_matrix_new[i][start_idx:end_idx])
        index_min_cost = np.argmin(ct[i])
        min_cost_indexes.append(index_min_cost)
        #4th way cosine cost
        for k in range(10):
            start_idx = k * 7
            end_idx = (k + 1) * 7
            ct[i][k] = max(average_cos_similarity[i][start_idx:end_idx])
        index_max_cost = np.argmax(ct[i])
        max_cos_indexes.append(index_max_cost)
        #index_min_cost = np.argmax(average_cos_similarity[i])
        #min_cost_indexes.append(index_min_cost)

    #print("digits recognised: ")
    print(min_cost_indexes)
    print(max_cos_indexes)
    return cost_matrix_new

#gamv thn aek
'''spectrogram_2 = np.abs(librosa.stft(word_matrix2[0], win_length=512, hop_length=256))  
# we increaced the window length of the sort fourier transform because it works better?!

# we check the amplitude in dp of the spectogram/coefficients(not sure if its correct


# we save the magnitudes of each spectogram's coefficients,without the phase(hence the abs)
spec2_mag=librosa.amplitude_to_db(spectrogram_2)'''
