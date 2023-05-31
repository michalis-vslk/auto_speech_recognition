from __future__ import print_function
import numpy as np
import librosa
import sounddevice as sd
# import noisereduce as nr
import matplotlib.pyplot as plt
import librosa.display
import filtering_and_segmentation
import training_datasets

# Loading the audio file
signal, sr = librosa.load("sample-1.wav")  # we change sr

# 1) Filtering the original signal, isolating each word and calculating the speaker's fundamental frequency
# Filtering with the use of a foreground vs background classifier
foreground_audio = filtering_and_segmentation.filtering(signal, sr)
# Isolating each word through segmentation
word_matrix = filtering_and_segmentation.segmentation(foreground_audio)
# Calculating the fundamental frequency of the speaker
average_fundamental_frequency = filtering_and_segmentation.fundamental_frequency_calculator(word_matrix, sr)
print("The fundamental frequency of the speaker is calculated at: " + str(average_fundamental_frequency) + " Hz")
# 3
t_dataset_matrix = training_datasets.training()
'''for i in range(len(word_matrix)):
    sd.play(word_matrix[i], sr)
    sd.wait()

for i in range(len(t_dataset_matrix)):
    sd.play(t_dataset_matrix[i], sr)
    sd.wait()'''
spec1,spec2,total_max_shape,num_train_data = training_datasets.cost_calculator(word_matrix,t_dataset_matrix)
predicted_classes = training_datasets.classify_with_mlp(spec1,spec2,total_max_shape,num_train_data)
print(predicted_classes)
'''
#this is our way,sort of. because we changed to check the dp amplitude


# Divide the spectrograms into smaller windows
window_size = 16  # Adjust the window size as needed
hop_length = 8   # Adjust the hop length as needed

#spectrogram_1_windows = librosa.util.frame(spectrogram_1, frame_length=window_size, hop_length=hop_length)
#spectrogram_2_windows = librosa.util.frame(spectrogram_2, frame_length=window_size, hop_length=hop_length)

# we seperate the magnitude in small windows so that we can compare them
spectrogram_1_windows = librosa.util.frame(spec1_mag, frame_length=window_size, hop_length=hop_length)
spectrogram_2_windows = librosa.util.frame(spec2_mag, frame_length=window_size, hop_length=hop_length)

print(spectrogram_1_windows.shape)
print(spectrogram_2_windows.shape)
# Calculate the cosine similarity for each window
cosine_similarities = []
for i in range(spectrogram_1_windows.shape[1]):
    vector_1 = spectrogram_1_windows[:, i].flatten()
    vector_2 = spectrogram_2_windows[:, i].flatten()

    # Pad or truncate the vectors to match the size
    if len(vector_2) > len(vector_1):
        size_diff = vector_2.size - vector_1.size
        vector_1 = np.concatenate((vector_1, np.zeros(size_diff)))
    else:
        size_diff = vector_1.size - vector_2.size
        vector_2 = np.concatenate((vector_2, np.zeros(size_diff)))

    # Compute cosine similarity between the vectors
    cos_similarity = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
    cosine_similarities.append(cos_similarity)

# Aggregate the cosine similarities
average_cos_similarity = np.mean(cosine_similarities)
print("Average Cosine Similarity:", average_cos_similarity)'''
'''
 #For testing audio
for i in range(len(word_matrix)):
    sd.play(word_matrix[i], sr)
    sd.wait()'''


