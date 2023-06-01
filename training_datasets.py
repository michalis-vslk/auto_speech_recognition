import random
from idlelib import history

import numpy as np
import librosa.display
from librosa.display import plt as plt2
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import plotting
import filtering_and_segmentation
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa
import librosa.display


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
    # cost_matrix_new = np.empty((len(word_matrix), len(t_dataset_matrix)))
    #ct = np.empty((len(word_matrix), int(len(t_dataset_matrix) / 7)))
    # ct2 = np.empty((len(word_matrix), int(len(t_dataset_matrix) / 7)))
    # cost_matrix_new = [None] * len(word_matrix) * len(t_dataset_matrix)
    # min_cost_indexes = []
    #max_cos_indexes = []
    # normal_distance = np.empty((len(word_matrix), len(t_dataset_matrix)))
    # median_value_train = np.empty(len(t_dataset_matrix))
    # std_deviation_train = np.empty(len(t_dataset_matrix))
    # median_per_training_word = np.empty(int(len(t_dataset_matrix) / 7))
    # std_deviation_per_training_word = np.empty(int(len(t_dataset_matrix) / 7))

    #average_cos_similarity = np.empty((len(word_matrix), len(t_dataset_matrix)))
    # training_digits_coefficients = []
    # max_len = max([len(i) for i in word_matrix])
    # max_len2 = max([len(i) for i in t_dataset_matrix])
    # total_max_length = max(max_len, max_len2)
    spec1_mags = []
    spec2_mags = []
    spec1_mag_max_shape_list = []
    spec2_mag_max_shape_list = []
    for i in range(len(word_matrix)):
        spectrogram_1 = np.abs(librosa.stft(word_matrix[i])) # 128 64
        spec1_mag = librosa.amplitude_to_db(spectrogram_1, ref=np.max)#normilize input spectrogram
        spec1_mag_max_shape_list.append(spec1_mag.shape[1])
        '''plt.figure(figsize=(8, 4))

        librosa.display.specshow(spec1_mag, y_axis='log')
        plt.title('Full spectrum')
        plt.colorbar(format='%+2.0f dB')

        plt.tight_layout()
        plt.show()'''
    for j in range(len(t_dataset_matrix)):
        spectrogram_2 = np.abs(librosa.stft(t_dataset_matrix[j]))  # 128 64
        spec2_mag = librosa.amplitude_to_db(spectrogram_2, ref=np.max)
        spec2_mag_max_shape_list.append(spec2_mag.shape[1])
        '''plt.figure(figsize=(8, 4))

        librosa.display.specshow(spec2_mag, y_axis='log')
        plt.title('Full spectrum')
        plt.colorbar(format='%+2.0f dB')

        plt.tight_layout()
        plt.show()'''
    temp1 = max(spec1_mag_max_shape_list)
    temp2 = max(spec2_mag_max_shape_list)
    total_max_shape = max(temp1, temp2)
    for i in range(len(word_matrix)):
        # size_diff1 = total_max_length - word_matrix[i].size
        # word_matrix[i] = np.concatenate((word_matrix[i], np.zeros(size_diff1)))
        spectrogram_1 = np.abs(librosa.stft(word_matrix[i]))  # 128 64
        spec1_mag = librosa.amplitude_to_db(spectrogram_1, ref=np.max)
        if spec1_mag.shape[1] < total_max_shape:
            size_diff = total_max_shape - spec1_mag.shape[1]
            spec1_mag = np.concatenate((spec1_mag, np.zeros((spec1_mag.shape[0], size_diff))), axis=1)
            '''plt.figure(figsize=(8, 4))

            librosa.display.specshow(spec1_mag, y_axis='log')
            plt.title('Full spectrum')
            plt.colorbar(format='%+2.0f dB')

            plt.tight_layout()
            plt.show()'''
        spec1_mags.append(spec1_mag[i])
        # median_value_1 = np.median(spec1_mag)
        # std_deviation_1 = np.std(spec1_mag)
    for j in range(len(t_dataset_matrix)):
        # we increased the window length of the sort fourier transform because it works better
        # we check the amplitude in dB of the spectrogram/coefficients
        # we save the magnitudes of each spectrogram's coefficients, without the phase(hence the abs)
        # size_diff2 = total_max_length - t_dataset_matrix[i].size
        # t_dataset_matrix[i] = np.concatenate((t_dataset_matrix[i], np.zeros(size_diff2)))
        spectrogram_2 = np.abs(librosa.stft(t_dataset_matrix[j]))  # 128 64
        spec2_mag = librosa.amplitude_to_db(spectrogram_2, ref=np.max)
        if spec2_mag.shape[1] < total_max_shape:
            size_diff2 = total_max_shape - spec2_mag.shape[1]
            spec2_mag = np.concatenate((spec2_mag, np.zeros((spec2_mag.shape[0], size_diff2))), axis=1)
        spec2_mags.append(spec2_mag[j])
        '''plt.figure(figsize=(8, 4))

        librosa.display.specshow(spec2_mag, y_axis='log')
        plt.title('Full spectrum')
        plt.colorbar(format='%+2.0f dB')

        plt.tight_layout()
        plt.show()'''
        '''if i == 1:
            spec2_mags.append(spec2_mag[i])'''
        '''# the best match for the word has the least cost (which is in the bottom right corner), we checked for
        # the other wav files
        cost_matrix, wp = librosa.sequence.dtw(X=spec1_mag, Y=spec2_mag)
        # make a list with minimum cost of each digit
        cost_matrix_new[i][j] = cost_matrix[-1, -1]
    
        window_size = 512  # Adjust the window size as needed #128 64
        hop_length = 256
        spectrogram_1_windows = librosa.util.frame(spec1_mag, frame_length=window_size, hop_length=hop_length)
        spectrogram_2_windows = librosa.util.frame(spec2_mag, frame_length=window_size, hop_length=hop_length)
        cosine_similarities = []
        for k in range(spectrogram_1_windows.shape[0]):
            vector_1 = spectrogram_1_windows[k, :].flatten()
            vector_2 = spectrogram_2_windows[k, :].flatten()
            if len(vector_2) > len(vector_1):
                print("here1")
                size_diff = vector_2.size - vector_1.size
                vector_1 = np.concatenate((vector_1, np.zeros(size_diff)))
            elif len(vector_2) < len(vector_1):
                print("here2")
                size_diff = vector_1.size - vector_2.size
                vector_2 = np.concatenate((vector_2, np.zeros(size_diff)))
            cos_similarity = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
            # print("cos similarity ",cos_similarity,"between:",i,"word","of",j,"training dataset,with window",k,"of each spec")
            cosine_similarities.append(cos_similarity)
    
            # Aggregate the cosine similarities
        average_cos_similarity[i][j] = np.nanmean(cosine_similarities)
        # print("Average Cosine Similarity:", average_cos_similarity[i])
        # median_value_train[i][j] = np.median(spec2_mag)
        # std_deviation_train[i][j] = np.std(spec2_mag)
    
    for j in range(len(t_dataset_matrix)):
        normal_distance[i][j] = np.abs(
            (median_value_1 - median_value_train[i][j]) / (std_deviation_1 - std_deviation_train[i][j]))
    
    for k in range(int(len(t_dataset_matrix) / 7)):
        start_idx = k * 7
        end_idx = (k + 1) * 7
        ct2[i][k] = min(normal_distance[i][start_idx:end_idx])
    index_min_cost = np.argmin(ct2[i])
    min_cost_indexes.append(index_min_cost)
    # index of MINIMUM COST
    # 1st way
    # index_min_cost = np.argmin(cost_matrix_new[i])
    # min_cost_indexes.append(index_min_cost)
    # 2nd way
    
    for k in range(10):
        start_idx = k * 7
        end_idx = (k + 1) * 7
        ct[i][k] = np.mean(cost_matrix_new[i][start_idx:end_idx])
    
    index_min_cost = np.argmin(ct[i])
    min_cost_indexes.append(index_min_cost)
    
    #3rd way
    for k in range(int(len(t_dataset_matrix) / 7)):
        start_idx = k * 7
        end_idx = (k + 1) * 7
        ct[i][k] = min(cost_matrix_new[i][start_idx:end_idx])
    index_min_cost = np.argmin(ct[i])
    min_cost_indexes.append(index_min_cost)
    # 4th way cosine cost
    for k in range(int(len(t_dataset_matrix) / 7)):
        start_idx = k * 7
        end_idx = (k + 1) * 7
        ct[i][k] = max(average_cos_similarity[i][start_idx:end_idx])
    index_max_cost = np.argmax(ct[i])
    max_cos_indexes.append(index_max_cost)
    # index_min_cost = np.argmax(average_cos_similarity[i])
    # min_cost_indexes.append(index_min_cost)
    
    # 5th way
    
    for k in range(int(len(t_dataset_matrix) / 7)):
    start_idx = k * 7
    end_idx = (k + 1) * 7
    median_per_training_word[k] = np.mean(median_value_train[start_idx:end_idx])
    std_deviation_per_training_word[k] = np.std(std_deviation_train[start_idx:end_idx])
    
    # print("digits recognised: ")
    # print(min_cost_indexes)
    print(max_cos_indexes)'''
    num_train_data = int(len(t_dataset_matrix) / 10)
    return spec1_mags, spec2_mags, total_max_shape, num_train_data


'''def cost_calculator(word_matrix, t_dataset_matrix):
    # ...
    spec1_mags = []
    spec2_mags = []
    spec1_mag_max_shape_list = []
    spec2_mag_max_shape_list = []
    for i in range(len(word_matrix)):
        spectrogram_1 = np.abs(librosa.stft(word_matrix[i]))
        spec1_mag = spectrogram_1
        spec1_mag_max_shape_list.append(spec1_mag.shape[1])
        plt.figure(figsize=(8, 4))

        librosa.display.specshow(librosa.amplitude_to_db(spec1_mag ,ref=np.max), y_axis='log')
        plt.title('Full spectrum')
        plt.colorbar(format='%+2.0f dB')

        plt.tight_layout()
        plt.show()
    for j in range(len(t_dataset_matrix)):
        spectrogram_2 = np.abs(librosa.stft(t_dataset_matrix[j]))
        spec2_mag = spectrogram_2
        spec2_mag_max_shape_list.append(spec2_mag.shape[1])
        plt.figure(figsize=(8, 4))

        librosa.display.specshow(librosa.amplitude_to_db(spec2_mag ,ref=np.max), y_axis='log')
        plt.title('Full spectrum')
        plt.colorbar(format='%+2.0f dB')

        plt.tight_layout()
        plt.show()
    temp1 = max(spec1_mag_max_shape_list)
    temp2 = max(spec2_mag_max_shape_list)
    total_max_shape = max(temp1, temp2)

    for i in range(len(word_matrix)):
        spectrogram_1 = np.abs(librosa.stft(word_matrix[i]))  #, win_length=2048, hop_length=1024
        spec1_mag = spectrogram_1
        if spec1_mag.shape[1] < total_max_shape:
            size_diff = total_max_shape - spec1_mag.shape[1]
            spec1_mag = np.concatenate((spec1_mag, np.zeros((spec1_mag.shape[0], size_diff))), axis=1)
        spec1_mags.append(spec1_mag[i])

        plt.figure(figsize=(8, 4))

        librosa.display.specshow(librosa.amplitude_to_db(spec1_mag ,ref=np.max), y_axis='log')
        plt.title('Full spectrum')
        plt.colorbar(format='%+2.0f dB')

        plt.tight_layout()
        plt.show()
        for j in range(len(t_dataset_matrix)):
            spectrogram_2 = np.abs(librosa.stft(t_dataset_matrix[j]))
            spec2_mag = spectrogram_2
            if spec2_mag.shape[1] < total_max_shape:
                size_diff2 = total_max_shape - spec2_mag.shape[1]
                spec2_mag = np.concatenate((spec2_mag, np.zeros((spec2_mag.shape[0], size_diff2))), axis=1)

            plt.figure(figsize=(8, 4))

            librosa.display.specshow(librosa.amplitude_to_db(spec2_mag ,ref=np.max), y_axis='log')
            plt.title('Full spectrum')
            plt.colorbar(format='%+2.0f dB')

            plt.tight_layout()
            plt.show()
            if i == 1:
                spec2_mags.append(spec2_mag[i])
    num_train_data = int(len(t_dataset_matrix) / 10)
    return spec1_mags, spec2_mags, total_max_shape, num_train_data'''


def classify_with_mlp(spec1_mags, spec2_mags, total_max_shape, n):
    spec1_mags = np.array(spec1_mags)
    spec2_mags = np.array(spec2_mags)
    spec1_labels = np.array([3, 5, 7, 9, 0, 2, 4, 6, 8, 1])
    #spec1_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    spec2_labels = np.array(
        [0] * n + [1] * n + [2] * n + [3] * n + [4] * n + [5] * n + [6] * n + [7] * n + [8] * n + [9] * n)

    # Split the data into training and validation sets
    train_spec2_mags, val_spec2_mags, train_spec2_labels, val_spec2_labels = train_test_split(
        spec2_mags, spec2_labels, test_size=0.3)  #, random_state=42

    # Create a multi-layer perceptron (MLP) model 64 32 10
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(total_max_shape,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        #tf.keras.layers.Dense(64, activation='tanh'),
        #tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.2)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the MLP model using train_spec2_mags as the input and train_spec2_labels as the target labels
    history = model.fit(train_spec2_mags, train_spec2_labels, epochs=150, batch_size=98,
                        validation_data=(val_spec2_mags, val_spec2_labels))

    # Reshape spec1_mags to match the input shape of the MLP model
    spec1_mags_reshaped = spec1_mags.reshape(spec1_mags.shape[0], -1)

    # Classify spec1_mags using the trained MLP model
    predictions = model.predict(spec1_mags_reshaped)
    predicted_classes = np.argmax(predictions, axis=1)

    # Plot training and validation loss
    plt2.plot(history.history['loss'])
    plt2.plot(history.history['val_loss'])
    plt2.title('Model Loss')
    plt2.xlabel('Epoch')
    plt2.ylabel('Loss')
    plt2.legend(['Train', 'Validation'])
    plt2.show()

    # Plot training and validation accuracy
    plt2.plot(history.history['accuracy'])
    plt2.plot(history.history['val_accuracy'])
    plt2.title('Model Accuracy')
    plt2.xlabel('Epoch')
    plt2.ylabel('Accuracy')
    plt2.legend(['Train', 'Validation'])
    plt2.show()

    # Create a confusion matrix
    cm = confusion_matrix(val_spec2_labels, np.argmax(model.predict(val_spec2_mags), axis=1))
    #cm = confusion_matrix(spec1_labels, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    correct_predictions = np.equal(predicted_classes, spec1_labels)

    # Calculate accuracy
    accuracy = np.mean(correct_predictions)

    # Print accuracy
    print("Accuracy:", accuracy)

    # Visualize the predicted classes and actual labels
    plt.scatter(range(len(spec1_labels)), spec1_labels, label='Actual Labels')
    plt.scatter(range(len(predicted_classes)), predicted_classes, label='Predicted Classes')
    plt.xlabel('Sample Index')
    plt.ylabel('Class')
    plt.title('Comparison of Actual Labels and Predicted Classes')
    plt.legend()
    plt.show()
    return predicted_classes


'''spectrogram_2 = np.abs(librosa.stft(word_matrix2[0], win_length=512, hop_length=256))  
# we increaced the window length of the sort fourier transform because it works better?!

# we check the amplitude in dp of the spectogram/coefficients(not sure if its correct


# we save the magnitudes of each spectogram's coefficients,without the phase(hence the abs)
spec2_mag=librosa.amplitude_to_db(spectrogram_2)'''
