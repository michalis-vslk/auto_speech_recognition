import numpy as np
import librosa.display
from sklearn.model_selection import train_test_split
import filtering_and_segmentation
import os
import tensorflow as tf
import librosa
import librosa.display


def training():
    t_datasets = []
    # assigning the proper directory
    directory = 'training dataset'
    # directory = 't2'

    # iterating over files in that directory
    for filename in os.scandir(directory):
        if filename.is_file():
            signal, sr = librosa.load(filename.path)
            foreground_audio = filtering_and_segmentation.filtering(signal, sr)
            word_matrix = filtering_and_segmentation.segmentation(foreground_audio)
            t_datasets.append(word_matrix[0])
    return t_datasets


def cost_calculator(word_matrix, t_dataset_matrix):
    num_train_data = int(len(t_dataset_matrix) / 10)
    spec1_mags = []
    spec2_mags = []
    spec1_mag_max_shape_list = []
    spec2_mag_max_shape_list = []
    for i in range(len(word_matrix)):
        spectrogram_1 = np.abs(librosa.stft(word_matrix[i]))  # 128 64
        spec1_mag = librosa.amplitude_to_db(spectrogram_1, ref=np.max)  # normilize input spectrogram
        spec1_mag_max_shape_list.append(spec1_mag.shape[1])
    for j in range(len(t_dataset_matrix)):
        spectrogram_2 = np.abs(librosa.stft(t_dataset_matrix[j]))  # 128 64
        spec2_mag = librosa.amplitude_to_db(spectrogram_2, ref=np.max)
        spec2_mag_max_shape_list.append(spec2_mag.shape[1])
    temp1 = max(spec1_mag_max_shape_list)
    temp2 = max(spec2_mag_max_shape_list)
    total_max_shape = max(temp1, temp2)
    total_max_shape2 = 1025
    for i in range(len(word_matrix)):

        spectrogram_1 = np.abs(librosa.stft(word_matrix[i]))  # 128 64
        spec1_mag = librosa.amplitude_to_db(spectrogram_1, ref=np.max)
        if spec1_mag.shape[1] < total_max_shape:
            size_diff = total_max_shape - spec1_mag.shape[1]
            spec1_mag = np.concatenate((spec1_mag, np.zeros((spec1_mag.shape[0], size_diff))), axis=1)

        spec1_mags.append(spec1_mag)
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
        spec2_mags.append(spec2_mag)

    return spec1_mags, spec2_mags, total_max_shape, num_train_data,total_max_shape2




def classify_with_mlp(spec1_mags, spec2_mags, total_max_shape, n,total_max_shape2):
    spec1_mags = np.array(spec1_mags)
    spec2_mags = np.array(spec2_mags)

    spec1_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    spec2_labels = np.array(
        [0] * n + [1] * n + [2] * n + [3] * n + [4] * n + [5] * n + [6] * n + [7] * n + [8] * n + [9] * n)

    # Flatten the spectrograms
    spec2_mags_flat = spec2_mags.reshape(spec2_mags.shape[0], -1)

    # Split the data into training and validation sets
    train_spec2_mags, val_spec2_mags, train_spec2_labels, val_spec2_labels = train_test_split(
        spec2_mags_flat, spec2_labels, test_size=0.3)

    # Create a multi-layer perceptron (MLP) model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(total_max_shape * total_max_shape2,)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    #best config yet
    ''' tf.keras.layers.Dense(64, activation='relu', input_shape=(total_max_shape * total_max_shape2,)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')'''
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    predicted_classes_history = []
    max_accuracy = []
    # Define a custom callback to store predicted classes at the end of each epoch
    class PredictionCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            spec1_mags_reshaped = spec1_mags.reshape(spec1_mags.shape[0], -1)
            predictions = model.predict(spec1_mags_reshaped)
            predicted_classes = np.argmax(predictions, axis=1)
            predicted_classes_history.append(predicted_classes)

            # Calculate accuracy
            correct_predictions = np.equal(predicted_classes, spec1_labels)
            my_accuracy = np.mean(correct_predictions)
            max_accuracy.append(my_accuracy)
            # Print accuracy
            print("Epoch", epoch + 1, "Accuracy:", my_accuracy,"classes:",predicted_classes)

            if my_accuracy == 1:
                self.model.stop_training = True
                print("Successfully identified the classes:", predicted_classes)

    # Train the MLP model
    history = model.fit(train_spec2_mags, train_spec2_labels, epochs=150, batch_size=112,
                        validation_data=(val_spec2_mags, val_spec2_labels),
                        callbacks=[PredictionCallback()])

    # Reshape spec1_mags to match the input shape of the MLP model
    spec1_mags_reshaped = spec1_mags.reshape(spec1_mags.shape[0], -1)
    # Classify spec1_mags using the trained MLP model
    predictions = model.predict(spec1_mags_reshaped)
    predicted_classes = np.argmax(predictions, axis=1)



    return predicted_classes,max(max_accuracy),history

