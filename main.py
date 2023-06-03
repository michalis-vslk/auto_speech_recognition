from __future__ import print_function
import sounddevice as sd

import matplotlib.pyplot as plt
import librosa.display
import filtering_and_segmentation
import training_datasets

# Loading the audio file
signal, sr = librosa.load("sample-3.wav")  # we change sr

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

spec1,spec2,total_max_shape,num_train_data,total_max_shape2 = training_datasets.cost_calculator(word_matrix,t_dataset_matrix)

predicted_classes,max_accuracy,history = training_datasets.classify_with_mlp(spec1,spec2,total_max_shape,num_train_data,total_max_shape2)
print(predicted_classes)
while True:
    if max_accuracy >= 0.7:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'])
        plt.show()

        # Plot training and validation accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'])
        plt.show()

        break
    else:
        predicted_classes, max_accuracy,history = training_datasets.classify_with_mlp(spec1, spec2, total_max_shape,
                                                                              num_train_data, total_max_shape2)

 #For testing audio
'''for i in range(len(word_matrix)):
    sd.play(word_matrix[i], sr)
    sd.wait()'''


