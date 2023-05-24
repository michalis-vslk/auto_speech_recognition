from __future__ import print_function
import numpy as np
import librosa
# import sounddevice as sd
# import noisereduce as nr
import matplotlib.pyplot as plt
import librosa.display
import filtering_and_segmentation


# Loading the audio file
signal, sr = librosa.load("sample-1.wav")  # we change sr

# 1) Filtering the original signal, isolating each word and calculating the speaker's fundamental frequency
# Filtering with the use of a foreground vs background classifier
foreground_audio = filtering_and_segmentation.filtering(signal, sr)
# Isolating each word through segmentation
word_matrix = filtering_and_segmentation.segmentation(signal, foreground_audio)
# Calculating the fundamental frequency of the speaker
average_fundamental_frequency = filtering_and_segmentation.fundamental_frequency_calculator(word_matrix, sr)
print("The fundamental frequency of the speaker is calculated at: " + str(average_fundamental_frequency) + " Hz")


''' For testing audio
for i in range(len(word_matrix)):
    sd.play(word_matrix[i], sr)
    sd.wait()'''
