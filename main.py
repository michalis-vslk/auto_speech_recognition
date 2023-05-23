from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import librosa
import sounddevice as sd
import noisereduce as nr
import librosa.display


signal, sr = librosa.load("sample-1.wav")  # we change sr
signal_emphasized = librosa.effects.preemphasis(signal)
S_full, phase = librosa.magphase(librosa.stft(signal_emphasized))
idx = slice(*librosa.time_to_frames([0, 18], sr=sr))

plt.figure(figsize=(12, 4))
plt.imshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
           origin='lower', aspect='auto')
plt.colorbar(format='%+2.0f dB')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram')
#plt.show()

S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))

# The output of the filter shouldn't be greater than the input
# if we assume signals are additive.  Taking the pointwise minimium
# with the input spectrum forces this.
S_filter = np.minimum(S_full, S_filter)

margin_i, margin_v = 2, 10
power = 2

mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)

mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)

# Once we have the masks, simply multiply them with the input spectrum
# to separate the components

S_foreground = mask_v * S_full
S_background = mask_i * S_full

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                         y_axis='log', sr=sr)
plt.title('Full spectrum')
plt.colorbar()

plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(S_background[:, idx], ref=np.max),
                         y_axis='log', sr=sr)
plt.title('Background')
plt.colorbar()
plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(S_foreground[:, idx], ref=np.max),
                         y_axis='log', x_axis='time', sr=sr)
plt.title('Foreground')
plt.colorbar()
plt.tight_layout()
#plt.show()

foreground_audio = librosa.istft(S_foreground * phase)
endpoints = librosa.effects.split(foreground_audio)

# Segment the audio signal into words
words = []
for start, end in endpoints:
    word = signal[start:end]
    words.append(word)

# Optionally, apply additional processing techniques to refine the word segments

# Save the words in a matrix or suitable data structure
word_matrix = []
for word in words:
    if len(word) > 5000:
        word_matrix.append(word)

# Access individual words in the matrix
for word in word_matrix:
    # Process each word as needed
    pass
'''
words = []
current_part = []
threshold = 1e-7
for sample in foreground_audio:
    if abs(sample) > threshold:
        current_part.append(sample)
    else:
        words.append(current_part)
        current_part = []'''

#print(len(word_matrix))
'''
fig, ax = plt.subplots(nrows=2, sharex="all", sharey="all", constrained_layout=True)

ax[0].set(title="Original Audio Waveform Graph", xlabel="TXT_TIME", ylabel="TXT_AMPLITUDE")
ax[1].set(title="Audio Waveform Graph", xlabel="TXT_TIME", ylabel="TXT_AMPLITUDE")

# Apply grid
ax[0].grid()
ax[1].grid()

librosa.display.waveshow(signal, sr=sr, ax=ax[0], label="TXT_ORIGINAL")
librosa.display.waveshow(signal, sr=sr, ax=ax[1], label="TXT_ORIGINAL")
librosa.display.waveshow(foreground_audio, sr=sr, ax=ax[1], label="TXT_FILTERED")

# Set legend
ax[1].legend()

# Show plot
plt.show()
'''
for word in word_matrix:
    print(len(word))
'''
for i in range(len(word_matrix)):
    sd.play(word_matrix[i], sr)
    sd.wait()'''


