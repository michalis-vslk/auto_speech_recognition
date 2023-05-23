import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np


def plot_spectrogram(spectrogram_full, idx):
    plt.figure(figsize=(12, 4))
    plt.imshow(librosa.amplitude_to_db(spectrogram_full[:, idx], ref=np.max),
               origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Spectrogram')
    plt.show()


def plot_foreground_background_comparison(spectrogram_full, spectrogram_background, spectrogram_foreground, idx, sr):
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram_full[:, idx], ref=np.max),
                             y_axis='log', sr=sr)
    plt.title('Full spectrum')
    plt.colorbar()

    plt.subplot(3, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram_background[:, idx], ref=np.max),
                             y_axis='log', sr=sr)
    plt.title('Background')
    plt.colorbar()
    plt.subplot(3, 1, 3)
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram_foreground[:, idx], ref=np.max),
                             y_axis='log', x_axis='time', sr=sr)
    plt.title('Foreground')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
