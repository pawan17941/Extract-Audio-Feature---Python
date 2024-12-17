# Extract-Audio-Feature---Python

Overview:
This project extracts various audio features (like Energy, Power, MFCCs, and Spectral Features) from audio files using Python. It also provides functionality to process video files by extracting audio and subsequently analyzing it for audio features.

Dependencies:
The project uses the following Python libraries:

Package	Description
moviepy	- Used to process video files and extract audio from them.
librosa -	A library for analyzing and extracting audio features.
numpy -	Numerical computing library for efficient data manipulation.
scipy -	Scientific computing tools used for mathematical operations.
soundfile -	Handles reading and writing of audio files.
matplotlib -	Used to visualize audio waveforms and extracted features.

Methodology
1. Audio and Video Processing
If the input is a video file, the script uses moviepy to extract audio from the video.
Audio is saved temporarily as a .wav file for further analysis.

from moviepy.editor import VideoFileClip

video_path = "sample_videos/video.mp4"
output_audio_path = "output/audio_from_video.wav"

clip = VideoFileClip(video_path)
clip.audio.write_audiofile(output_audio_path)

2. Audio Feature Extraction
Using librosa, the script extracts key features:

Energy and Power
Mel Frequency Cepstral Coefficients (MFCCs)
Spectral Features (Spectral Centroid, Bandwidth, and Roll-off)

Code for Extracting Features
import librosa
import numpy as np

# Load audio file
audio_path = "sample_audio/audio_file.wav"
y, sr = librosa.load(audio_path)

# Energy and Power
energy = np.sum(np.square(y))
print(f"Energy: {energy}")

# Extract MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
print(f"MFCCs Shape: {mfccs.shape}")

# Spectral Features
centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

3. Visualization
The script uses matplotlib to plot:

The audio waveform.
The extracted MFCCs as a heatmap.

Code for Visualization
import matplotlib.pyplot as plt

# Plot audio waveform
plt.figure(figsize=(10, 4))
plt.plot(y)
plt.title("Audio Waveform")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()

# Plot MFCCs
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.colorbar()
plt.title("MFCCs")
plt.show()


Conclusion
This project demonstrates how to extract audio features and visualize them using Python. It uses libraries like moviepy for video processing and librosa for audio analysis, making it ideal for audio-related applications like speech recognition, music analysis, or feature engineering for machine learning.
