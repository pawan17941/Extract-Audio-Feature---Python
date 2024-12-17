import librosa
import numpy as np
import moviepy.editor as mp
import matplotlib.pyplot as plt
import soundfile as sf
import cv2

def extract_audio(video_file=None, duration=90):
    if video_file:
        video = mp.VideoFileClip(video_file).subclip(0, duration)
        audio_file = "output_audio.wav"
        video.audio.write_audiofile(audio_file)
        audio, sr = librosa.load(audio_file, sr=None)
    else:
        cap = cv2.VideoCapture(0)
        frames = []
        for _ in range(int(duration * 30)):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        audio, sr = librosa.load(librosa.example('trumpet'), sr=None)
    return audio, sr

def calculate_audio_features(audio, sr):
    features = {}
    features['Energy'] = np.sum(audio**2)
    features['Dynamic Range'] = np.max(audio) - np.min(audio)
    features['MFCCs'] = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features['Spectral Centroid'] = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features['Spectral Rolloff'] = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features['Spectral Bandwidth'] = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    return features

def visualize_features(audio, sr, features):
    plt.figure(figsize=(14, 8))
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(audio, sr=sr)
    plt.title("Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.subplot(3, 1, 2)
    times = librosa.times_like(features['Spectral Centroid'], sr=sr)
    plt.semilogy(times, features['Spectral Centroid'][0], label='Spectral Centroid')
    plt.xlabel("Time (s)")
    plt.ylabel("Hz")
    plt.title("Spectral Centroid")
    plt.legend()
    plt.subplot(3, 1, 3)
    librosa.display.specshow(features['MFCCs'], x_axis='time', sr=sr)
    plt.colorbar()
    plt.title("MFCCs")
    plt.xlabel("Time (s)")
    plt.ylabel("MFCC Coefficients")
    plt.tight_layout()
    plt.show()

def main():
    video_file = r"C:\Users\A\Downloads\Saki.mp4"
    duration = 90
    audio, sr = extract_audio(video_file, duration)
    features = calculate_audio_features(audio, sr)
    print("Extracted Features:")
    for key, value in features.items():
        print(f"{key}: {value[:5] if isinstance(value, np.ndarray) else value}")
    visualize_features(audio, sr, features)

if __name__ == "__main__":
    main()
