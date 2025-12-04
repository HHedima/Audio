import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf

import tkinter as tk
from tkinter import ttk
from tkinter import *
import threading
import os

# instance variables
audio_file = 'untitled.wav'
directory = 'C:\\Users\\warha\\OneDrive\\Documents\\coding\\Audio\\'
output_file = 'output_0.wav'
file_counter = 0
freqMin = 30
freqMax = 20000
energyThreshold = 0.1 # adjust sensitivity
y, sr = librosa.load(audio_file, sr=None) # load audio file
midi_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
tempo = 120
beat_frames = []


# functions

# file control
# file must be in this directory
def set_audio_file(file_path):
    global audio_file, y, sr, selected_file_label
    audio_file = os.path.join(directory, file_path)
    y, sr = librosa.load(audio_file, sr=None)
    selected_file_label.config(text=f"Current File: {audio_file}")

def create_output_file(output_file):
    global file_counter
    while os.path.exists(output_file):
        file_counter += 1
        output_file = f"output_{file_counter}.wav"
    return output_file
    

# audio control
def stop_audio():
    sd.stop()

def play_audio():
    # convert to stereo if mono
    if len(y.shape) == 1: 
        y_stereo = np.column_stack((y, y))
    else:
        y_stereo = y
    sd.play(y_stereo, sr)

def record():
    print("Recording...")
    global recording, is_recording
    is_recording = True
    recording = []

    def callback(indata, frames, time, status):
        if is_recording:
            recording.append(indata.copy())

    with sd.InputStream(samplerate=sr, channels=2, callback=callback):
        while is_recording:
            sd.sleep(100)

    # Save the recording
    recorded_audio = np.concatenate(recording, axis=0)
    global output_file
    output_file = create_output_file(output_file)
    sf.write(output_file, recorded_audio, sr)
    print("Recording saved to", output_file)

def stop_recording():
    global is_recording
    is_recording = False
    print("Recording stopped.")

# find frequency with highest energy
def dominant_frequency(y, sr):
    # Compute the Short-Time Fourier Transform (STFT)
    D = np.abs(librosa.stft(y))
    # Compute the frequency bins
    freqs = librosa.fft_frequencies(sr=sr)
    # Compute the energy for each frequency bin
    energy = np.sum(D**2, axis=1)
    # Find the index of the frequency with the highest energy
    dominant_idx = np.argmax(energy)
    dominant_freq = freqs[dominant_idx]
    return dominant_freq

def hertz_to_midi(freq):
    if freq <= 0:
        return "N/A"
    midi_number = 69 + 12 * np.log2(freq / 440.0)
    midi_number = int(round(midi_number))
    octave = (midi_number // 12) - 1
    note_index = midi_number % 12
    note_name = midi_notes[note_index]
    return f"{note_name}{octave}"

def speed_up_audio(factor):
    global y, sr
    if factor <= 0:
        print("Speed factor must be greater than 0.")
        return
    y = librosa.effects.time_stretch(y, rate=factor)

    print(f"Audio speed changed by a factor of {factor}.")

def change_pitch_semitones(semitones):
    global y, sr
    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
    print(f"Audio pitch changed by {semitones} semitones.")

def find_key_signature(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    key_index = np.argmax(chroma_mean)
    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return key_names[key_index]

def estimate_tempo(y, sr):
    global tempo, beat_frames
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    print(f"Tempo: {tempo} BPM")
    return tempo


# GUI setup
def main():
    root = tk.Tk()
    root.title("Audio Player")
    root.geometry("640x360")

    # Playback Frame
    playback_frame = ttk.Frame(root)
    playback_frame.pack(side=tk.BOTTOM, pady=10)

    play_button = ttk.Button(playback_frame, text="Play Audio", command=play_audio)
    play_button.grid(row=0, column=0, padx=5, pady=5)

    stop_button = ttk.Button(playback_frame, text="Stop Audio", command=stop_audio)
    stop_button.grid(row=0, column=1, padx=5, pady=5)

    start_record_button = ttk.Button(playback_frame, text="Start Recording", command=lambda: threading.Thread(target=record).start())
    start_record_button.grid(row=0, column=2, padx=5, pady=5)

    stop_record_button = ttk.Button(playback_frame, text="Stop Recording", command=stop_recording)
    stop_record_button.grid(row=0, column=3, padx=5, pady=5)


    # Playback Modifications
    playback_speed_label = ttk.Label(playback_frame, text="Speed Factor:")
    playback_speed_label.grid(row=1, column=0, padx=5, pady=5)
    speed_factor_entry = ttk.Entry(playback_frame, width=10)
    speed_factor_entry.insert(0, "1.0") 
    speed_factor_entry.grid(row=1, column=1, padx=5, pady=5)

    speed_up_button = ttk.Button(playback_frame, text="Change Speed", command=lambda: speed_up_audio(float(speed_factor_entry.get())))
    speed_up_button.grid(row=1, column=2, padx=5, pady=5)

    pitch_shift_label = ttk.Label(playback_frame, text="Pitch Shift (semitones):")
    pitch_shift_label.grid(row=2, column=0, padx=5, pady=5)
    pitch_shift_entry = ttk.Entry(playback_frame, width=10)
    pitch_shift_entry.insert(0, "0")
    pitch_shift_entry.grid(row=2, column=1, padx=5, pady=5)

    pitch_shift_button = ttk.Button(playback_frame, text="Change Pitch", command=lambda: change_pitch_semitones(int(pitch_shift_entry.get())))
    pitch_shift_button.grid(row=2, column=2, padx=5, pady=5)

    # Audio Analysis Frame
    analysis_frame = ttk.Frame(root)
    analysis_frame.pack(side=tk.BOTTOM, pady=10)

    freq_label = ttk.Label(analysis_frame, text="Dominant Frequency:")
    freq_label.grid(row=0, column=0, padx=5, pady=5)

    dominant_freq_value = ttk.Label(analysis_frame, text="N/A")
    dominant_freq_value.grid(row=0, column=1, padx=5, pady=5)

    tempo_label = ttk.Label(analysis_frame, text="Estimated Tempo:")
    tempo_label.grid(row=0, column=2, padx=5, pady=5)

    tempo_value = ttk.Label(analysis_frame, text="N/A")
    tempo_value.grid(row=0, column=3, padx=5, pady=5)

    key_signature_label = ttk.Label(analysis_frame, text="Key Signature:")
    key_signature_label.grid(row=0, column=4, padx=5, pady=5)

    key_signature_value = ttk.Label(analysis_frame, text=find_key_signature(y, sr))
    key_signature_value.grid(row=0, column=5, padx=5, pady=5)


    def update_Analysis():
        y, sr = librosa.load(audio_file, sr=None)
        freq = dominant_frequency(y, sr)
        dominant_freq_value.config(text=f"{freq:.2f} Hz" + f" ({hertz_to_midi(freq)})")
        est_tempo = estimate_tempo(y, sr)
        tempo_value.config(text=f"Tempo: {tempo} BPM")
        key_signature_value.config(text=find_key_signature(y, sr))
        

    update_button = ttk.Button(analysis_frame, text="Update Analysis", command=update_Analysis)
    update_button.grid(row=1, column=0, columnspan=2, pady=5)




    # File Selection Frame
    file_selection_frame = ttk.Frame(root)
    file_selection_frame.pack(side=tk.TOP, pady=40)

    file_label = ttk.Label(file_selection_frame, text="Audio File:")
    file_label.grid(row=1, column=0, padx=5, pady=5)

    audio_file_entry = ttk.Entry(file_selection_frame, width=40)
    audio_file_entry.insert(0, audio_file)
    audio_file_entry.grid(row=1, column=1, padx=5, pady=5)

    set_button = ttk.Button(file_selection_frame, text="Set", command=lambda: set_audio_file(audio_file_entry.get()))
    set_button.grid(row=1, column=2, padx=5, pady=5)
    global selected_file_label
    selected_file_label = ttk.Label(file_selection_frame, text=f"Current File: {audio_file}")
    selected_file_label = ttk.Label(file_selection_frame, text=f"Current File: {audio_file}")
    selected_file_label.grid(row=0, column=0, columnspan=3, pady=5)


    

    root.mainloop()

if __name__ == "__main__":
    main()