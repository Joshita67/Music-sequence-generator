import streamlit as st
from keras.models import load_model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pretty_midi
import os

# 📍 Load your trained model
MODEL_PATH = "sequence_model.keras"
model = load_model(MODEL_PATH)

# 🎛️ Generate dummy input matching model's expected shape
def generate_dummy_input(sequence_length=100, num_features=1):
    return np.random.rand(sequence_length, num_features).astype(np.float32)

# 🔮 Predict sequence from input
def predict_sequence(input_sequence):
    input_sequence = np.expand_dims(input_sequence, axis=0)  # shape: (1, seq_len, num_features)
    return model.predict(input_sequence)

# 📊 Plot predicted sequence
def plot_sequence(sequence):
    plt.figure(figsize=(10, 4))
    plt.plot(sequence.squeeze(), marker='o', linestyle='-', color='purple')
    plt.title("Predicted Music Sequence")
    plt.xlabel("Time Step")
    plt.ylabel("Note/Pitch Value")
    st.pyplot(plt)

# 🎹 Convert sequence to MIDI file
def sequence_to_midi(sequence, output_path="generated_sequence/output.mid"):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    for i, pitch in enumerate(sequence.squeeze()):
        note = pretty_midi.Note(
            velocity=100,
            pitch=int(np.clip(pitch, 21, 108)),  # Restrict to piano note range
            start=i * 0.5,
            end=(i + 1) * 0.5
        )
        instrument.notes.append(note)

    pm.instruments.append(instrument)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pm.write(output_path)
    return output_path

# 🎧 Streamlit UI
st.title("🎼 Music Sequence Generator")
st.markdown("Upload MIDI or generate dummy input to create musical predictions using your trained model.")

# 🎯 Input selection
choice = st.radio("Choose input method:", ["Generate Dummy Input", "Upload MIDI (coming soon)"])

if choice == "Generate Dummy Input":
    input_sequence = generate_dummy_input()
    st.success("✅ Dummy input sequence generated.")
else:
    uploaded_file = st.file_uploader("Upload MIDI file (not yet supported)", type=["mid", "midi"])
    if uploaded_file:
        st.info("MIDI support coming soon! Using dummy input instead.")
        input_sequence = generate_dummy_input()
    else:
        input_sequence = None

# 🔮 Predict and display output
if input_sequence is not None:
    st.subheader("🔮 Predicted Output")
    predicted_sequence = predict_sequence(input_sequence)
    st.write(predicted_sequence)

    # 📊 Show graph
    st.subheader("📊 Visual Sequence Plot")
    plot_sequence(predicted_sequence)

    # 🎹 Create and share MIDI
    st.subheader("🎹 MIDI Playback")
    midi_path = sequence_to_midi(predicted_sequence)
    st.audio(midi_path, format='audio/midi')
    with open(midi_path, "rb") as f:
        st.download_button("Download MIDI File 🎶", data=f, file_name="generated_music.mid")
