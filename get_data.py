import mirdata
import mir_eval
import librosa
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder

#####################################################################
# SETTINGS
#####################################################################

DATASET = "beatles"        # any mirdata dataset with chords
VOCAB = "majmin7"          # MIREX vocab: majmin, majmin7, tertian, root, bass
SR = 22050
HOP = 512
OUT_DIR = "./processed_data"
os.makedirs(OUT_DIR, exist_ok=True)

#####################################################################
# 1. Load dataset
#####################################################################

dataset = mirdata.initialize(DATASET)
#dataset.download()
tracks = dataset.load_tracks()

missing_audio_count = 0
missing_tracks = []

for track_id, track in tracks.items():
    try:
        # Attempt to load audio
        _ = track.audio
    except Exception as e:
        missing_audio_count += 1
        missing_tracks.append((track_id, str(e)))

print(f"Number of tracks without audio: {missing_audio_count}")
if missing_tracks:
    print("Tracks missing audio and their errors:")
    for tid, err in missing_tracks:
        print(f"{tid}: {err}")
#####################################################################
# helper functions
#####################################################################

def reduce_vocab(intervals, labels, vocab):
    """Reduce given chord labels to MIREX vocabulary."""
    return mir_eval.chord.reduce_chord_labels(labels, vocab=vocab)

def framewise_labels(intervals, labels, n_frames, hop_length, sr):
    """Convert interval labels → framewise labels."""
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)
    return mir_eval.chord.to_chord_labels(intervals, labels, times)

#####################################################################
# 2. Prepare encoder for vocabulary
#####################################################################

# We'll gather all labels before fitting encoder
all_reduced_labels = []

for track_id, track in tracks.items():
    if not track.chords_ref:
        continue
    _, labels = track.chords_ref.to_interval_values()
    reduced = reduce_vocab(None, labels, VOCAB)
    all_reduced_labels.extend(reduced)

encoder = LabelEncoder()
encoder.fit(all_reduced_labels)

#####################################################################
# 3. Process tracks
#####################################################################

for track_id, track in tracks.items():
    if not track.chords_ref:
        continue
    
    print(f"Processing: {track_id}")

    # ---- Load audio ----
    audio, _ = librosa.load(track.audio_path, sr=SR)
    
    # ---- Extract features (CQT) ----
    C = librosa.cqt(audio, sr=SR, hop_length=HOP)
    logC = librosa.amplitude_to_db(np.abs(C))
    n_frames = logC.shape[1]
    
    # ---- Load chord intervals & labels ----
    intervals, labels = track.chords_ref.to_interval_values()
    
    # ---- Normalize to MIREX vocabulary ----
    labels = reduce_vocab(intervals, labels, VOCAB)
    
    # ---- Convert to framewise labels ----
    fw_labels = framewise_labels(intervals, labels, n_frames, HOP, SR)
    
    # ---- Integer-encode them ----
    fw_encoded = encoder.transform(fw_labels)

    # ---- Save ----
    np.save(os.path.join(OUT_DIR, f"{track_id}_features.npy"), logC)
    np.save(os.path.join(OUT_DIR, f"{track_id}_labels.npy"), fw_encoded)

#####################################################################
# 4. Save label→index mapping
#####################################################################

mapping = {label: int(idx) for label, idx in zip(encoder.classes_, encoder.transform(encoder.classes_))}
with open(os.path.join(OUT_DIR, "class_map.json"), "w") as f:
    json.dump(mapping, f, indent=2)

print("Done! Training-ready data saved.")
