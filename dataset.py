import numpy as np
import torch
from torch.utils.data import Dataset
import mirdata
import mir_eval.chord
import math


class BeatlesChordDataset(Dataset):
    def __init__(self, root, vocab="majmin", fps=100):
        self.dataset = mirdata.initialize("beatles", data_home=root)
        self.dataset.validate()

        all_track_ids = self.dataset.track_ids
        self.vocab = vocab
        self.fps = fps

        # Compute global max length for padding and keep only tracks with audio
        self.max_samples = 0
        self.lengths = {}
        self.track_ids = []
        self._audio_arrays = {}

        for tid in all_track_ids:
            track = self.dataset.track(tid)
            audio = track.audio  # may be (y, sr) or (None, None)
            if not audio:
                continue
            y, sr = audio
            if y is None:
                continue

            # Ensure audio is a numpy array and cache it
            y = np.asarray(y)
            self._audio_arrays[tid] = (y, sr)

            self.track_ids.append(tid)
            self.lengths[tid] = len(y)
            self.max_samples = max(self.max_samples, len(y))

    @staticmethod
    def pad(arr, target_len):
        return np.pad(arr, (0, target_len - len(arr)), mode="constant")

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        # Use cached audio (ensured to be a numpy array in __init__)
        y, sr = self._audio_arrays[track_id]

        y = self.pad(y, self.max_samples).astype(np.float32)

        # --- Load Beatles chord intervals ---
        track = self.dataset.track(track_id)
        datachord = track.chords
        intervals = datachord.intervals
        labels = datachord.labels

        # --- Normalize chord vocabulary using mir_eval ---
        reduced_labels = [
            mir_eval.chord.reduce_chord_label(lbl, self.vocab)
            for lbl in labels
        ]

        # --- Convert to frame labels ---
        duration = len(y) / sr
        num_frames = math.ceil(duration * self.fps)
        frame_times = np.arange(num_frames) / self.fps

        frame_labels = mir_eval.chord.to_chord_labels(
            frame_times, intervals, reduced_labels
        )

        # Encode chord strings to integers
        unique_labels = sorted(list(set(frame_labels)))
        label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
        frame_labels_idx = np.array([label_to_idx[l] for l in frame_labels])

        return (
            torch.from_numpy(y),
            torch.from_numpy(frame_labels_idx),
        )


