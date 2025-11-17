import numpy as np
import torch
from torch.utils.data import Dataset
import mirdata
import mir_eval.chord as mrc
import math
import librosa


def compute_cqt(y, sr, hop_length, n_bins=84, bins_per_octave=12):
    """
    Compute log magnitude CQT and return shape (freq_bins, time_frames).
    Use librosa.cqt -> magnitude -> log.
    """
    C = librosa.cqt(y, sr=sr, hop_length=hop_length,
                    n_bins=n_bins, bins_per_octave=bins_per_octave)
    mag = np.abs(C)
    logC = librosa.amplitude_to_db(mag, ref=np.max)
    return logC.astype(np.float32)  # shape (n_bins, n_frames)


class BeatlesChordDataset(Dataset):
    def __init__(self, root, fps=100):
        self.dataset = mirdata.initialize("beatles", data_home=root)
        self.fps = fps

        all_track_ids = self.dataset.track_ids

        # ---------------------------
        # STEP 1 — Load only VALID audio tracks
        # ---------------------------
        self._audio_arrays = {}
        self.track_ids = []     # only valid tracks
        self.lengths = {}
        self.max_samples = 0

        for tid in all_track_ids:
            track = self.dataset.track(tid)

            try:
                audio = track.audio   # may raise
            except Exception:
                continue  # skip completely invalid entries

            if audio is None:
                continue

            # audio may be (y, sr) or y-only
            if isinstance(audio, (tuple, list)):
                if len(audio) == 0:
                    continue
                y = audio[0]
                sr = audio[1] if len(audio) > 1 else None
            else:
                y = audio
                sr = None

            if y is None:
                continue

            y = np.asarray(y)

            # fallback sample rate
            if sr is None:
                sr = getattr(track, "sample_rate", None) or \
                     getattr(track, "sr", None) or 22050

            # Cache audio
            self._audio_arrays[tid] = (y, sr)
            self.track_ids.append(tid)
            self.lengths[tid] = len(y)
            self.max_samples = max(self.max_samples, len(y))

        print(f"Loaded {len(self.track_ids)} tracks with valid audio.")

        # ---------------------------
        # STEP 2 — Build vocabulary from all chord labels
        # ---------------------------
        all_labels = set()
        for tid in self.track_ids:
            track = self.dataset.track(tid)
            all_labels.update(track.chords.labels)

        all_labels.add(mrc.NO_CHORD)
        all_labels = sorted(all_labels)

        # numeric encodings
        roots, bitmaps, basses = mrc.encode_many(all_labels)
        encodings = []
        for r, b, ba in zip(roots, bitmaps, basses):
            encodings.append((int(r), tuple(b.tolist()), int(ba)))

        self.encoding_to_idx = {enc: i for i, enc in enumerate(encodings)}
        self.label_to_encoding = dict(zip(all_labels, encodings))

        print(f"Chord vocabulary size: {len(self.encoding_to_idx)} classes.")

    @staticmethod
    def pad(arr, target_len):
        return np.pad(arr, (0, target_len - len(arr)), mode="constant")

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        track = self.dataset.track(track_id)

        # Load cached audio
        y, sr = self._audio_arrays[track_id]
        y = self.pad(y, self.max_samples).astype(np.float32)

        # Chord intervals
        intervals = track.chords.intervals
        labels = track.chords.labels

        # Frame-wise time grid
        duration = len(y) / sr
        n_frames = math.ceil(duration * self.fps)
        frame_times = np.arange(n_frames) / self.fps

        # start with NO_CHORD everywhere
        no_enc = self.label_to_encoding[mrc.NO_CHORD]
        no_idx = self.encoding_to_idx[no_enc]

        frame_labels = np.full(n_frames, no_idx, dtype=np.int64)

        # Fill interval labels
        for (start, end), lab in zip(intervals, labels):
            try:
                r, b, ba = mrc.encode(lab)
                enc = (int(r), tuple(b.tolist()), int(ba))
                idx_c = self.encoding_to_idx[enc]
            except Exception:
                idx_c = no_idx

            mask = (frame_times >= start) & (frame_times < end)
            frame_labels[mask] = idx_c

        return (
            torch.from_numpy(y),
            torch.from_numpy(frame_labels),
        )


class FramewiseSpectrogramDataset(BeatlesChordDataset):
    """
    Wraps an existing dataset that returns (audio, frame_labels) where:
      - audio is a 1D numpy array
      - frame_labels is 1D numpy array of length n_frames (fps resolution)
    This class converts audio -> CQT features and slices into segments of seg_frames.
    Each item returned: (spec_segment, label_segment, valid_len)
      - spec_segment: tensor (C=1, F, T_seg)  [float32]
      - label_segment: tensor (T_seg,) with class indices
      - valid_len: number of valid frames (for last shortened segment)
    """
    def __init__(self, base_dataset, fps=100, seg_seconds=8.0,
                 n_bins=84, bins_per_octave=12, hop_length=None):
        """
        base_dataset: your BeatlesChordDataset (or similar) that yields (y, labels)
        fps: frames per second for labels
        seg_seconds: segment length in seconds for training windows
        hop_length: samples between CQT frames; if None computed as sr // fps
        """
        self.base = base_dataset
        self.fps = fps
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.seg_frames = int(seg_seconds * fps)
        self.hop_length = hop_length  # may be set per track if sr varies

        # Build index: for each base track, how many segments it yields
        self.index = []  # list of tuples (track_idx, seg_start_frame, n_valid_frames_in_segment)
        for tidx in range(len(self.base)):
            y, labels = self.base[tidx]  # base returns (y, label_frames)
            # base may return torch tensors; convert to np
            if isinstance(y, torch.Tensor):
                y = y.numpy()
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()

            # Need sr: try to get via base.dataset.track(...) if available
            # We'll require base to expose its sampling rate in audio_map or so
            # Try duck-typing: base has attribute audio_map dict tid->(y,sr) or _audio_arrays
            sr = None
            if hasattr(self.base, "_audio_arrays"):
                # base.track_ids[idx] -> track id; but our base uses indices as tidx
                # We attempt to get sr from cached audio
                try:
                    track_id = self.base.track_ids[tidx]
                    sr = self.base._audio_arrays[track_id][1]
                except Exception:
                    sr = None

            if sr is None:
                sr = 44100

            hop = self.hop_length or max(1, int(round(sr / fps)))
            # number of frames available in this track
            n_frames = len(labels)
            # compute number of segments starting positions (stride = seg_frames, non-overlapping)
            start = 0
            while start < n_frames:
                valid = min(self.seg_frames, n_frames - start)
                self.index.append((tidx, start, valid, sr, hop))
                start += self.seg_frames  # non-overlapping; you could also slide by hop/2

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        tidx, start_frame, valid_frames, sr, hop = self.index[idx]
        y, labels = self.base[tidx]
        if isinstance(y, torch.Tensor):
            y = y.numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()

        # compute CQT for this track if necessary (we compute full CQT then slice frames)
        # Compute once per __getitem__ (could be cached if you want)
        spec = compute_cqt(y, sr, hop_length=hop, n_bins=self.n_bins,
                           bins_per_octave=self.bins_per_octave)
        # spec shape: (F, T_full)
        # Ensure spec has sufficient frames: if short pad with -80 dB (log)
        T_full = spec.shape[1]
        required_end = start_frame + self.seg_frames
        if required_end > T_full:
            # pad on right with low values
            pad_amount = required_end - T_full
            pad = np.full((spec.shape[0], pad_amount), spec.min(), dtype=np.float32)
            spec = np.concatenate([spec, pad], axis=1)

        seg = spec[:, start_frame:start_frame + self.seg_frames]  # (F, T_seg)
        seg = np.expand_dims(seg, axis=0)  # (1, F, T)

        lbl_seg = np.zeros(self.seg_frames, dtype=np.int64)  # default 0 if needed
        lbl_seg[:valid_frames] = labels[start_frame:start_frame + valid_frames]
        # for frames beyond valid_frames (padding area), keep a sentinel (e.g., -100) or NO_CHORD idx
        # We assume base labels include NO_CHORD index; otherwise use -1
        if valid_frames < self.seg_frames:
            lbl_seg[valid_frames:] = labels[start_frame + valid_frames - 1]  # or NO_CHORD

        return torch.from_numpy(seg), torch.from_numpy(lbl_seg), valid_frames