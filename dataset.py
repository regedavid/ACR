import numpy as np
import torch
from torch.utils.data import Dataset
import mirdata
import mir_eval.chord as mrc
import math
import librosa
from util import map_to_majmin


def decompose_label(label):
    """Splits 'C:min7/G' into 'C', 'min7', 'G'."""
    if label == mrc.NO_CHORD:
        return "N", "N", "N"
    try:
        root, quality, intervals, bass = mrc.split(label)
        
        # 2. Handle Empty Quality
        # If mir_eval returns empty string, map it to 'N' (or 'open', 'none')
        qual_str = quality if quality != "" else "N"
        
        # 3. Handle Empty Bass (mir_eval usually defaults to '1', but just in case)
        bass_str = bass if bass != "" else "1"
        
        return root, qual_str, bass_str
        
    except Exception:
        # Fallback for parsing errors
        return "N", "N", "N"

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
    def __init__(self, root, fps=100, multi_target=False):
        self.multi_target = multi_target
        self.dataset = mirdata.initialize("beatles", data_home=root)
        self.fps = fps

        all_track_ids = self.dataset.track_ids

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
                print(f"Loaded audio for track {tid} with sr={sr}")
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
            
        self.sample_rate = self._audio_arrays[self.track_ids[0]][1]  # assume uniform sr across tracks

        print(f"Loaded {len(self.track_ids)} tracks with valid audio.")

        all_labels = set()
        for tid in self.track_ids:
            track = self.dataset.track(tid)
            all_labels.update(track.chords.labels)

        all_labels.add(mrc.NO_CHORD)
        all_labels = sorted(all_labels)

        if self.multi_target:
            # Build 3 separate Vocabs
            roots, qualities, basses = set(), set(), set()
            for lab in all_labels:
                r, q, b = decompose_label(lab)
                roots.add(r)
                qualities.add(q)
                basses.add(b)
            
            roots.add("N")
            qualities.add("N")
            basses.add("N")
            
            self.root2idx = {r: i for i, r in enumerate(sorted(roots))}
            self.qual2idx = {q: i for i, q in enumerate(sorted(qualities))}
            self.bass2idx = {b: i for i, b in enumerate(sorted(basses))}
            
            self.idx2root = {i: r for r, i in self.root2idx.items()}
            self.idx2qual = {i: q for q, i in self.qual2idx.items()}
            self.idx2bass = {i: b for b, i in self.bass2idx.items()}
            
            self.n_roots = len(self.root2idx)
            self.n_qualities = len(self.qual2idx)
            self.n_bass = len(self.bass2idx)
            print(f"Multi-Target Vocabs: {self.n_roots} Roots, {self.n_qualities} Qualities, {self.n_bass} Basses")
        else:
            self.label_to_idx = {lab: i for i, lab in enumerate(all_labels)}
            self.idx_to_label = {i: lab for lab, i in self.label_to_idx.items()}
            self.n_classes = len(all_labels)

        print(f"Chord vocabulary size: {self.n_classes} classes.")

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
        #y = self.pad(y, self.max_samples).astype(np.float32)

        # Chord intervals
        intervals = track.chords.intervals
        labels = track.chords.labels

        # Frame-wise time grid
        duration = len(y) / sr
        n_frames = math.ceil(duration * self.fps)
        frame_times = np.arange(n_frames) / self.fps

        # start with NO_CHORD everywhere
        no_idx = self.label_to_idx[mrc.NO_CHORD]

        frame_labels = np.full(n_frames, no_idx, dtype=np.int64)

        if self.multi_target:
            # Create 3 tensor arrays
            root_labels = np.zeros(n_frames, dtype=np.int64)
            qual_labels = np.zeros(n_frames, dtype=np.int64)
            bass_labels = np.zeros(n_frames, dtype=np.int64)
            
            # Fill them
            for (start, end), lab in zip(intervals, labels):
                r_str, q_str, b_str = decompose_label(lab)
                mask = (frame_times >= start) & (frame_times < end)
                
                root_labels[mask] = self.root2idx.get(r_str, 0)
                qual_labels[mask] = self.qual2idx.get(q_str, 0)
                bass_labels[mask] = self.bass2idx.get(b_str, 0)

            return torch.from_numpy(y), (torch.from_numpy(root_labels), torch.from_numpy(qual_labels), torch.from_numpy(bass_labels))
        else:
            # Fill interval labels
            for (start, end), lab in zip(intervals, labels):
                idx_c = self.label_to_idx.get(lab, no_idx)
                mask = (frame_times >= start) & (frame_times < end)
                frame_labels[mask] = idx_c

            return (
                torch.from_numpy(y),
                torch.from_numpy(frame_labels),
            )


class BeatlesMajMinChordDataset(Dataset):
    def __init__(self, root, fps=100):
        self.dataset = mirdata.initialize("beatles", data_home=root)
        self.fps = fps

        all_track_ids = self.dataset.track_ids

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
                print(f"Loaded audio for track {tid} with sr={sr}")
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
            
        self.sample_rate = self._audio_arrays[self.track_ids[0]][1]  # assume uniform sr across tracks

        print(f"Loaded {len(self.track_ids)} tracks with valid audio.")


        all_labels = set()
        for tid in self.track_ids:   # assuming your mirdata dataset
            track = self.dataset.track(tid)
            for lab in track.chords.labels:
                mapped = map_to_majmin(lab)
                if mapped is not None:  # ignore chords that cannot be mapped
                    all_labels.add(mapped)

        all_labels.add(mrc.NO_CHORD)
        all_labels = sorted(all_labels)

        # numeric encodings
        # roots, bitmaps, basses = mrc.encode_many(all_labels)
        # encodings = []
        # for r, b, ba in zip(roots, bitmaps, basses):
        #     encodings.append((int(r), tuple(b.tolist()), int(ba)))

        # self.encoding_to_idx = {enc: i for i, enc in enumerate(encodings)}
        # self.label_to_encoding = dict(zip(all_labels, encodings))
        
        self.label_to_idx = {lab: i for i, lab in enumerate(all_labels)}
        self.idx_to_label = {i: lab for lab, i in self.label_to_idx.items()}
        self.n_classes = len(all_labels)

        print(f"Chord vocabulary size: {self.n_classes} classes.")

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
        no_idx = self.label_to_idx[mrc.NO_CHORD]

        frame_labels = np.full(n_frames, no_idx, dtype=np.int64)

        # Fill interval labels
        for (start, end), lab in zip(intervals, labels):
            idx_c = self.label_to_idx.get(map_to_majmin(lab), no_idx)
            mask = (frame_times >= start) & (frame_times < end)
            frame_labels[mask] = idx_c

        return (
            torch.from_numpy(y),
            torch.from_numpy(frame_labels),
        )


class SegmentWrapper(torch.utils.data.Dataset):
    """
    Wrap a dataset that returns (audio, labels) and split each track
    into fixed-length overlapping or non-overlapping segments.
    """
    def __init__(self, base_dataset, segment_seconds=8.0, hop_seconds=None):
        self.base = base_dataset
        self.fps = base_dataset.fps
        self.sample_rate = base_dataset.sample_rate
        self.seg_samples = int(segment_seconds * base_dataset.sample_rate)
        self.hop_samples = int(hop_seconds * base_dataset.sample_rate) if hop_seconds is not None else self.seg_samples // 2
        self.seg_frames = int(segment_seconds * self.base.fps)
        self.hop_frames = int(hop_seconds * self.base.fps) if hop_seconds is not None else self.seg_frames // 2
        # (dataset_index, start_sample)
        self.index_map = []

        for tid in range(len(base_dataset)):
            audio, labels = base_dataset[tid]
            total_samples = len(audio)

            for start_sample in range(0, total_samples, self.hop_samples):
                start_frame = int(start_sample * self.fps / self.sample_rate)

                # stop if we exceed the label length entirely
                if start_frame >= len(labels):
                    break

                self.index_map.append((tid, start_sample, start_frame))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        tid, start_sample, start_frame = self.index_map[idx]

        audio, labels = self.base[tid]

        end_sample = start_sample + self.seg_samples
        end_frame = start_frame + self.seg_frames
        # Pad if segment exceeds track length
        audio_seg = audio[start_sample:end_sample]
        if len(audio_seg) < self.seg_samples:
            pad = self.seg_samples - len(audio_seg)
            audio_seg = torch.nn.functional.pad(audio_seg, (0, pad), mode="constant", value=0.0)

        if isinstance(labels, tuple): # Multi-target case
            root_seg, qual_seg, bass_seg = labels
            
            # Helper to slice and pad
            def slice_pad(tensor_data):
                seg = tensor_data[start_frame:end_frame]
                if len(seg) < self.seg_frames:
                    pad = self.seg_frames - len(seg)
                    # Use 0 (usually No Chord) for padding
                    seg = torch.nn.functional.pad(seg, (0, pad), mode="constant", value=0)
                return seg
            
            return audio_seg, (slice_pad(root_seg), slice_pad(qual_seg), slice_pad(bass_seg))
        else:
            label_seg = labels[start_frame:end_frame]
            if len(label_seg) < self.seg_frames:
                pad = self.seg_frames - len(label_seg)
                label_seg = torch.nn.functional.pad(label_seg, (0, pad), mode="constant", value=self.base.label_to_idx[mrc.NO_CHORD])

            return audio_seg, label_seg