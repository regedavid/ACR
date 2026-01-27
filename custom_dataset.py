import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset
from dataset import BeatlesChordDataset
import librosa
import os
import mir_eval.chord as mrc
import mir_eval.io
import numpy as np
import tempfile
from dataset import decompose_label

def clean_lab_file(lab_path):
    """
    Read .lab file, remove empty rows, and return as string.
    Can be written to temp file for mir_eval.io.load_labeled_intervals.
    """
    with open(lab_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    
    # Filter out empty/whitespace-only lines
    non_empty_lines = [ln for ln in lines if ln.strip()]
    
    return "".join(non_empty_lines)

def collect_external_labels(root_dir):
    """
    Scan external dataset and collect all unique chord labels.
    Returns: set of label strings
    """
    all_labels = set()
    
    for entry in os.scandir(root_dir):
        if not entry.is_dir():
            continue
        subdir = entry.path
        
        # Find .lab file (prefer full.lab)
        lab_path = None
        for fn in os.listdir(subdir):
            if fn.endswith(".lab"):
                lab_path = os.path.join(subdir, fn)
                if fn == "full.lab":
                    break
        
        if lab_path is None:
            continue
        
        try:
            cleaned_content = clean_lab_file(lab_path)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.lab', delete=False) as tmp:
                tmp.write(cleaned_content)
                tmp_path = tmp.name
            
            _, labels = mir_eval.io.load_labeled_intervals(tmp_path)
            os.unlink(tmp_path)
            all_labels.update(labels)
        except Exception:
            continue
    
    return all_labels

class ExternalChordDataset(Dataset):
    def __init__(self, root_dir, fps=100, sample_rate=44100, label_to_idx=None, multi_target=False, root2idx=None, qual2idx=None, bass2idx=None):
        """
        root_dir: directory containing subdirs with audio + full.lab
        label_to_idx: mapping from string chord label to integer index
        """
        self.root_dir = root_dir
        self.fps = fps
        self.sample_rate = sample_rate
        self.items = []
        self.multi_target = multi_target
        
        if self.multi_target:
            self.root2idx = root2idx
            self.qual2idx = qual2idx
            self.bass2idx = bass2idx
            
            self.n_root_idx = self.root2idx.get("N", 0)
            self.n_qual_idx = self.qual2idx.get("N", 0) 
            self.n_bass_idx = self.bass2idx.get("N", 0)
    
        else:
            self.label_to_idx = label_to_idx or {mrc.NO_CHORD: 0}
            self.no_chord_idx = self.label_to_idx.get(mrc.NO_CHORD, 0)
        
        for entry in os.scandir(root_dir):
            if not entry.is_dir():
                continue
            subdir = entry.path
            
            # Find .lab file (prefer full.lab)
            lab_path = None
            for fn in os.listdir(subdir):
                if fn.endswith(".lab"):
                    lab_path = os.path.join(subdir, fn)
                    if fn == "full.lab":
                        break
            
            # Find audio file
            audio_path = None
            for fn in os.listdir(subdir):
                if fn.lower().endswith(".wav"):
                    audio_path = os.path.join(subdir, fn)
                    break
            
            if lab_path is None or audio_path is None:
                continue
            
            try:
                # Clean lab file and load intervals
                cleaned_content = clean_lab_file(lab_path)
                # Write to temp file for mir_eval
                with tempfile.NamedTemporaryFile(mode='w', suffix='.lab', delete=False) as tmp:
                    tmp.write(cleaned_content)
                    tmp_path = tmp.name
                
                intervals, labels = mir_eval.io.load_labeled_intervals(tmp_path)
                os.unlink(tmp_path)
                
                if self.multi_target:
                    # Decompose and Convert to 3 indices
                    chords = []
                    for start, end, lab in zip(intervals[:, 0], intervals[:, 1], labels):
                        r, q, b = decompose_label(lab)
                        r_idx = self.root2idx.get(r, self.n_root_idx)
                        q_idx = self.qual2idx.get(q, self.n_qual_idx)
                        b_idx = self.bass2idx.get(b, self.n_bass_idx)
                        chords.append((start, end, r_idx, q_idx, b_idx))
                else:        
                    # Convert labels to indices
                    chords = [(start, end, self.label_to_idx.get(lab, self.no_chord_idx))
                            for start, end, lab in zip(intervals[:, 0], intervals[:, 1], labels)]
                
                self.items.append({
                    "audio_path": audio_path,
                    "chords": chords,
                })
            except Exception as e:
                print(f"Skipping {subdir}: {e}")
                continue

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        spec = self.items[idx]
        y, sr = librosa.load(spec["audio_path"], sr=self.sample_rate, mono=True)
        y = torch.from_numpy(y.astype(np.float32))
        
        duration = float(len(y)) / float(sr)
        n_frames = int(np.ceil(duration * self.fps))
        frame_times = np.arange(n_frames, dtype=np.float32) / self.fps
        
        if self.multi_target:
            # Create 3 tensors
            r_labels = torch.zeros(n_frames, dtype=torch.long)
            q_labels = torch.zeros(n_frames, dtype=torch.long)
            b_labels = torch.zeros(n_frames, dtype=torch.long)
            
            for start, end, r_idx, q_idx, b_idx in spec["chords"]:
                mask = (frame_times >= start) & (frame_times < end)
                r_labels[mask] = r_idx
                q_labels[mask] = q_idx
                b_labels[mask] = b_idx
            
            return y, (r_labels, q_labels, b_labels)
        else:
            labels = torch.full((n_frames,), fill_value=self.no_chord_idx, dtype=torch.long)
        
            for start, end, lab_idx in spec["chords"]:
                mask = (frame_times >= start) & (frame_times < end)
                labels[mask] = lab_idx
            
            return y, labels


class UnifiedConcatDataset(ConcatDataset):
    """
    ConcatDataset with shared vocabulary and dataset attributes.
    Makes it a drop-in replacement for BeatlesChordDataset.
    """
    def __init__(self,
                datasets,
                fps,
                sample_rate,
                label_to_idx=None, 
                idx_to_label=None,
                root2idx=None,
                qual2idx=None,
                bass2idx=None,
                idx2root=None,
                idx2qual=None,
                idx2bass=None,
            ):
        super().__init__(datasets)
        self.fps = fps
        self.sample_rate = sample_rate

        # --- Single Target Attributes ---
        self.label_to_idx = label_to_idx
        self.idx_to_label = idx_to_label
        if self.label_to_idx:
            self.n_classes = len(self.label_to_idx)
        else:
            self.n_classes = None

        # --- Multi Target Attributes ---
        self.root2idx = root2idx
        self.qual2idx = qual2idx
        self.bass2idx = bass2idx
        
        self.idx2root = idx2root
        self.idx2qual = idx2qual
        self.idx2bass = idx2bass

        if self.root2idx:
            # Counts
            self.n_roots = len(self.root2idx)
            self.n_qualities = len(self.qual2idx)
            self.n_bass = len(self.bass2idx)
            
            # Ignore Indices (Safe lookup for "N")
            self.n_root_idx = self.root2idx.get("N", 0)
            self.n_qual_idx = self.qual2idx.get("N", 0)
            self.n_bass_idx = self.bass2idx.get("N", 0)
        else:
            self.n_roots = None
            self.n_qualities = None
            self.n_bass = None


def pad_collate_fn(batch, pad_label_idx=0):
    """
    Collate function for batching variable-length audio and labels.
    Pads to the max length in the batch.
    
    Args:
        batch: list of (audio, labels) tuples
        pad_label_idx: value to use for padding labels (typically NO_CHORD index)
    
    Returns:
        (audio_batch, labels_batch): padded tensors of shape (B, max_audio_len) and (B, max_label_len)
    """
    audios, labels = zip(*batch)
    
    is_multi = isinstance(labels[0], (tuple, list))
    
    # Find max lengths in this batch
    max_audio_len = max(a.shape[-1] for a in audios)
    if is_multi:
        # labels is list of tuples: [(r1, q1, b1), (r2, q2, b2), ...]
        # We need to stack Roots, Quals, and Basses separately
        roots, quals, basses = zip(*labels)
        
        max_len = max(r.shape[-1] for r in roots) # Assume r, q, b have same len
        
        def pad_stack(tensor_list):
            padded = []
            for t in tensor_list:
                if t.shape[-1] < max_len:
                    t = F.pad(t, (0, max_len - t.shape[-1]), value=pad_label_idx)
                padded.append(t)
            return torch.stack(padded)

        padded_roots = pad_stack(roots)
        padded_quals = pad_stack(quals)
        padded_basses = pad_stack(basses)
        
        padded_labels = (padded_roots, padded_quals, padded_basses)
        
    else:
        # Standard Single Label
        max_label_len = max(l.shape[-1] for l in labels)
        padded_labels_list = []
        for label in labels:
            if label.shape[-1] < max_label_len:
                label = F.pad(label, (0, max_label_len - label.shape[-1]), value=pad_label_idx)
            padded_labels_list.append(label)
        padded_labels = torch.stack(padded_labels_list)

    # Pad Audio
    padded_audios = []
    for audio in audios:
        if audio.shape[-1] < max_audio_len:
            audio = F.pad(audio, (0, max_audio_len - audio.shape[-1]), value=0.0)
        padded_audios.append(audio)
    
    return torch.stack(padded_audios), padded_labels


def build_combined_dataset(beatles_root, external_root, fps=100, sample_rate=None, multi_target=False):
    # 1. Load Beatles
    beatles_ds = BeatlesChordDataset(root=beatles_root, fps=fps, multi_target=multi_target)
    
    # 2. Collect External Labels
    external_labels = collect_external_labels(external_root)
    ext_sr = sample_rate if sample_rate is not None else beatles_ds.sample_rate

    if multi_target:
        # --- Multi-Target Logic ---
        ext_roots, ext_quals, ext_basses = set(), set(), set()
        for lab in external_labels:
            r, q, b = decompose_label(lab)
            ext_roots.add(r); ext_quals.add(q); ext_basses.add(b)

        # Merge & Force "N"
        all_roots = set(beatles_ds.root2idx.keys()) | ext_roots | {"N"}
        all_quals = set(beatles_ds.qual2idx.keys()) | ext_quals | {"N"}
        all_basses = set(beatles_ds.bass2idx.keys()) | ext_basses | {"N"}

        # Create Maps
        root2idx = {r: i for i, r in enumerate(sorted(all_roots))}
        qual2idx = {q: i for i, q in enumerate(sorted(all_quals))}
        bass2idx = {b: i for i, b in enumerate(sorted(all_basses))}
        
        idx2root = {i: r for r, i in root2idx.items()}
        idx2qual = {i: q for q, i in qual2idx.items()}
        idx2bass = {i: b for b, i in bass2idx.items()}

        # Update Beatles
        beatles_ds.root2idx = root2idx
        beatles_ds.qual2idx = qual2idx
        beatles_ds.bass2idx = bass2idx
        # (We don't strictly need to update beatles_ds.n_roots etc. anymore 
        #  because we read from 'combined', but it's good practice)
        beatles_ds.n_root_idx = root2idx["N"]
        beatles_ds.n_qual_idx = qual2idx["N"]
        beatles_ds.n_bass_idx = bass2idx["N"]

        # Create External
        extra_ds = ExternalChordDataset(
            external_root, fps=fps, sample_rate=ext_sr,
            multi_target=True,
            root2idx=root2idx, qual2idx=qual2idx, bass2idx=bass2idx
        )

        # Create Unified Wrapper (PASS EVERYTHING HERE)
        combined = UnifiedConcatDataset(
            datasets=[beatles_ds, extra_ds],
            fps=fps,
            sample_rate=ext_sr,
            root2idx=root2idx, qual2idx=qual2idx, bass2idx=bass2idx,
            idx2root=idx2root, idx2qual=idx2qual, idx2bass=idx2bass
        )
        
        return combined, root2idx, qual2idx, bass2idx

    else:
        # --- Single-Target Logic ---
        all_labels = sorted(set(beatles_ds.label_to_idx.keys()) | external_labels)
        label_to_idx = {lab: i for i, lab in enumerate(all_labels)}
        idx_to_label = {i: lab for lab, i in label_to_idx.items()}

        beatles_ds.label_to_idx = label_to_idx
        beatles_ds.idx_to_label = idx_to_label

        extra_ds = ExternalChordDataset(
            external_root, fps=fps, sample_rate=ext_sr,
            multi_target=False, label_to_idx=label_to_idx,
        )

        combined = UnifiedConcatDataset(
            datasets=[beatles_ds, extra_ds],
            fps=fps,
            sample_rate=ext_sr,
            label_to_idx=label_to_idx,
            idx_to_label=idx_to_label
        )
        
        return combined, label_to_idx, idx_to_label, None


if __name__ == "__main__":
    combined, l2i, i2l = build_combined_dataset(
        beatles_root="mir_datasets2/beatles",
        external_root="dataset_eval",
        fps=100,
        sample_rate=None,
    )
    print(f"Chord vocab size: {len(l2i)}")
    beatles_len = len(combined.datasets[0])
    external_len = len(combined.datasets[1])
    print(f"Beatles tracks: {beatles_len}, External tracks: {external_len}")

    b_audio, b_labels = combined.datasets[0][0]
    e_audio, e_labels = combined.datasets[1][0]
    print("Beatles sample:", b_audio.shape, b_labels.shape, "labels min/max", b_labels.min().item(), b_labels.max().item())
    print("External sample:", e_audio.shape, e_labels.shape, "labels min/max", e_labels.min().item(), e_labels.max().item())

    c_audio, c_labels = combined[0]
    c_audio2, c_labels2 = combined[beatles_len]  # first external sample
    print("Combined[0]:", c_audio.shape, c_labels.shape)
    print("Combined[first external]:", c_audio2.shape, c_labels2.shape)