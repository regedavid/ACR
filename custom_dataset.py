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
    def __init__(self, root_dir, fps=100, sample_rate=44100, label_to_idx=None):
        """
        root_dir: directory containing subdirs with audio + full.lab
        label_to_idx: mapping from string chord label to integer index
        """
        self.root_dir = root_dir
        self.fps = fps
        self.sample_rate = sample_rate
        self.items = []
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
    def __init__(self, datasets, label_to_idx, idx_to_label, fps, sample_rate):
        super().__init__(datasets)
        self.label_to_idx = label_to_idx
        self.idx_to_label = idx_to_label
        self.n_classes = len(label_to_idx)
        self.fps = fps
        self.sample_rate = sample_rate


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
    
    # Find max lengths in this batch
    max_audio_len = max(a.shape[-1] for a in audios)
    max_label_len = max(l.shape[-1] for l in labels)
    
    # Pad each sample
    padded_audios = []
    padded_labels = []
    
    for audio, label in zip(audios, labels):
        if audio.shape[-1] < max_audio_len:
            audio = F.pad(audio, (0, max_audio_len - audio.shape[-1]), value=0.0)
        if label.shape[-1] < max_label_len:
            label = F.pad(label, (0, max_label_len - label.shape[-1]), value=pad_label_idx)
        
        padded_audios.append(audio)
        padded_labels.append(label)
    
    return torch.stack(padded_audios), torch.stack(padded_labels)


def build_combined_dataset(beatles_root, external_root, fps=100, sample_rate=None):
    """
    Build Beatles + external chord datasets with a unified vocabulary.

    Returns:
        combined (ConcatDataset): Beatles + external datasets sharing label mapping
        label_to_idx (dict): unified chord label -> index
        idx_to_label (dict): index -> chord label
    """
    # Load Beatles first to get its labels and sample rate
    beatles_ds = BeatlesChordDataset(root=beatles_root, fps=fps)

    # Collect labels from external labs
    external_labels = collect_external_labels(external_root)

    # Merge vocabularies
    all_labels = sorted(set(beatles_ds.label_to_idx.keys()) | external_labels)
    label_to_idx = {lab: i for i, lab in enumerate(all_labels)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}

    # Remap Beatles dataset to unified vocab
    beatles_ds.label_to_idx = label_to_idx
    beatles_ds.idx_to_label = idx_to_label
    beatles_ds.n_classes = len(label_to_idx)

    # Use provided sample_rate or Beatles sample rate for external audio
    ext_sr = sample_rate if sample_rate is not None else beatles_ds.sample_rate

    # Build external dataset with unified vocab
    extra_ds = ExternalChordDataset(
        external_root,
        fps=fps,
        sample_rate=ext_sr,
        label_to_idx=label_to_idx,
    )

    combined = UnifiedConcatDataset(
        datasets=[beatles_ds, extra_ds],
        label_to_idx=label_to_idx,
        idx_to_label=idx_to_label,
        fps=fps,
        sample_rate=ext_sr,
    )
    return combined, label_to_idx, idx_to_label


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