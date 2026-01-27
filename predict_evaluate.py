"""
Prediction and evaluation module for the chord recognition model.

Provides:
- ChordPredictor: Load trained model and make predictions on audio
- ChordEvaluator: Evaluate predictions against reference labels using mir_eval.chord metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import os
import json
import librosa
from typing import Dict, List, Tuple, Optional, Union
import mir_eval.chord as mrc
import mirdata
from pathlib import Path

from cnn_acr import CNNTransformerChordModel, HMMDecoder
from dataset import BeatlesChordDataset, BeatlesMajMinChordDataset
from train_pl import LightningChordModel
import pytorch_lightning as pl
from hmm_utils import get_transition_matrix, make_transition_matrix_sticky
from custom_dataset import build_combined_dataset, UnifiedConcatDataset


class ChordPredictor:
    """
    Wrapper for making chord predictions on audio.
    
    Loads a trained model (from PyTorch Lightning checkpoint or direct model)
    and provides methods to predict chord labels from raw audio.
    """
    
    def __init__(
        self,
        model: Union[str, CNNTransformerChordModel],
        label_to_idx: Dict[str, int],
        idx_to_label: Dict[int, str],
        transition_matrix: Optional[np.ndarray] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            model: Path to checkpoint (.ckpt) or a loaded CNNTransformerChordModel
            label_to_idx: Mapping from chord label to index
            idx_to_label: Mapping from index to chord label
            transition_matrix: Optional transition matrix for HMM decoding
            device: "cuda" or "cpu"
        """
        self.device = device
        self.label_to_idx = label_to_idx
        self.idx_to_label = idx_to_label
        self.transition_matrix = transition_matrix
        
        # Load model if path provided
        if isinstance(model, str):
            self.model = self._load_checkpoint(model)
        else:
            self.model = model
        
        self.model = self.model.to(device)
        self.model.eval()
    
    @staticmethod
    def _load_checkpoint(checkpoint_path: str) -> CNNTransformerChordModel:
        """Load model from a Lightning checkpoint using the correct Lightning module class."""
        # Map to CPU on load to avoid GPU requirements during init; we'll move to device later
        lightning_module = LightningChordModel.load_from_checkpoint(
            checkpoint_path,
            map_location=torch.device("cpu"),
        )
        return lightning_module.model
    
    def predict(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on audio.
        
        Args:
            audio: Audio waveform (float32, mono)
            sr: Sample rate
        
        Returns:
            (predicted_labels, confidence_scores) where:
                - predicted_labels: (n_frames,) array of chord label indices
                - confidence_scores: (n_frames,) array of softmax confidences [0-1]
        """
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        
        with torch.no_grad():
            logits = self.model(audio_tensor)  # (T, n_classes) or (B, T, n_classes)
            
            # Handle batch dimension
            if logits.dim() == 3:
                logits = logits.squeeze(0)  # (T, n_classes)
            
            # Get predictions and confidence
            probs = F.softmax(logits, dim=-1)  # (T, n_classes)
            confidences, predicted_indices = torch.max(probs, dim=-1)
        
        predicted_indices = predicted_indices.cpu().numpy()
        confidences = confidences.cpu().numpy()
        
        return predicted_indices, confidences

    import librosa

    def predict_sliding(
        self,
        audio: np.ndarray,
        sr: int,
        segment_seconds: float = 8.0,
        hop_seconds: float = 4.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sliding-window inference with optional HMM smoothing.
        """
        model_sr = getattr(self.model.frontend, "sr", sr)
        if sr != model_sr:
            raise ValueError(f"Audio sample rate {sr} does not match model sample rate {model_sr}.")

        hop_len = self.model.frontend.hop_length
        n_classes = self.model._n_classes

        seg_samples = int(round(segment_seconds * model_sr))
        hop_samples = int(round(hop_seconds * model_sr))
        total_frames = math.ceil(len(audio) / hop_len)

        # Accumulators for overlap-add of logits
        accum_logits = torch.zeros((total_frames, n_classes), device=self.device)
        counts = torch.zeros(total_frames, device=self.device)

        audio_tensor = torch.from_numpy(audio).float().to(self.device)

        with torch.no_grad():
            for start in range(0, len(audio), hop_samples):
                end = start + seg_samples
                chunk = audio_tensor[start:end]
                if chunk.numel() == 0:
                    break
                if chunk.numel() < seg_samples:
                    pad = seg_samples - chunk.numel()
                    chunk = F.pad(chunk, (0, pad))

                logits = self.model(chunk)
                if logits.dim() == 3:
                    logits = logits.squeeze(0)

                T_chunk = logits.shape[0]
                start_frame = start // hop_len
                end_frame = min(start_frame + T_chunk, total_frames)

                span = end_frame - start_frame
                if span <= 0:
                    continue

                accum_logits[start_frame:end_frame] += logits[:span]
                counts[start_frame:end_frame] += 1

        # Normalize logits
        mask = counts > 0
        averaged_logits = torch.zeros_like(accum_logits)
        averaged_logits[mask] = accum_logits[mask] / counts[mask].unsqueeze(1)

        # Get Probabilities
        probs = F.softmax(averaged_logits, dim=-1) # Shape: (Total_Frames, n_classes)
        
        if self.transition_matrix is not None:
            # HMM Viterbi Decoding
            # Librosa expects: (n_states, n_steps) -> We must transpose probs
            emission_probs = probs.cpu().numpy().T 
            
            # Run Viterbi (finds the global optimal path)
            predicted_indices = librosa.sequence.viterbi(emission_probs, self.transition_matrix)
            
            # Use the raw probabilities of the chosen path as 'confidence'
            # (Or simpler: just return the max probability for that frame)
            confidences, _ = torch.max(probs, dim=-1)
            confidences = confidences.cpu().numpy()
            
        else:
            # Standard Greedy Prediction (Noisy)
            confidences, predicted_indices = torch.max(probs, dim=-1)
            predicted_indices = predicted_indices.cpu().numpy()
            confidences = confidences.cpu().numpy()

        return predicted_indices, confidences
    
    def predict_to_labels(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions and convert to chord labels.
        
        Args:
            audio: Audio waveform
            sr: Sample rate
        
        Returns:
            (predicted_labels, confidence_scores) where:
                - predicted_labels: (n_frames,) array of chord label strings
                - confidence_scores: (n_frames,) array of confidence scores
        """
        pred_indices, confidences = self.predict(audio, sr)
        pred_labels = np.array([self.idx_to_label[idx] for idx in pred_indices])
        return pred_labels, confidences
    
    def predict_from_track(
        self, 
        dataset: Union[BeatlesChordDataset, BeatlesMajMinChordDataset],
        track_id: str,
        use_sliding: bool = True,
        segment_seconds: float = 8.0,
        hop_seconds: float = 4.0
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Make predictions on a Beatles track from the dataset.
        
        Args:
            dataset: BeatlesChordDataset or BeatlesMajMinChordDataset instance
            track_id: Track ID (e.g., "0001_-_Please_Please_Me")
            use_sliding: If True, run sliding-window inference (recommended for full songs)
            segment_seconds: Window length (used when use_sliding=True)
            hop_seconds: Hop length (used when use_sliding=True)
        
        Returns:
            (predicted_labels, confidence_scores, duration_seconds)
        """
        track = dataset.dataset.track(track_id)
        audio_data = track.audio
        
        # Handle different audio formats
        if isinstance(audio_data, (tuple, list)):
            y, sr = audio_data[0], audio_data[1]
        else:
            y = audio_data
            sr = getattr(track, "sample_rate", None) or 22050
        
        y = np.asarray(y).astype(np.float32)
        
        # Normalize if needed
        if np.max(np.abs(y)) > 1.0:
            y = y / np.max(np.abs(y))
        
        if use_sliding:
            pred_indices, confidences = self.predict_sliding(
                y, sr, segment_seconds=segment_seconds, hop_seconds=hop_seconds
            )
            pred_labels = np.array([self.idx_to_label[idx] for idx in pred_indices])
        else:
            pred_labels, confidences = self.predict_to_labels(y, sr)
        duration = len(y) / sr
        
        return pred_labels, confidences, duration
    
    def predict_from_audio_file(
        self,
        audio_path: str,
        target_sr: int = 44100,
        use_sliding: bool = True,
        segment_seconds: float = 8.0,
        hop_seconds: float = 4.0
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Make predictions on an audio file (supports any format librosa can load).
        
        Args:
            audio_path: Path to audio file (mp3, wav, flac, etc.)
            target_sr: Target sample rate for loading (should match model's expected sr)
            use_sliding: If True, run sliding-window inference (recommended)
            segment_seconds: Window length (used when use_sliding=True)
            hop_seconds: Hop length (used when use_sliding=True)
        
        Returns:
            (predicted_labels, confidence_scores, duration_seconds)
        """
        # Load audio file
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        y = y.astype(np.float32)
        
        # Normalize if needed
        if np.max(np.abs(y)) > 1.0:
            y = y / np.max(np.abs(y))
        
        if use_sliding:
            pred_indices, confidences = self.predict_sliding(
                y, sr, segment_seconds=segment_seconds, hop_seconds=hop_seconds
            )
            pred_labels = np.array([self.idx_to_label[idx] for idx in pred_indices])
        else:
            pred_labels, confidences = self.predict_to_labels(y, sr)
        
        duration = len(y) / sr
        
        return pred_labels, confidences, duration


class ChordEvaluator:
    """
    Evaluate chord predictions against reference labels using mir_eval.chord metrics.
    
    Provides various evaluation metrics such as:
    - mirex (overall accuracy)
    - root (root accuracy)
    - majmin (major/minor accuracy)
    - weighted_accuracy with custom weighting
    - And all other mir_eval.chord comparison functions
    """
    
    def __init__(self, fps: int = 100):
        """
        Args:
            fps: Frames per second (for converting frame indices to time intervals)
        """
        self.fps = fps
    
    def frames_to_intervals(
        self, 
        frame_labels: np.ndarray, 
        fps: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert frame-level labels to time intervals.
        
        Args:
            frame_labels: (n_frames,) array of chord labels
            fps: Frames per second (defaults to self.fps)
        
        Returns:
            (intervals, labels) where:
                - intervals: (n_segments, 2) array of [start_time, end_time]
                - labels: (n_segments,) array of chord labels for each interval
        """
        if fps is None:
            fps = self.fps
        
        # Merge consecutive identical frames
        intervals = []
        labels = []
        
        current_label = frame_labels[0]
        start_frame = 0
        
        for i in range(1, len(frame_labels)):
            if frame_labels[i] != current_label:
                # End current interval
                end_frame = i
                intervals.append([start_frame / fps, end_frame / fps])
                labels.append(current_label)
                
                # Start new interval
                current_label = frame_labels[i]
                start_frame = i
        
        # Add final interval
        intervals.append([start_frame / fps, len(frame_labels) / fps])
        labels.append(current_label)
        
        return np.array(intervals), np.array(labels)
    
    def evaluate_mirex(
        self,
        ref_intervals: np.ndarray,
        ref_labels: np.ndarray,
        est_intervals: np.ndarray,
        est_labels: np.ndarray,
    ) -> float:
        """
        MIREX chord accuracy (highest level comparison).
        
        Args:
            ref_intervals: Reference chord intervals (n, 2)
            ref_labels: Reference chord labels (n,)
            est_intervals: Estimated chord intervals (m, 2)
            est_labels: Estimated chord labels (m,)
        
        Returns:
            Accuracy score (0-1)
        """
        score, _ = mrc.mirex(ref_labels, est_labels)
        return score
    
    def evaluate_root(
        self,
        ref_intervals: np.ndarray,
        ref_labels: np.ndarray,
        est_intervals: np.ndarray,
        est_labels: np.ndarray,
    ) -> float:
        """Root accuracy: only considers the root note."""
        score, _ = mrc.root(ref_labels, est_labels)
        return score
    
    def evaluate_majmin(
        self,
        ref_intervals: np.ndarray,
        ref_labels: np.ndarray,
        est_intervals: np.ndarray,
        est_labels: np.ndarray,
    ) -> float:
        """Major/minor accuracy: root + major/minor quality."""
        score, _ = mrc.majmin(ref_labels, est_labels)
        return score
    
    def evaluate_sevenths(
        self,
        ref_intervals: np.ndarray,
        ref_labels: np.ndarray,
        est_intervals: np.ndarray,
        est_labels: np.ndarray,
    ) -> float:
        """Seventh chords accuracy."""
        score, _ = mrc.sevenths(ref_labels, est_labels)
        return score
    
    def evaluate_thirds(
        self,
        ref_intervals: np.ndarray,
        ref_labels: np.ndarray,
        est_intervals: np.ndarray,
        est_labels: np.ndarray,
    ) -> float:
        """Triads (3-note chords) accuracy."""
        score, _ = mrc.triads(ref_labels, est_labels)
        return score
    
    def evaluate_all(
        self,
        ref_intervals: np.ndarray,
        ref_labels: np.ndarray,
        est_intervals: np.ndarray,
        est_labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute all available metrics.
        
        Returns:
            Dictionary with all metric scores
        """
        metrics = {}
        
        try:
            metrics["mirex"] = self.evaluate_mirex(ref_intervals, ref_labels, est_intervals, est_labels)
        except Exception as e:
            print(f"Warning: mirex evaluation failed: {e}")
            metrics["mirex"] = None
        
        try:
            metrics["root"] = self.evaluate_root(ref_intervals, ref_labels, est_intervals, est_labels)
        except Exception as e:
            print(f"Warning: root evaluation failed: {e}")
            metrics["root"] = None
        
        try:
            metrics["majmin"] = self.evaluate_majmin(ref_intervals, ref_labels, est_intervals, est_labels)
        except Exception as e:
            print(f"Warning: majmin evaluation failed: {e}")
            metrics["majmin"] = None
        
        try:
            metrics["sevenths"] = self.evaluate_sevenths(ref_intervals, ref_labels, est_intervals, est_labels)
        except Exception as e:
            print(f"Warning: sevenths evaluation failed: {e}")
            metrics["sevenths"] = None
        
        try:
            metrics["triads"] = self.evaluate_thirds(ref_intervals, ref_labels, est_intervals, est_labels)
        except Exception as e:
            print(f"Warning: triads evaluation failed: {e}")
            metrics["triads"] = None
        
        return metrics
    
    def evaluate_from_intervals(
        self,
        ref_intervals: np.ndarray,
        ref_labels: np.ndarray,
        est_intervals: np.ndarray,
        est_labels: np.ndarray,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate using time intervals directly (no frame conversion).
        
        Uses mir_eval.chord.evaluate() which handles interval alignment automatically.
        This avoids length mismatch errors when comparing predictions to references.
        
        Args:
            ref_intervals: Reference chord intervals (n, 2)
            ref_labels: Reference chord labels (n,)
            est_intervals: Estimated chord intervals (m, 2)
            est_labels: Estimated chord labels (m,)
            metrics: List of metric names to compute. If None, compute all.
        
        Returns:
            Dictionary of metric scores
        """
        # Validate and convert inputs
        ref_intervals = np.asarray(ref_intervals)
        ref_labels = np.asarray(ref_labels)
        est_intervals = np.asarray(est_intervals)
        est_labels = np.asarray(est_labels)
        
        # Convert labels to lists (mir_eval expects lists, not numpy arrays)
        ref_labels = ref_labels.tolist() if isinstance(ref_labels, np.ndarray) else ref_labels
        est_labels = est_labels.tolist() if isinstance(est_labels, np.ndarray) else est_labels
        
        # Validate shapes
        if ref_intervals.size == 0 or est_intervals.size == 0:
            print("Warning: Empty intervals provided. Returning empty scores.")
            return {}
        
        if ref_intervals.ndim != 2 or est_intervals.ndim != 2:
            print("Warning: Intervals must be 2D arrays (n, 2). Reshaping...")
            if ref_intervals.ndim == 1:
                ref_intervals = ref_intervals.reshape(-1, 2)
            if est_intervals.ndim == 1:
                est_intervals = est_intervals.reshape(-1, 2)
        
        if len(ref_labels) != len(ref_intervals):
            print(f"Warning: Reference labels ({len(ref_labels)}) and intervals ({len(ref_intervals)}) length mismatch.")
            return {}
        
        if len(est_labels) != len(est_intervals):
            print(f"Warning: Estimated labels ({len(est_labels)}) and intervals ({len(est_intervals)}) length mismatch.")
            return {}
        
        # Use mir_eval's built-in evaluate function which handles interval alignment
        try:
            # est_intervals, est_labels = mrc.util.adjust_intervals(
            #     est_intervals, est_labels, 
            #     ref_intervals.min(), ref_intervals.max(),
            #     mrc.NO_CHORD, mrc.NO_CHORD
            # )
            scores = mrc.evaluate(ref_intervals, ref_labels, est_intervals, est_labels)
        except Exception as e:
            print(f"Warning: evaluate() failed: {e}. Trying individual metrics...")
            scores = self._evaluate_individual_metrics(ref_intervals, ref_labels, est_intervals, est_labels)
        
        # If metrics specified, return subset; otherwise return all from evaluate
        if metrics is None:
            return scores
        
        # Filter to requested metrics
        results = {}
        for metric in metrics:
            if metric in scores:
                results[metric] = scores[metric]
            else:
                print(f"Warning: Metric '{metric}' not in evaluate output")
        
        return results
    
    @staticmethod
    def _evaluate_individual_metrics(
        ref_intervals: np.ndarray,
        ref_labels: List[str],
        est_intervals: np.ndarray,
        est_labels: List[str],
    ) -> Dict[str, float]:
        """
        Compute individual chord metrics when the main evaluate() function fails.
        
        Args:
            ref_intervals: Reference chord intervals (n, 2)
            ref_labels: Reference chord labels (n,)
            est_intervals: Estimated chord intervals (m, 2)
            est_labels: Estimated chord labels (m,)
        
        Returns:
            Dictionary of individually computed metric scores
        """
        scores = {}
        metric_functions = [
            ('mirex', mrc.mirex),
            ('root', mrc.root),
            ('majmin', mrc.majmin),
            ('triads', mrc.triads),
            ('thirds', mrc.thirds),
            ('tetrads', mrc.tetrads),
            ('sevenths', mrc.sevenths),
        ]
        
        for metric_name, metric_func in metric_functions:
            try:
                score, _ = metric_func(ref_labels, est_labels)
                scores[metric_name] = score
                print(f"  ✓ {metric_name}: {score:.4f}")
            except Exception as e:
                print(f"  ✗ {metric_name} failed: {e}")
                scores[metric_name] = None
        
        # Compute segmentation metrics if possible
        seg_metrics = [
            ('seg', mrc.seg),
            ('underseg', mrc.underseg),
            ('overseg', mrc.overseg),
        ]
        
        for metric_name, metric_func in seg_metrics:
            try:
                score = metric_func(ref_intervals, est_intervals)
                scores[metric_name] = score
                print(f"  ✓ {metric_name}: {score:.4f}")
            except Exception as e:
                print(f"  ✗ {metric_name} failed: {e}")
                scores[metric_name] = None
        
        return scores
    
    @staticmethod
    def write_lab_file(
        intervals: np.ndarray,
        labels: np.ndarray,
        output_path: str,
    ) -> None:
        """
        Write chord predictions to a .lab file format.
        
        Args:
            intervals: (n_segments, 2) array of [start_time, end_time] in seconds
            labels: (n_segments,) array of chord label strings
            output_path: Path to save the .lab file
        """
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w") as f:
            for (start, end), label in zip(intervals, labels):
                f.write(f"{start:.6f}\t{end:.6f}\t{label}\n")
        
        print(f"Chord predictions saved to {output_path}")


def load_lab_file(lab_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load chord annotations from a .lab file.
    
    Args:
        lab_path: Path to .lab file
    
    Returns:
        (intervals, labels) where:
            - intervals: (n, 2) array of [start_time, end_time]
            - labels: (n,) array of chord label strings
    """
    intervals = []
    labels = []
    
    with open(lab_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t') if '\t' in line else line.split()
            if len(parts) >= 3:
                start = float(parts[0])
                end = float(parts[1])
                label = parts[2]
                intervals.append([start, end])
                labels.append(label)
    
    return np.array(intervals), np.array(labels)


class BatchChordEvaluator:
    """
    Evaluate predictions on multiple tracks and compute aggregate statistics.
    """
    
    def __init__(
        self,
        predictor: ChordPredictor,
        dataset: Union[BeatlesChordDataset, BeatlesMajMinChordDataset],
        fps: int = 100,
    ):
        """
        Args:
            predictor: ChordPredictor instance
            dataset: Dataset instance
            fps: Frames per second
        """
        self.predictor = predictor
        self.dataset = dataset
        self.evaluator = ChordEvaluator(fps=fps)
        self.fps = fps
    
    def evaluate_track(
        self,
        track_id: str,
        metrics: Optional[List[str]] = None,
        save_lab: Optional[str] = None,
    ) -> Dict[str, Union[float, Dict]]:
        """
        Evaluate a single track.
        
        Args:
            track_id: Track ID to evaluate
            metrics: List of metrics to compute
            save_lab: Optional path to save predicted chords as .lab file
        
        Returns:
            Dictionary with evaluation results
        """
        # Get reference annotations
        track = self.dataset.dataset.track(track_id)
        ref_intervals = track.chords.intervals
        ref_labels = np.array(track.chords.labels)
        
        # Get predictions
        pred_labels, confidences, duration = self.predictor.predict_from_track(
            self.dataset, track_id
        )
        
        # Convert frame-level predictions to intervals
        est_intervals, est_labels = self.evaluator.frames_to_intervals(pred_labels, self.fps)
        
        # Save to .lab file if requested
        if save_lab is not None:
            self.evaluator.write_lab_file(est_intervals, est_labels, save_lab)
        
        # Evaluate
        eval_metrics = self.evaluator.evaluate_from_intervals(
            ref_intervals, ref_labels, est_intervals, est_labels, metrics
        )
        
        return {
            "track_id": track_id,
            "metrics": eval_metrics,
            "num_frames": len(pred_labels),
            "duration": duration,
            "mean_confidence": float(np.mean(confidences)),
        }
    
    def evaluate_multiple_tracks(
        self,
        track_ids: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        save_lab_dir: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate multiple tracks and compute aggregate statistics.
        
        Args:
            track_ids: List of track IDs to evaluate. If None, evaluate all.
            metrics: List of metrics to compute
            save_lab_dir: Optional directory to save predicted chords as .lab files (one per track)
        
        Returns:
            Dictionary with per-track results and aggregate statistics
        """
        if track_ids is None:
            track_ids = self.dataset.track_ids
        
        results = []
        all_metrics = {}
        
        print(f"Evaluating {len(track_ids)} tracks...")
        
        for i, track_id in enumerate(track_ids):
            print(f"  [{i+1}/{len(track_ids)}] Evaluating {track_id}...", end=" ", flush=True)
            try:
                # Generate .lab path if requested
                save_lab = None
                if save_lab_dir is not None:
                    save_lab = f"{save_lab_dir}/{track_id}.lab"
                
                track_result = self.evaluate_track(track_id, metrics, save_lab=save_lab)
                results.append(track_result)
                
                # Accumulate metrics
                for metric_name, score in track_result["metrics"].items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    if score is not None:
                        all_metrics[metric_name].append(score)
                
                print("✓")
            except Exception as e:
                print(f"✗ (Error: {e})")
        
        # Compute aggregate statistics
        aggregate_stats = {}
        for metric_name, scores in all_metrics.items():
            if scores:
                aggregate_stats[metric_name] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                }
        
        return {
            "track_results": results,
            "aggregate_stats": aggregate_stats,
            "num_tracks_evaluated": len(results),
        }


class CustomDatasetEvaluator:
    """
    Evaluate predictions on custom datasets with audio files and .lab annotations.
    
    Use this when your data doesn't follow the mirdata structure.
    """
    
    def __init__(
        self,
        predictor: ChordPredictor,
        fps: int = 100,
    ):
        """
        Args:
            predictor: ChordPredictor instance
            fps: Frames per second
        """
        self.predictor = predictor
        self.evaluator = ChordEvaluator(fps=fps)
        self.fps = fps
    
    def evaluate_audio_file(
        self,
        audio_path: str,
        lab_path: str,
        metrics: Optional[List[str]] = None,
        save_lab: Optional[str] = None,
        target_sr: int = 44100,
    ) -> Dict[str, Union[float, Dict]]:
        """
        Evaluate a single audio file against its .lab annotation.
        
        Args:
            audio_path: Path to audio file
            lab_path: Path to reference .lab annotation file
            metrics: List of metrics to compute
            save_lab: Optional path to save predicted chords as .lab file
            target_sr: Sample rate for loading audio
        
        Returns:
            Dictionary with evaluation results
        """
        # Load reference annotations
        ref_intervals, ref_labels = load_lab_file(lab_path)
        
        # Get predictions
        pred_labels, confidences, duration = self.predictor.predict_from_audio_file(
            audio_path, target_sr=target_sr
        )
        
        # Convert frame-level predictions to intervals
        est_intervals, est_labels = self.evaluator.frames_to_intervals(pred_labels, self.fps)
        
        # Save to .lab file if requested
        if save_lab is not None:
            self.evaluator.write_lab_file(est_intervals, est_labels, save_lab)
        
        # Round intervals to 5 decimal places
        est_intervals = np.round(est_intervals, 5)
        ref_intervals = np.round(ref_intervals, 5)
        
        # Evaluate
        eval_metrics = self.evaluator.evaluate_from_intervals(
            ref_intervals, ref_labels, est_intervals, est_labels, metrics
        )
        
        return {
            "audio_path": audio_path,
            "metrics": eval_metrics,
            "num_frames": len(pred_labels),
            "duration": duration,
            "mean_confidence": float(np.mean(confidences)),
        }
    
    def evaluate_directory(
        self,
        audio_dir: str,
        lab_dir: str,
        audio_ext: str = ".wav",
        metrics: Optional[List[str]] = None,
        save_lab_dir: Optional[str] = None,
        target_sr: int = 44100,
    ) -> Dict:
        """
        Evaluate all audio files in a directory against corresponding .lab files.
        
        Args:
            audio_dir: Directory containing audio files
            lab_dir: Directory containing .lab annotation files
            audio_ext: Audio file extension (e.g., ".wav", ".mp3")
            metrics: List of metrics to compute
            save_lab_dir: Optional directory to save predicted .lab files
            target_sr: Sample rate for loading audio
        
        Returns:
            Dictionary with per-file results and aggregate statistics
        """
        audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith(audio_ext)])
        
        results = []
        all_metrics = {}
        
        print(f"Evaluating {len(audio_files)} audio files...")
        
        for i, audio_file in enumerate(audio_files):
            audio_path = os.path.join(audio_dir, audio_file)
            base_name = os.path.splitext(audio_file)[0]
            lab_path = os.path.join(lab_dir, f"{base_name}.lab")
            
            if not os.path.exists(lab_path):
                print(f"  [{i+1}/{len(audio_files)}] {audio_file}... ✗ (No .lab file found)")
                continue
            
            print(f"  [{i+1}/{len(audio_files)}] {audio_file}...", end=" ", flush=True)
            
            try:
                # Generate .lab path if requested
                save_lab = None
                if save_lab_dir is not None:
                    save_lab = os.path.join(save_lab_dir, f"{base_name}.lab")
                
                file_result = self.evaluate_audio_file(
                    audio_path, lab_path, metrics, save_lab=save_lab, target_sr=target_sr
                )
                results.append(file_result)
                
                # Accumulate metrics
                for metric_name, score in file_result["metrics"].items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    if score is not None:
                        all_metrics[metric_name].append(score)
                
                print("✓")
            except Exception as e:
                print(f"✗ (Error: {e})")
        
        # Compute aggregate statistics
        aggregate_stats = {}
        for metric_name, scores in all_metrics.items():
            if scores:
                aggregate_stats[metric_name] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                }
        
        return {
            "file_results": results,
            "aggregate_stats": aggregate_stats,
            "num_files_evaluated": len(results),
        }
    
    def evaluate_file_pairs(
        self,
        file_pairs: List[Tuple[str, str]],
        metrics: Optional[List[str]] = None,
        save_lab_dir: Optional[str] = None,
        target_sr: int = 44100,
    ) -> Dict:
        """
        Evaluate specific audio-lab file pairs.
        
        Args:
            file_pairs: List of (audio_path, lab_path) tuples
            metrics: List of metrics to compute
            save_lab_dir: Optional directory to save predicted .lab files
            target_sr: Sample rate for loading audio
        
        Returns:
            Dictionary with per-file results and aggregate statistics
        """
        results = []
        all_metrics = {}
        
        print(f"Evaluating {len(file_pairs)} file pairs...")
        
        for i, (audio_path, lab_path) in enumerate(file_pairs):
            audio_name = os.path.basename(audio_path)
            print(f"  [{i+1}/{len(file_pairs)}] {audio_name}...", end=" ", flush=True)
            
            try:
                # Generate .lab path if requested
                save_lab = None
                if save_lab_dir is not None:
                    base_name = os.path.splitext(audio_name)[0]
                    save_lab = os.path.join(save_lab_dir, f"{base_name}.lab")
                
                file_result = self.evaluate_audio_file(
                    audio_path, lab_path, metrics, save_lab=save_lab, target_sr=target_sr
                )
                results.append(file_result)
                
                # Accumulate metrics
                for metric_name, score in file_result["metrics"].items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    if score is not None:
                        all_metrics[metric_name].append(score)
                
                print("✓")
            except Exception as e:
                print(f"✗ (Error: {e})")
        
        # Compute aggregate statistics
        aggregate_stats = {}
        for metric_name, scores in all_metrics.items():
            if scores:
                aggregate_stats[metric_name] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                }
        
        return {
            "file_results": results,
            "aggregate_stats": aggregate_stats,
            "num_files_evaluated": len(results),
        }
    
    def evaluate_nested_directories(
        self,
        root_dir: str,
        audio_filename: Optional[str] = None,
        lab_filename: Optional[str] = None,
        audio_ext: str = ".wav",
        lab_ext: str = ".lab",
        metrics: Optional[List[str]] = None,
        save_lab_dir: Optional[str] = None,
        target_sr: int = 44100,
    ) -> Dict:
        """
        Evaluate audio files in nested subdirectories (one audio + one lab per subdirectory).
        
        Directory structure expected:
            root_dir/
                subdir1/
                    audio.wav
                    annotation.lab
                subdir2/
                    audio.wav
                    annotation.lab
                ...
        
        Args:
            root_dir: Root directory containing subdirectories
            audio_filename: Specific audio filename to look for (e.g., "audio.wav"). 
                           If None, finds first file matching audio_ext
            lab_filename: Specific lab filename to look for (e.g., "annotation.lab").
                         If None, finds first file matching lab_ext
            audio_ext: Audio file extension if audio_filename not specified
            lab_ext: Lab file extension if lab_filename not specified
            metrics: List of metrics to compute
            save_lab_dir: Optional directory to save predicted .lab files (named by subdirectory)
            target_sr: Sample rate for loading audio
        
        Returns:
            Dictionary with per-file results and aggregate statistics
        """
        subdirs = sorted([d for d in os.listdir(root_dir) 
                         if os.path.isdir(os.path.join(root_dir, d))])
        
        results = []
        all_metrics = {}
        
        print(f"Evaluating {len(subdirs)} subdirectories in {root_dir}...")
        
        for i, subdir in enumerate(subdirs):
            subdir_path = os.path.join(root_dir, subdir)
            files_in_subdir = os.listdir(subdir_path)
            
            # Find audio file
            if audio_filename:
                audio_file = audio_filename if audio_filename in files_in_subdir else None
            else:
                audio_candidates = [f for f in files_in_subdir if f.endswith(audio_ext)]
                audio_file = audio_candidates[0] if audio_candidates else None
            
            # Find lab file
            if lab_filename:
                lab_file = lab_filename if lab_filename in files_in_subdir else None
            else:
                lab_candidates = [f for f in files_in_subdir if f.endswith(lab_ext)]
                lab_file = lab_candidates[0] if lab_candidates else None
            
            if not audio_file:
                print(f"  [{i+1}/{len(subdirs)}] {subdir}... ✗ (No audio file found)")
                continue
            
            if not lab_file:
                print(f"  [{i+1}/{len(subdirs)}] {subdir}... ✗ (No .lab file found)")
                continue
            
            audio_path = os.path.join(subdir_path, audio_file)
            lab_path = os.path.join(subdir_path, lab_file)
            
            print(f"  [{i+1}/{len(subdirs)}] {subdir}...", end=" ", flush=True)
            
            try:
                # Generate .lab path if requested
                save_lab = None
                if save_lab_dir is not None:
                    os.makedirs(save_lab_dir, exist_ok=True)
                    save_lab = os.path.join(save_lab_dir, f"{subdir}.lab")
                
                file_result = self.evaluate_audio_file(
                    audio_path, lab_path, metrics, save_lab=save_lab, target_sr=target_sr
                )
                file_result["subdirectory"] = subdir  # Add subdir info to result
                results.append(file_result)
                
                # Accumulate metrics
                for metric_name, score in file_result["metrics"].items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    if score is not None:
                        all_metrics[metric_name].append(score)
                
                print("✓")
            except Exception as e:
                print(f"✗ (Error: {e})")
        
        # Compute aggregate statistics
        aggregate_stats = {}
        for metric_name, scores in all_metrics.items():
            if scores:
                aggregate_stats[metric_name] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                }
        
        return {
            "file_results": results,
            "aggregate_stats": aggregate_stats,
            "num_files_evaluated": len(results),
        }


# Convenience functions
def load_predictor_from_checkpoint(
    checkpoint_path: str,
    dataset: Union[BeatlesChordDataset, BeatlesMajMinChordDataset, UnifiedConcatDataset],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> ChordPredictor:
    """
    Load a ChordPredictor from a checkpoint and dataset.
    
    Args:
        checkpoint_path: Path to PyTorch Lightning checkpoint
        dataset: Dataset instance (to get label mappings)
        device: Device to load model on
    
    Returns:
        Initialized ChordPredictor
    """
    return ChordPredictor(
        model=checkpoint_path,
        label_to_idx=dataset.label_to_idx,
        idx_to_label=dataset.idx_to_label,
        device=device,
    )


if __name__ == "__main__":
    # Example usage for prediction and evaluation
    extended_ds = True
    # 1. Load dataset and predictor
    print("Loading dataset and model...")
    
    if extended_ds:
        ds, label_to_idx, idx_to_label = build_combined_dataset(
            beatles_root='./mir_datasets2/beatles',
            external_root='./dataset_ext',
            fps=100,
            sample_rate=None  # Will use Beatles sample rate
        )
        predictor = load_predictor_from_checkpoint("model_ep-100_ext.ckpt", ds)

        if not os.path.exists("transition_matrix_ext.npy"):
            transition_matrix = get_transition_matrix(ds, ds.n_classes)
            np.save("transition_matrix_ext.npy", transition_matrix)
        else:
            transition_matrix = np.load("transition_matrix_ext.npy")
    else:
        ds = BeatlesChordDataset("./mir_datasets2/beatles")
        predictor = load_predictor_from_checkpoint("best_model.ckpt", ds)
        transition_matrix = np.load("trans_matrix.npy") # Load it next time
    
    # Validate transition matrix shape matches model classes
        
    transition_matrix = make_transition_matrix_sticky(transition_matrix, self_prob=0.99)
    predictor.transition_matrix = transition_matrix
    
    print(f"Loaded model with {ds.n_classes} chord classes\n")
    
    # # 2. Single track prediction
    # print("=" * 60)
    # print("SINGLE TRACK PREDICTION")
    # print("=" * 60)
    # track_id = ds.track_ids[0]
    # print(f"Predicting on track: {track_id}")
    # pred_labels, confidences, duration = predictor.predict_from_track(ds, track_id)
    # print(f"  Duration: {duration:.2f}s")
    # print(f"  Predicted frames: {len(pred_labels)}")
    # print(f"  Mean confidence: {np.mean(confidences):.3f}\n")
    
    # # 3. Single track evaluation with .lab file
    # print("=" * 60)
    # print("SINGLE TRACK EVALUATION")
    # print("=" * 60)
    # evaluator = ChordEvaluator(fps=100)
    
    # track_result = BatchChordEvaluator(predictor, ds).evaluate_track(
    #     track_id, 
    #     metrics=["mirex", "root", "majmin", "majmin_inv", "sevenths", "sevenths_inv"],
    #     save_lab=f"predictions/{track_id}.lab"
    # )
    # print(f"Track: {track_result['track_id']}")
    # print(f"Metrics:")
    # for metric, score in track_result["metrics"].items():
    #     if score is not None:
    #         print(f"  {metric}: {score:.3f}")
    # print(f"Mean confidence: {track_result['mean_confidence']:.3f}\n")
    
    # # 4. Batch evaluation on multiple tracks with .lab export
    # print("=" * 60)
    # print("BATCH EVALUATION")
    # print("=" * 60)
    # batch_eval = BatchChordEvaluator(predictor, ds)
    
    # # Evaluate subset of tracks (use all if you want: track_ids=None)
    # track_subset = ds.track_ids[:5]  # Evaluate first 5 tracks
    # results = batch_eval.evaluate_multiple_tracks(
    #     track_ids=track_subset,
    #     metrics=["mirex", "root", "majmin"],
    #     save_lab_dir="predictions/lab_files"  # Save all predictions as .lab files
    # )
    
    # # 5. Print aggregate statistics
    # print("\n" + "=" * 60)
    # print("AGGREGATE STATISTICS")
    # print("=" * 60)
    # print(f"Evaluated {results['num_tracks_evaluated']} tracks\n")
    # for metric, stats in results["aggregate_stats"].items():
    #     print(f"{metric.upper()}:")
    #     print(f"  Mean:  {stats['mean']:.3f}")
    #     print(f"  Std:   {stats['std']:.3f}")
    #     print(f"  Min:   {stats['min']:.3f}")
    #     print(f"  Max:   {stats['max']:.3f}\n")
    
    # # 6. Access individual track results
    # print("=" * 60)
    # print("INDIVIDUAL TRACK RESULTS")
    # print("=" * 60)
    # for track_result in results["track_results"]:
    #     print(f"{track_result['track_id']:40s} | MIREX: {track_result['metrics']['mirex']:.3f} | Root: {track_result['metrics']['root']:.3f}")
    
    # 7. Evaluate custom dataset with nested subdirectories
    print("\n" + "=" * 60)
    print("CUSTOM DATASET - NESTED SUBDIRECTORIES")
    print("=" * 60)
    
    custom_eval = CustomDatasetEvaluator(predictor, fps=100)
    
    # Option 1: Auto-detect files by extension
    custom_results = custom_eval.evaluate_nested_directories(
        root_dir="test_ds",
        audio_ext=".wav",      # Finds first .wav in each subdirectory
        lab_ext=".lab",        # Finds first .lab in each subdirectory
        metrics=["mirex", "root", "majmin", "majmin_inv", "sevenths", "sevenths_inv", "overseg", "underseg", "seg"],
        save_lab_dir="predictions/extended_dataset",
        target_sr=44100
    )
    
    # Option 2: Specify exact filenames (uncomment to use)
    # custom_results = custom_eval.evaluate_nested_directories(
    #     root_dir="dataset_eval",
    #     audio_filename="audio.wav",        # Exact audio filename
    #     lab_filename="annotation.lab",     # Exact lab filename
    #     metrics=["mirex", "root", "majmin", "majmin_inv", "sevenths", "sevenths_inv", "overseg", "underseg", "seg"],
    #     save_lab_dir="predictions/custom_dataset"
    # )
    
    # Print custom dataset results
    print(f"\nEvaluated {custom_results['num_files_evaluated']} files\n")
    print("Aggregate Statistics:")
    for metric, stats in custom_results["aggregate_stats"].items():
        print(f"{metric.upper()}:")
        print(f"  Mean:  {stats['mean']:.4f}")
        print(f"  Std:   {stats['std']:.4f}")
        print(f"  Min:   {stats['min']:.4f}")
        print(f"  Max:   {stats['max']:.4f}\n")
    
    # Show per-subdirectory results with key metrics
    print("\nPer-subdirectory results:")
    print(f"{'Subdirectory':40s} | {'MIREX':>7s} | {'Root':>7s} | {'MajMin':>7s} | {'OverSeg':>7s} | {'UnderSeg':>7s} | {'Seg':>7s}")
    print("-" * 110)
    for file_result in custom_results["file_results"]:
        subdir = file_result.get('subdirectory', 'unknown')
        mirex = file_result['metrics'].get('mirex', None)
        root = file_result['metrics'].get('root', None)
        majmin = file_result['metrics'].get('majmin', None)
        overseg = file_result['metrics'].get('overseg', None)
        underseg = file_result['metrics'].get('underseg', None)
        seg = file_result['metrics'].get('seg', None)
        
        mirex_str = f"{mirex:.4f}" if mirex is not None else "N/A"
        root_str = f"{root:.4f}" if root is not None else "N/A"
        majmin_str = f"{majmin:.4f}" if majmin is not None else "N/A"
        overseg_str = f"{overseg:.4f}" if overseg is not None else "N/A"
        underseg_str = f"{underseg:.4f}" if underseg is not None else "N/A"
        seg_str = f"{seg:.4f}" if seg is not None else "N/A"
        
        print(f"{subdir:40s} | {mirex_str:>7s} | {root_str:>7s} | {majmin_str:>7s} | {overseg_str:>7s} | {underseg_str:>7s} | {seg_str:>7s}")
    
    # Save results to JSON file
    os.makedirs("predictions", exist_ok=True)
    json_output_path = "predictions/extended_dataset_results.json"
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_serializable(custom_results)
    
    with open(json_output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nEvaluation results saved to {json_output_path}")
    
    # # 8. Evaluate single audio file (not in mirdata structure)
    # print("\n" + "=" * 60)
    # print("SINGLE AUDIO FILE EVALUATION")
    # print("=" * 60)
    # single_result = custom_eval.evaluate_audio_file(
    #     audio_path="dataset_eval/1058_Ain't_No_Sunshine/Ain't_No_Sunshine.wav",
    #     lab_path="dataset_eval/1058_Ain't_No_Sunshine/full.lab",
    #     metrics=["mirex", "root", "majmin", "majmin_inv", "sevenths", "sevenths_inv", "overseg", "underseg", "seg"],
    #     save_lab="predictions/my_song_predicted.lab",
    #     target_sr=44100
    # )
    # print(f"File: {single_result['audio_path']}")
    # print(f"Duration: {single_result['duration']:.2f}s")
    # print("Metrics:")
    # for metric, score in single_result["metrics"].items():
    #     if score is not None:
    #         print(f"  {metric}: {score:.3f}")