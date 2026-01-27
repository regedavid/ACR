import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint # <--- 1. Import Callback
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import mir_eval.chord as mrc
import os
import pickle
import argparse

# ---------------------------
# Import your model and dataset
# ---------------------------
from cnn_acr import CNNTransformerChordModel, FrontendType
from dataset import BeatlesChordDataset, BeatlesMajMinChordDataset, SegmentWrapper
from custom_dataset import build_combined_dataset

# ===========================================================
# 1) LightningModule for the CNN+Transformer chord estimator
# ===========================================================
class LightningChordModel(pl.LightningModule):
    def __init__(
        self,
        n_classes=None,
        lr=3e-4,
        sample_rate=44100,
        fps=100,
        segment_seconds=8.0,
        ignore_index=-100,
        multi_target=False,
        n_roots=None, n_qualities=None, n_basses=None,
        ignore_root=-100, ignore_qual=-100, ignore_bass=-100,
        **model_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.multi_target = multi_target

        # Build underlying model
        if self.multi_target:
            from cnn_acr import MultiHeadChordModel
            self.model = MultiHeadChordModel(
                n_roots=n_roots, n_qualities=n_qualities, n_bass=n_basses,
                **model_kwargs
            )
        else:
            self.model = CNNTransformerChordModel(
                n_classes=n_classes,
                **model_kwargs
            )

        self.lr = lr

    def forward(self, audio_batch):
        return self.model(audio_batch)

    def training_step(self, batch, batch_idx):
        if self.multi_target:
            audio, (r_true, q_true, b_true) = batch
            audio = audio.to(self.device)
            r_true, q_true, b_true = r_true.long(), q_true.long(), b_true.long()
            
            # Forward
            r_logits, q_logits, b_logits = self.model(audio) # (B, T, C)
            
            # Flatten
            def flatten_loss(logits, targets):
                return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

            loss = flatten_loss(r_logits, r_true) + \
                   flatten_loss(q_logits, q_true) + \
                   0.5 * flatten_loss(b_logits, b_true) # Weight bass less?
            
            self.log("train_loss", loss, on_step=True, prog_bar=True)
            return loss
        else:
            audio, labels = batch
            audio = audio.to(self.device)
            labels = labels.to(self.device).long()
            logits = self(audio)              # (B, T, n_classes)
            B, T, C = logits.shape

            ignore_idx = getattr(self.hparams, "ignore_index", -100)

            loss = F.cross_entropy(
                logits.reshape(B * T, C),
                labels.reshape(B * T),
                ignore_index=ignore_idx,
            )
            
            # Log training loss
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss

    def validation_step(self, batch, batch_idx):
        if self.multi_target:
            # --- MULTI-TARGET VALIDATION ---
            audio, (r_true, q_true, b_true) = batch
            r_logits, q_logits, b_logits = self(audio)

            # 1. Compute Losses
            loss_r = F.cross_entropy(r_logits.transpose(1, 2), r_true, ignore_index=self.hparams.ignore_root)
            loss_q = F.cross_entropy(q_logits.transpose(1, 2), q_true, ignore_index=self.hparams.ignore_qual)
            loss_b = F.cross_entropy(b_logits.transpose(1, 2), b_true, ignore_index=self.hparams.ignore_bass)
            total_loss = loss_r + loss_q + 0.5 * loss_b

            # 2. Compute Accuracies
            def get_acc(logits, target, ignore_idx):
                preds = logits.argmax(dim=-1)
                mask = target != ignore_idx
                if mask.sum() > 0:
                    return (preds[mask] == target[mask]).float().mean()
                return torch.tensor(0.0, device=self.device)

            acc_r = get_acc(r_logits, r_true, self.hparams.ignore_root)
            acc_q = get_acc(q_logits, q_true, self.hparams.ignore_qual)
            acc_b = get_acc(b_logits, b_true, self.hparams.ignore_bass)

            # 3. Log Everything
            self.log("val_loss", total_loss, on_epoch=True, prog_bar=True)
            self.log("val_acc_root", acc_r, on_epoch=True)
            self.log("val_acc_qual", acc_q, on_epoch=True)
            self.log("val_acc_bass", acc_b, on_epoch=True)

        else:
            # --- SINGLE-TARGET VALIDATION ---
            audio, labels = batch
            logits = self(audio)
            
            loss = F.cross_entropy(
                logits.transpose(1, 2), 
                labels, 
                ignore_index=self.hparams.ignore_index
            )
            
            preds = logits.argmax(dim=-1)
            mask = labels != self.hparams.ignore_index
            if mask.sum() > 0:
                acc = (preds[mask] == labels[mask]).float().mean()
            else:
                acc = torch.tensor(0.0, device=self.device)

            self.log("val_loss", loss, on_epoch=True, prog_bar=True)
            self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if "val_loss" in metrics and "val_acc" in metrics:
            print(f"\nEpoch {self.current_epoch}: Val Loss: {metrics['val_loss']:.4f} | Val Acc: {metrics['val_acc']:.4f}")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# ===========================================================
# 2) LightningDataModule for loading Beatles segments
# ===========================================================
class ChordDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root,
        base_dataset,
        batch_size=4,
        fps=100,
        segment_seconds=8.0,
        val_split=0.1,
        shuffle=True,
        num_workers=4,  # Added this param
        **dataset_kwargs
    ):
        super().__init__()
        self.root = root
        self.base_dataset = base_dataset
        self.batch_size = batch_size
        self.fps = fps
        self.segment_seconds = segment_seconds
        self.val_split = val_split
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.dataset_kwargs = dataset_kwargs
        self.setup_done = False

    def setup(self, stage=None):
        if self.setup_done:
            return

        full = SegmentWrapper(
            self.base_dataset,
            segment_seconds=self.segment_seconds,
            hop_seconds=self.segment_seconds / 2,
        )

        total = len(full)
        val_size = int(total * self.val_split)
        train_size = total - val_size

        self.train_ds, self.val_ds = random_split(full, [train_size, val_size])
        self.setup_done = True

        # Weight calculation and caching
        
        if self.segment_seconds == 8.0:
            cache_path = os.path.join(self.root, ".segment_weights.pkl")
        elif self.segment_seconds == 20.0:
            cache_path = os.path.join(self.root, ".segment_weights_20sec.pkl")
        
        if os.path.exists(cache_path):
            print(f"Loading cached segment weights from {cache_path}")
            with open(cache_path, "rb") as f:
                sample_weights = pickle.load(f)
        else:
            no_idx = self.base_dataset.label_to_idx[mrc.NO_CHORD]
            print("Calculating segment weights... (this may take a minute)")
            # Use 0 workers for calculation to avoid overhead/pickling issues
            dl = DataLoader(full, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
            sample_weights = []

            for audio, labels in dl:
                lab = labels[0] if isinstance(labels, torch.Tensor) else labels[0]
                lab_arr = lab.detach().cpu().numpy() if isinstance(lab, torch.Tensor) else lab
                non_ignored_frames = (lab_arr != no_idx).sum()
                total_frames = len(lab_arr)

                # Calculate proportion of valid frames
                if total_frames > 0:
                    weight = non_ignored_frames / total_frames
                else:
                    weight = 0.0

                # Give a small minimum weight to avoid dropping segments entirely
                # (e.g., all-N segments still get sampled, just very rarely)
                sample_weights.append(max(weight, 0.01))
                print(f"Processed segment {len(sample_weights)}/{total}", end="\r")

            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(sample_weights, f)
                print(f"Segment weights cached to {cache_path}")
            except Exception as e:
                print(f"Warning: Could not cache segment weights. {e}")

        # map weights to train subset indices
        train_indices = getattr(self.train_ds, "indices", list(range(train_size)))
        train_weights = [sample_weights[i] for i in train_indices]
        self.train_sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)

        print(f"Train segments: {len(self.train_ds)}")
        print(f"Val segments:   {len(self.val_ds)}")

    # ---- DataLoaders ----
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=getattr(self, "train_sampler", None),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0), # Faster epoch transition
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )


# ===========================================================
# 3) Train Script (Local Version)
# ===========================================================
if __name__ == "__main__":
    seed = 1234
    pl.seed_everything(seed, workers=True)
    
    # Handle Command Line Arguments
    parser = argparse.ArgumentParser(description="Train Chord Model Locally")
    parser.add_argument("--data_dir", type=str, default="./mir_datasets2/beatles", 
                        help="Path to the Beatles dataset root folder")
    parser.add_argument("--external_root", type=str, default=None, 
                        help="Path to external dataset folder (e.g., dataset_eval). If provided, will combine with Beatles data.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--experiment_name", type=str, default="chord_model", 
                        help="Name of the experiment for organizing logs and checkpoints")
    parser.add_argument("--frontend", type=str, default="mel", choices=["mel", "cqt"],
                        help="Frontend type: 'mel' for Mel spectrogram or 'cqt' for Constant-Q Transform")
    parser.add_argument("--n_mels", type=int, default=128, 
                        help="Number of Mel bands (only used if --frontend=mel)")
    parser.add_argument("--n_fft", type=int, default=2048,
                        help="FFT size (only used if --frontend=mel)")
    parser.add_argument("--n_cqt_bins", type=int, default=84,
                        help="Number of CQT bins (only used if --frontend=cqt)")
    parser.add_argument("--segment_seconds", type=float, default=8.0,
                        help="Segment length in seconds for training")
    parser.add_argument("--multi_layer", action="store_true",
                        help="Enable multi-task learning (Root, Quality, Bass)")
    args = parser.parse_args()

    # Determine optimal workers
    if "SLURM_CPUS_PER_TASK" in os.environ:
        workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    else:
        workers = 4

    print(f"Using {workers} dataloader workers.")

    print(f"Data Root: {args.data_dir}")
    print(f"Workers: {workers}")

    # Use combined dataset if external_root is provided, otherwise just Beatles
    if args.external_root:
        print(f"Using combined dataset: Beatles + External ({args.external_root})")
        if args.multi_layer:
            ds, r2i, q2i, b2i = build_combined_dataset(
                beatles_root=args.data_dir,
                external_root=args.external_root,
                fps=100,
                multi_target=True
            )
            print(f"Vocab: {len(r2i)} Roots, {len(q2i)} Qualities, {len(b2i)} Basses")
        else:
            # Single target returns (ds, l2i, i2l, None)
            ds, l2i, i2l, _ = build_combined_dataset(
                beatles_root=args.data_dir,
                external_root=args.external_root,
                fps=100,
                multi_target=False
            )
            print(f"Vocab: {len(l2i)} Chord Classes")
    else:
        print("Using Beatles dataset only")
        ds = BeatlesChordDataset(args.data_dir, fps=100, multi_target=args.multi_layer)
    
        
    print(f"Using {args.frontend.upper()} frontend")

    # Build frontend-specific kwargs
    frontend_kwargs = {}
    if args.frontend == "mel":
        frontend_kwargs = {"n_mels": args.n_mels, "n_fft": args.n_fft}
    elif args.frontend == "cqt":
        frontend_kwargs = {"n_cqt_bins": args.n_cqt_bins}

    if args.multi_layer:
        # Multi-Target Initialization
        model = LightningChordModel(
            multi_target=True,
            n_roots=ds.n_roots,
            n_qualities=ds.n_qualities,
            n_basses=ds.n_bass,
            ignore_bass=-100,
            ignore_qual=-100,
            ignore_root=-100,
            # Common args
            lr=3e-4,
            frontend_type=args.frontend,
            d_model=256,
            nhead=8,
            num_layers=4,
            segment_seconds=args.segment_seconds,
            **frontend_kwargs,
        )
        monitor_metric = "val_loss" # Accuracy is complex in multi-head, verify loss instead
    else:
        # Single-Target Initialization
        no_idx = ds.label_to_idx[mrc.NO_CHORD]
        model = LightningChordModel(
            multi_target=False,
            n_classes=ds.n_classes,
            ignore_index=-100,
            # Common args
            lr=3e-4,
            frontend_type=args.frontend,
            d_model=256,
            nhead=8,
            num_layers=4,
            segment_seconds=args.segment_seconds,
            **frontend_kwargs,
        )
        monitor_metric = "val_acc"

    # 3. Initialize DataModule
    dm = ChordDataModule(
        root=args.data_dir,
        base_dataset=ds,
        batch_size=args.batch_size,
        fps=100,
        segment_seconds=args.segment_seconds,
        num_workers=workers
    )

    # 4. Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        mode="min" if monitor_metric == "val_loss" else "max",
        dirpath=f"checkpoints/{args.experiment_name}/",
        filename="{epoch:02d}-{" + monitor_metric + ":.2f}",
        save_top_k=1,
        verbose=True,
    )

    # 5. Logger
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=args.experiment_name,
        default_hp_metric=False,
    )

    # 6. Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",  # Will use GPU if available, else CPU
        devices=1,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
        logger=logger,
    )

    # 7. Train
    trainer.fit(model, dm)

    # 8. Final Save
    # final_checkpoint_path = f"checkpoints/{args.experiment_name}/final_model.ckpt"
    # trainer.save_checkpoint(final_checkpoint_path)
    # print(f"Training complete. Final model saved to '{final_checkpoint_path}'")