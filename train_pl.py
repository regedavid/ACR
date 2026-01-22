import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint # <--- 1. Import Callback
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import mir_eval.chord as mrc
import os
import pickle
import argparse

# ---------------------------
# Import your model and dataset
# ---------------------------
from cnn_acr import CNNTransformerChordModel
from dataset import BeatlesChordDataset, BeatlesMajMinChordDataset, SegmentWrapper

# ===========================================================
# 1) LightningModule for the CNN+Transformer chord estimator
# ===========================================================
class LightningChordModel(pl.LightningModule):
    def __init__(
        self,
        n_classes,
        lr=3e-4,
        sample_rate=44100,
        fps=100,
        segment_seconds=8.0,
        ignore_index=-100,
        **model_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Build underlying model
        self.model = CNNTransformerChordModel(
            n_classes=n_classes,
            sample_rate=sample_rate,
            fps=fps,
            **model_kwargs
        )

        self.lr = lr

    def forward(self, audio_batch):
        return self.model(audio_batch)

    def training_step(self, batch, batch_idx):
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
        audio, labels = batch
        audio = audio.to(self.device)
        labels = labels.to(self.device).long()
        logits = self(audio)
        B, T, C = logits.shape

        ignore_idx = getattr(self.hparams, "ignore_index", -100)

        loss = F.cross_entropy(
            logits.reshape(B * T, C),
            labels.reshape(B * T),
            ignore_index=ignore_idx,
        )

        # Accuracy: framewise argmax
        preds = logits.argmax(dim=-1)
        mask = labels != ignore_idx
        
        if mask.any():
            acc = (preds[mask] == labels[mask]).float().mean()
        else:
            acc = torch.tensor(0.0, device=self.device)

        self.log("val_loss", loss, on_epoch=True, prog_bar=False)
        self.log("val_acc", acc, on_epoch=True, prog_bar=False)
        return {"val_loss": loss, "val_acc": acc}

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
        no_idx = self.base_dataset.label_to_idx[mrc.NO_CHORD]
        cache_path = os.path.join(self.root, ".segment_weights.pkl")
        
        if os.path.exists(cache_path):
            print(f"Loading cached segment weights from {cache_path}")
            with open(cache_path, "rb") as f:
                sample_weights = pickle.load(f)
        else:
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
    # Handle Command Line Arguments
    parser = argparse.ArgumentParser(description="Train Chord Model Locally")
    parser.add_argument("--data_dir", type=str, default="./mir_datasets2/beatles", 
                        help="Path to the Beatles dataset root folder")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    # Determine optimal workers
    if "SLURM_CPUS_PER_TASK" in os.environ:
        workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    else:
        workers = 4

    print(f"Using {workers} dataloader workers.")

    print(f"Data Root: {args.data_dir}")
    print(f"Workers: {workers}")

    ds = BeatlesMajMinChordDataset(args.data_dir, fps=100)
    n_classes = len(ds.label_to_idx)
    no_idx = ds.label_to_idx[mrc.NO_CHORD]

    # 2. Initialize Model
    model = LightningChordModel(
        n_classes=n_classes,
        lr=3e-4,
        d_model=256,
        nhead=8,
        num_layers=4,
        segment_seconds=8.0,
        ignore_index=no_idx,
    )

    # 3. Initialize DataModule
    dm = ChordDataModule(
        root=args.data_dir,
        base_dataset=ds,
        batch_size=args.batch_size,
        fps=100,
        segment_seconds=8.0,
        num_workers=workers
    )

    # 4. Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        dirpath="checkpoints/",
        filename="chord-model-{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        verbose=True,
    )

    # 5. Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",  # Will use GPU if available, else CPU
        devices=1,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
    )

    # 6. Train
    trainer.fit(model, dm)

    # 7. Final Save
    trainer.save_checkpoint("final_chord_model.ckpt")
    print("Training complete. Final model saved to 'final_chord_model.ckpt'")