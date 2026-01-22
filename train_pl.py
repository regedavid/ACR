import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import mir_eval.chord as mrc   # add this import
import os
import pickle

from cnn_acr import CNNTransformerChordModel
from dataset import BeatlesChordDataset, BeatlesMajMinChordDataset, SegmentWrapper


class LightningChordModel(pl.LightningModule):
    def __init__(
        self,
        n_classes,
        lr=3e-4,
        sample_rate=44100,
        fps=100,
        segment_seconds=8.0,
        ignore_index=-100,   # add ignore_index hyperparam
        **model_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Build underlying model from your earlier code
        self.model = CNNTransformerChordModel(
            n_classes=n_classes,
            sample_rate=sample_rate,
            fps=fps,
            **model_kwargs
        )

        self.lr = lr

    # ----- Forward -----
    def forward(self, audio_batch):
        audio_batch = audio_batch.to(self.device)
        return self.model(audio_batch)

    # ----- Training Step -----
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

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ----- Validation Step -----
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

        # Accuracy: framewise argmax (compute only on non-ignored frames)
        preds = logits.argmax(dim=-1)
        mask = labels != ignore_idx
        if mask.any():
            acc = (preds[mask] == labels[mask]).float().mean()
        else:
            acc = torch.tensor(0.0, device=self.device)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    # ----- Optimizer -----
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

        # build per-segment weights: downweight full-NO_CHORD segments
        no_idx = self.base_dataset.label_to_idx[mrc.NO_CHORD]

        # try to load cached weights
        cache_path = os.path.join(self.root, ".segment_weights.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                sample_weights = pickle.load(f)
        else:
            # Use a DataLoader with multiple workers to compute labels in parallel.
            # This is much faster than calling full[i] repeatedly in Python.
            num_workers = min(8, (os.cpu_count() or 1))
            dl = DataLoader(full, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=False)
            sample_weights = []
            for audio, labels in dl:
                # labels may be tensor or ndarray; handle both
                lab = labels[0] if isinstance(labels, torch.Tensor) else labels[0]
                lab_arr = lab.detach().cpu().numpy() if isinstance(lab, torch.Tensor) else lab
                sample_weights.append(0.1 if (lab_arr == no_idx).all() else 1.0)
            # cache to disk for faster next runs
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(sample_weights, f)
            except Exception:
                pass

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
            shuffle=False,               # sampler + shuffle=False
            sampler=getattr(self, "train_sampler", None),
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )



if __name__ == "__main__":
    ROOT = "./mir_datasets2/beatles"

    # First get the dataset so we know vocabulary size
    ds = BeatlesMajMinChordDataset(ROOT, fps=100)
    n_classes = len(ds.label_to_idx)

    # pass dataset NO_CHORD index as ignore_index
    no_idx = ds.label_to_idx[mrc.NO_CHORD]

    model = LightningChordModel(
        n_classes=n_classes,
        lr=3e-4,
        d_model=256,
        nhead=8,
        num_layers=4,
        segment_seconds=8.0,
        ignore_index=no_idx,   # <- pass here
    )

    dm = ChordDataModule(
        root=ROOT,
        base_dataset=ds,
        batch_size=4,
        fps=100,
        segment_seconds=8.0,
    )

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="gpu",
        devices=1,
        gradient_clip_val=1.0,
        precision=32,
        log_every_n_steps=20,
    )

    trainer.fit(model, dm)
