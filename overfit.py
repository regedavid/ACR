import pytorch_lightning as pl
import mir_eval.chord as mrc

from train_pl import LightningChordModel, ChordDataModule
from dataset import BeatlesMajMinChordDataset

ROOT = "./mir_datasets2/beatles"

# Build dataset + compute vocab / NO_CHORD index
ds = BeatlesMajMinChordDataset(ROOT, fps=100)
n_classes = len(ds.label_to_idx)
no_idx = ds.label_to_idx[mrc.NO_CHORD]

# Create model and datamodule: batch_size=1 and val_split=0 so validation doesn't interfere
model = LightningChordModel(
    n_classes=n_classes,
    lr=3e-4,
    d_model=256,
    nhead=8,
    num_layers=4,
    segment_seconds=8.0,
    ignore_index=no_idx,
)

dm = ChordDataModule(
    root=ROOT,
    base_dataset=ds,
    batch_size=1,
    fps=100,
    segment_seconds=8.0,
    val_split=0.0,   # no val set for pure overfit
)

# Use Lightning's overfit_batches to train on a single batch repeatedly.
# If no GPU is available, change accelerator="cpu".
trainer = pl.Trainer(
    max_epochs=200,
    overfit_batches=1,    # train repeatedly on a single batch
    accelerator="gpu",
    devices=1,
    gradient_clip_val=1.0,
    precision=32,
    log_every_n_steps=1,
)

trainer.fit(model, dm)