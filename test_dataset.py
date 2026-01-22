import numpy as np
from dataset import BeatlesChordDataset
from torch.utils.data import DataLoader
from dataset import SegmentWrapper

dataset = BeatlesChordDataset(root="./mir_datasets2/beatles", fps=100)
sample = dataset[0]
dataset = SegmentWrapper(dataset, segment_seconds=8.0, hop_seconds=4.0)
sample = dataset[0]
for i in range(len(dataset)):
    sample = dataset[i]
    if isinstance(sample, dict):
        summary = {k: getattr(v, "shape", (len(v) if hasattr(v, "__len__") else type(v))) for k, v in sample.items()}
    elif isinstance(sample, (list, tuple)):
        summary = [getattr(v, "shape", (len(v) if hasattr(v, "__len__") else type(v))) for v in sample]
    else:
        summary = getattr(sample, "shape", (len(sample) if hasattr(sample, "__len__") else type(sample)))
    print(f"Index {i}: {summary}")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
for batch in dataloader:
    print(batch)
    break
# to test __getitem__

print(f"Dataset length: {len(dataset)}")