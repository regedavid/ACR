import numpy as np
from dataset import BeatlesChordDataset

dataset = BeatlesChordDataset(root="./mir_datasets/beatles", vocab="majmin7", fps=100)