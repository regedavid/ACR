import math
import torch
import librosa
import torch.nn as nn
import torch.nn.functional as F

# Try to use torchaudio if available (faster / GPU-capable). Fall back to librosa if needed.
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except Exception:
    TORCHAUDIO_AVAILABLE = False
    import librosa
    import numpy as np


class MelFrontend(nn.Module):
    def __init__(
        self,
        sample_rate: int = 44100,
        fps: int = 100,
        n_mels: int = 128,
        n_fft: int = 2048,
        power: float = 2.0,
        f_min: int = 0,
        f_max: int | None = None,
    ):
        super().__init__()
        self.sr = sample_rate
        self.fps = fps
        self.hop_length = int(round(sample_rate / fps))  # e.g., 441
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2
        self.power = power

        if TORCHAUDIO_AVAILABLE:
            self.melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                power=self.power,
                n_mels=self.n_mels,
                f_min=self.f_min,
                f_max=self.f_max,
                norm='slaney',
                mel_scale='htk',
            )
            self.db = torchaudio.transforms.AmplitudeToDB(stype='power')  # convert power -> dB
        else:
            # librosa-based fallback (performed in forward)
            self.melspec = None

    def forward(self, audio):  # audio: (B, seg_samples) or (seg_samples,) torch tensor
        """
        Returns: mel_db tensor shaped (B, n_mels, T)
        """
        single_input = False
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            single_input = True

        if TORCHAUDIO_AVAILABLE:
            # expects (B, samples)
            # torchaudio's MelSpectrogram expects (B, 1, N) for some versions; but current API accepts (B, N)
            mel = self.melspec(audio)  # -> (B, n_mels, T)
            mel = self.db(mel)
            
            total_samples = audio.shape[-1]
            n_frames_expected = int(round(total_samples / self.hop_length))
            self.n_frames = n_frames_expected

            # Truncate or pad mel to match n_frames_expected
            if mel.shape[-1] > n_frames_expected:
                mel = mel[..., :n_frames_expected]
            elif mel.shape[-1] < n_frames_expected:
                pad_amount = n_frames_expected - mel.shape[-1]
                mel = F.pad(mel, (0, pad_amount))  # pad last dimension
            # clamp or normalize as needed
            return mel if not single_input else mel.squeeze(0)
        else:
            # fallback: use librosa per item (slower, CPU)
            mel_list = []
            audio_np = audio.detach().cpu().numpy()
            for i in range(audio_np.shape[0]):
                y = audio_np[i]
                S = librosa.feature.melspectrogram(
                    y=y,
                    sr=self.sr,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.n_fft,
                    power=self.power,
                    n_mels=self.n_mels,
                    fmin=self.f_min,
                    fmax=self.f_max,
                )
                S_db = librosa.power_to_db(S, ref=np.max).astype('float32')
                mel_list.append(S_db)
            mel_np = np.stack(mel_list, axis=0)
            return torch.from_numpy(mel_np)

# ------------------------
# 2) CNN encoder
# ------------------------
class CNNEncoder(nn.Module):
    """
    Small 2D-CNN front end that reduces frequency dimension and preserves time resolution.
    Input: (B, 1, F, T)
    Output: (B, T, d_model)
    """
    def __init__(self, n_mels=128, d_model=256, chans=[64, 128, 256]):
        super().__init__()
        self.n_mels = n_mels
        self.d_model = d_model

        # Conv blocks with frequency pooling only (keep time dimension)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, chans[0], kernel_size=(7,3), padding=(3,1)),
            nn.BatchNorm2d(chans[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1))  # halve frequency, keep time
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(chans[0], chans[1], kernel_size=(5,3), padding=(2,1)),
            nn.BatchNorm2d(chans[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1))  # halve frequency again
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(chans[1], chans[2], kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(chans[2]),
            nn.ReLU(inplace=True),
            # no pooling in time or freq here
        )

        # After two freq pooling: freq_dim ~ n_mels // 4
        self.freq_reduction = max(1, n_mels // 4)
        # projection to desired d_model across frequency dim
        self.proj = nn.Conv2d(chans[2], d_model, kernel_size=(self.freq_reduction, 1))

    def forward(self, x):
        # x: (B, 1, F, T)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # (B, chans3, F//4, T)
        # project frequency to 1
        x = self.proj(x)   # (B, d_model, 1, T)
        x = x.squeeze(2)   # (B, d_model, T)
        x = x.permute(0, 2, 1)  # (B, T, d_model)
        return x

# ------------------------
# 3) Transformer encoder wrapper
# ------------------------
class TransformerEncoderModule(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1, max_len=5000):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True  # (B, T, D) in newer torch
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # simple learned positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x, src_key_padding_mask=None):
        # x: (B, T, D)
        B, T, D = x.shape
        if T > self.pos_embed.size(1):
            # Interpolate positional embeddings to match longer sequence lengths
            pe = self.pos_embed.permute(0, 2, 1)  # (1, D, L)
            pe = F.interpolate(pe, size=T, mode="linear", align_corners=False)
            pe = pe.permute(0, 2, 1)  # (1, T, D)
            pe = pe.to(x.dtype).to(x.device)
        else:
            pe = self.pos_embed[:, :T, :].to(x.dtype).to(x.device)
        x = x + pe
        return self.transformer(x, src_key_padding_mask=src_key_padding_mask)

# ------------------------
# 4) Full model
# ------------------------
class CNNTransformerChordModel(nn.Module):
    def __init__(
        self,
        n_classes: int,
        sample_rate: int = 44100,
        fps: int = 100,
        n_mels: int = 128,
        n_fft: int = 2048,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        freq_channels=[64,128,256],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.frontend = MelFrontend(sample_rate=sample_rate, fps=fps, n_mels=n_mels, n_fft=n_fft)
        self.cnn = CNNEncoder(n_mels=n_mels, d_model=d_model, chans=freq_channels)
        self.transformer = TransformerEncoderModule(d_model=d_model, nhead=nhead, num_layers=num_layers,
                                                    dim_feedforward=d_model*4, dropout=dropout)
        self.classifier = nn.Linear(d_model, n_classes)
        self._n_classes = n_classes
        self._fps = fps

    def forward(self, audio): 
        """
        audio: Tensor (B, seg_samples) or (seg_samples,) -> returns logits (B, T, n_classes)
        """
        single = False
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            single = True

        # Mel frontend: returns (B, n_mels, T)
        mel = self.frontend(audio)  # (B, n_mels, T)
        mel = mel.to(audio.dtype).to(audio.device)
        # ensure shape for CNN: (B, 1, F, T)
        mel = mel.unsqueeze(1)

        # CNN -> (B, T, d_model)
        x = self.cnn(mel)

        # Transformer -> (B, T, d_model)
        x = self.transformer(x)

        # classifier -> (B, T, n_classes)
        logits = self.classifier(x)

        return logits.squeeze(0) if single else logits  # keep consistent dims


class HMMDecoder:
    def __init__(self, transition_matrix):
        """
        transition_matrix: np.array of shape (n_classes, n_classes)
        """
        self.A = transition_matrix
        # Pre-compute log transition matrix for Viterbi
        # (librosa viterbi takes probabilities, but having them ready is good practice)
        self.n_classes = transition_matrix.shape[0]

    def decode(self, logits):
        """
        logits: Tensor of shape (T, n_classes) or (B, T, n_classes)
        Returns: decoded_indices (T,) or (B, T)
        """
        # Handle batch dimension
        if logits.dim() == 3:
            # Loop over batch (Viterbi is not easily batchable in standard librosa)
            batch_preds = []
            for i in range(logits.shape[0]):
                batch_preds.append(self.decode(logits[i]))
            return np.stack(batch_preds)

        # 1. Convert Logits to Probabilities
        # Softmax gives P(Observation | State)
        probs = F.softmax(logits, dim=-1).cpu().numpy() # Shape: (T, n_classes)
        
        # 2. Transpose for Librosa
        # Librosa expects (n_classes, n_steps)
        probs = probs.T 
        
        # 3. Run Viterbi
        # p: State likelihoods
        # transition: Transition matrix P(next|curr)
        path = librosa.sequence.viterbi(probs, self.A)
        
        return path