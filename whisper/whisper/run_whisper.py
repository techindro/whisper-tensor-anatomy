import sys
import os
import torch
import whisper
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from whisper.model import ModelDimensions, Whisper

dims = ModelDimensions(
    n_mels=80,
    n_audio_ctx=1500,
    n_audio_state=384,
    n_audio_head=6,
    n_audio_layer=4,
    n_vocab=51865,
    n_text_ctx=448,
    n_text_state=384,
    n_text_head=6,
    n_text_layer=4,
)
model = Whisper(dims)

n_ctx = 5
mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
print(mask)

audio_data = torch.randn(1, 80, 3000)

try:
    audio_features = model.encoder(audio_data)
    print("Encoder Features Shape:", audio_features.shape)

    tokens = torch.randint(0, 50000, (1, 10))
    logits = model.decoder(tokens, audio_features)
    print("Decoder Logits Shape:", logits.shape)
except Exception as e:
    print("Error:", e)
