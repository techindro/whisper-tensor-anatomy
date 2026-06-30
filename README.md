# Whisper Tensor Anatomy

A minimal test bench to track tensor dimension flow inside OpenAI's Whisper model and validate weight compression via SVD.

## What this does
1. **Bypasses Disk I/O:** `run_whisper.py` uses synthetic audio data (`torch.randn(1, 80, 3000)`) to simulate a 30-second audio stream directly into the architecture.
2. **Tracks Matrix Shapes:** Custom print statements inside `model.py` (`qkv_attention`) capture dynamic shapes before matrix multiplication.
3. **Validates SVD:** `test.py` applies Singular Value Decomposition on a 1000x1000 weight matrix to verify low-rank approximation mechanics.

## Architecture & Math Checked
* **Attention Mechanism:** Verified the $Q \times K^T$ transpose and matrix multiplication dimensions inside Multi-Head Attention.
* **Decoder Masking:** Inspected the upper triangular matrix (`triu_(1)`) operation filling future tokens with `-inf` to prevent look-ahead leaks.
* **SVD Compression:** Evaluated how matrix energy concentrates into top singular values for low-rank approximations.

## Files
* `model.py`: Modified Whisper core layer with attention shape hooks.
* `run_whisper.py`: Entry point using synthetic data pipelines.
* `test.py`: Standalone SVD matrix energy distribution test.
