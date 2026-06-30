import torch

print("--- Step 1: Creating a Large AI Weight Matrix ---")
A = torch.randn(1000, 1000)

print("\n--- Step 2: Applying Singular Value Decomposition (SVD) ---")
U, S, Vh = torch.linalg.svd(A)

print("Left-Singular Matrix (U) Shape:", U.shape)
print("Singular Values (S) Shape:", S.shape)
print("Right-Singular Matrix (Vh) Shape:", Vh.shape)

print("\n--- Step 3: Validating Top Information Weights ---")
top_10_energy = torch.sum(S[:10] ** 2) / torch.sum(S**2)
print(
    f"Top 10 singular values contain {top_10_energy.item()*100:.2f}% of the total matrix information!"
)
