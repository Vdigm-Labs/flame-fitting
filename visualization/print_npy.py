import numpy as np

# Load embedding file
embedding_path = "mediapipe_landmark_embedding.npz"
data = np.load(embedding_path)

# Extract arrays
lmk_face_idx = data["lmk_face_idx"]
lmk_b_coords = data["lmk_b_coords"]
landmark_indices = data["landmark_indices"]

# Print info
print(f"\n📂 Loaded from: {embedding_path}")

print("\n🔹 lmk_face_idx (triangle index per landmark):")
print(f"Shape: {lmk_face_idx.shape}")
print(lmk_face_idx)

print("\n🔹 lmk_b_coords (barycentric coordinates):")
print(f"Shape: {lmk_b_coords.shape}")
print(lmk_b_coords)

print("\n🔹 landmark_indices (Mediapipe mesh indices):")
print(f"Shape: {landmark_indices.shape}")
print(landmark_indices)
