"""
track_lmk105.py
──────────────────────────────────────────────────────────────
1) Mediapipe FaceMesh  → detect 478 facial landmarks
2) Extract 105 key landmarks (based on embedding NPZ)
3) MiDaS DPT_Hybrid    → predict depth map
4) Generate & save 3D landmarks (x, y, z)
    • ./data/scan_lmks.npy      (105,3)
    • ./data/embedding_105.npz  (lmk_face_idx, lmk_b_coords)
──────────────────────────────────────────────────────────────
Required pip packages:
numpy torch torchvision torchaudio timm mediapipe opencv-python
"""

import cv2, torch, numpy as np, mediapipe as mp
from pathlib import Path

# ─────────── User Input ───────────
IMG_PATH = "./data/test.jpg"                       # Input image to analyze
EMB_PATH = "./models/mediapipe_landmark_embedding.npz"  # 105-point embedding reference
SAVE_LMK = "./data/scan_lmks.npy"                  # Output path for 3D landmarks
SAVE_EMB = "./data/embedding_105.npz"              # Optional: save embedding copy

# ─────────── Load MiDaS Model ───────────
device = "cuda" if torch.cuda.is_available() else "cpu"
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
midas.to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform = midas_transforms.dpt_transform  # Includes ToTensor and unsqueeze

# ─────────── Load Mediapipe FaceMesh ────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                  max_num_faces=1,
                                  refine_landmarks=True)

# ─────────── 1) Load Image ───────────
img_bgr = cv2.imread(IMG_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"Image not found: {IMG_PATH}")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w, _ = img_rgb.shape

# ─────────── 2) Detect 2D Landmarks ─────
res = face_mesh.process(img_rgb)
if not res.multi_face_landmarks:
    raise RuntimeError("No face detected 😢")
all_xy = np.array([[lm.x * w, lm.y * h] for lm in res.multi_face_landmarks[0].landmark])  # (478,2)

# ─────────── 3) Load 105-Point Embedding ─
emb = np.load(EMB_PATH)
lmk_face_idx = emb["lmk_face_idx"]        # (105,)
lmk_b_coords = emb["lmk_b_coords"]        # (105,3)
idx105       = emb["landmark_indices"]    # (105,)
xy_105 = all_xy[idx105]                   # (105,2)

# ─────────── 4) Estimate Depth with MiDaS ─
input_tensor = transform(img_rgb).to(device)   # (1,C,H,W)
with torch.no_grad():
    depth = midas(input_tensor)
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
    ).squeeze().cpu().numpy()                  # (H,W)

z_105 = np.array([depth[int(y), int(x)] for x, y in xy_105])
lmk_3d = np.stack([xy_105[:, 0], xy_105[:, 1], z_105], axis=1)  # (105,3)

# ─────────── 5) Save Results ───────────
Path(SAVE_LMK).parent.mkdir(parents=True, exist_ok=True)
np.save(SAVE_LMK, lmk_3d.astype(np.float32))
np.savez(SAVE_EMB, lmk_face_idx=lmk_face_idx, lmk_b_coords=lmk_b_coords)
print(f"✅ Saved 3D landmarks: {SAVE_LMK}  (shape {lmk_3d.shape})")
print(f"✅ Saved embedding copy: {SAVE_EMB}")
    