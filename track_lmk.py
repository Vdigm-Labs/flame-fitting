"""
track_lmk105.py
──────────────────────────────────────────────────────────────
1) Mediapipe FaceMesh  → 478 포인트 검출
2) 105 개 중요 포인트만 추출 (embedding NPZ 참조)
3) MiDaS DPT_Hybrid    → depth 맵 예측
4) (x, y, z) 3D 랜드마크 생성 및 저장
    • ./data/scan_lmks.npy      (105,3)
    • ./data/embedding_105.npz  (lmk_face_idx, lmk_b_coords)
──────────────────────────────────────────────────────────────
필수 pip 패키지:
numpy torch torchvision torchaudio timm mediapipe opencv-python
"""

import cv2, torch, numpy as np, mediapipe as mp
from pathlib import Path

# ─────────── 사용자 입력 ───────────
IMG_PATH = "./data/test.jpg"                       # 분석할 이미지
EMB_PATH = "./models/mediapipe_landmark_embedding.npz"  # 105포인트 임베딩
SAVE_LMK = "./data/scan_lmks.npy"                  # 저장 경로
SAVE_EMB = "./data/embedding_105.npz"              # 선택: 임베딩 재저장

# ─────────── MiDaS 로드 ───────────
device = "cuda" if torch.cuda.is_available() else "cpu"
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
midas.to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform = midas_transforms.dpt_transform  # 이미 ToTensor+unsqueeze 포함

# ─────────── Mediapipe FaceMesh ────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                  max_num_faces=1,
                                  refine_landmarks=True)

# ─────────── 1) 이미지 불러오기 ────
img_bgr = cv2.imread(IMG_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"이미지 없음: {IMG_PATH}")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w, _ = img_rgb.shape

# ─────────── 2) Mediapipe 2D 랜드마크 ─
res = face_mesh.process(img_rgb)
if not res.multi_face_landmarks:
    raise RuntimeError("얼굴을 찾지 못했어요 😢")
all_xy = np.array([[lm.x * w, lm.y * h] for lm in res.multi_face_landmarks[0].landmark])  # (478,2)

# ─────────── 3) 105 포인트 임베딩 로드 ─
emb = np.load(EMB_PATH)
lmk_face_idx = emb["lmk_face_idx"]        # (105,)
lmk_b_coords = emb["lmk_b_coords"]        # (105,3)
idx105       = emb["landmark_indices"]    # (105,)
xy_105 = all_xy[idx105]                   # (105,2)

# ─────────── 4) MiDaS depth 추정 ──────
input_tensor = transform(img_rgb).to(device)   # (1,C,H,W)
with torch.no_grad():
    depth = midas(input_tensor)
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
    ).squeeze().cpu().numpy()                  # (H,W)

z_105 = np.array([depth[int(y), int(x)] for x, y in xy_105])
lmk_3d = np.stack([xy_105[:, 0], xy_105[:, 1], z_105], axis=1)  # (105,3)

# ─────────── 5) 결과 저장 ────────────
Path(SAVE_LMK).parent.mkdir(parents=True, exist_ok=True)
np.save(SAVE_LMK, lmk_3d.astype(np.float32))
np.savez(SAVE_EMB, lmk_face_idx=lmk_face_idx, lmk_b_coords=lmk_b_coords)
print(f"✅ 3D 랜드마크 저장: {SAVE_LMK}  (shape {lmk_3d.shape})")
print(f"✅ 임베딩 복사본 저장: {SAVE_EMB}")
    