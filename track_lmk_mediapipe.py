"""
quick_test_lmk3d_105.py
───────────────────────────────────────────────────────
▶ Mediapipe FaceMesh → 478포인트 전체 3D 검출
▶ 105포인트만 임베딩 기반 추출 (x, y 픽셀 변환 + z 그대로)
▶ z는 깊이 scale 없이 상대 값 유지
───────────────────────────────────────────────────────
필수 패키지: mediapipe · numpy · opencv-python
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

# ─────────── 설정 ───────────
IMG_PATH = "./data/madong.jpg"                       # 분석 이미지
EMB_PATH = "./models/mediapipe_landmark_embedding.npz"  # 105포인트 임베딩
SAVE_LMK = "./data/my_scan_lmks.npy"             # 결과 저장 (105,3)

# ─────────── Mediapipe FaceMesh ───────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

# ─────────── 이미지 로드 ───────────
img_bgr = cv2.imread(IMG_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"이미지 없음: {IMG_PATH}")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w, _ = img_rgb.shape

# ─────────── 랜드마크 3D 검출 ───────────
res = face_mesh.process(img_rgb)
if not res.multi_face_landmarks:
    raise RuntimeError("❌ 얼굴을 찾을 수 없습니다.")

all_lmk = res.multi_face_landmarks[0].landmark

# (478, 3) → x, y 픽셀 변환, z 그대로
all_xyz = np.array([
    [lm.x * w, lm.y * h * -1, lm.z] for lm in all_lmk
], dtype=np.float32)

# ─────────── 105포인트 임베딩 기반 추출 ───────────
emb = np.load(EMB_PATH)
idx_105 = emb["landmark_indices"]   # (105,)

lmk_3d_105 = all_xyz[idx_105]       # (105, 3)

# ─────────── 결과 저장 ───────────
Path(SAVE_LMK).parent.mkdir(parents=True, exist_ok=True)
np.save(SAVE_LMK, lmk_3d_105)
print(f"✅ 105 랜드마크 (Mediapipe z 포함) 저장 ▶ {SAVE_LMK}  (shape {lmk_3d_105.shape})")
