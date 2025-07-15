import numpy as np
import pickle
from pathlib import Path

# ── 경로 설정 ──────────────────────────────────────────────
PKL_PATH = Path("../models/flame_static_embedding.pkl")           # 51 행 pkl
SRC_PATH = Path("../models/mediapipe_landmark_embedding.npz")      # 105 행 npz
DST_PATH = SRC_PATH.with_name("mediapipe_landmark_51embedding.npz")

# 1️⃣ pkl 에서 51 행 그대로 불러오기
with open(PKL_PATH, "rb") as f:
    pkl_data = pickle.load(f, encoding="latin1")

lmk_face_idx = pkl_data["lmk_face_idx"]          # (51,)
lmk_b_coords = pkl_data["lmk_b_coords"]          # (51,3)  또는 (3,51)

# 2️⃣ npz 에서 105 개 landmark_indices 불러오기
all_indices = np.load(SRC_PATH)["landmark_indices"]   # (105,)

# 3️⃣ 51 개만 추릴 keep_idx (원하는 순서)
keep_idx = np.array([
    10, 12, 11, 15, 13, 3, 5, 1, 2, 0,
    52, 54, 55, 57, 58, 60, 61, 62, 64, 37,
    47, 44, 38, 41, 39, 22, 29, 31, 21, 23,
    25, 72, 71, 69, 65, 87, 89, 90, 98, 102,
    68, 84, 80, 86, 75, 66, 93, 91, 101, 67,
    83
])

# 4️⃣ 새 landmark_indices 생성 (51,)
new_landmark_indices = all_indices[keep_idx]

# 5️⃣ 필요 시 lmk_b_coords shape 맞추기 (행이 51이어야 함)
if lmk_b_coords.shape[0] != 51:
    lmk_b_coords = lmk_b_coords.T   # (3,51) → (51,3)

# 6️⃣ 새 npz 저장
np.savez(
    DST_PATH,
    lmk_face_idx=lmk_face_idx,
    lmk_b_coords=lmk_b_coords,
    landmark_indices=new_landmark_indices
)

print("✅ Saved 51-landmark embedding:", DST_PATH)
print("   lmk_face_idx      :", lmk_face_idx.shape)
print("   lmk_b_coords      :", lmk_b_coords.shape)
print("   landmark_indices  :", new_landmark_indices.shape)
