"""
track_lmk_kr.py
────────────────────────────────────────────────────────────
English ⇢ Korean detailed inline comment version
(수정본 – 2025-07-12, Z 스케일 자동 보정 + 코 기준 중심화 반영)
"""

# ------------------------------------------------------------------------------------------
# EN: Imports – bring in all libraries we need
# KO: 필요한 라이브러리 임포트
# ------------------------------------------------------------------------------------------
from pathlib import Path
import cv2
import numpy as np
import torch
import mediapipe as mp
from PIL import Image
from scipy.ndimage import map_coordinates, median_filter
from torchvision.transforms.functional import to_tensor  # noqa: F401 (torch.hub transform 내부 요구)

# ------------------------------------------------------------------------------------------
# EN: User-defined paths / parameters
# KO: 사용자 설정 (경로 및 파라미터)
# ------------------------------------------------------------------------------------------
IMG_PATH  = "./data/test.jpg"                         # 입력 사진
EMB_PATH  = "./models/mediapipe_landmark_embedding.npz"  # 105포인트 임베딩 파일
SAVE_LMK  = "./data/my_scan_lmks.npy"                 # (105,3) 3D 랜드마크 저장
SAVE_EMB  = "./data/mediapipe_landmark_embedding.npz" # FLAME용 임베딩 사본
LEFT_EYE, RIGHT_EYE = 33, 263                         # FaceMesh 눈 인덱스
NOSE_TIP = 1                                          # FaceMesh 코 끝 인덱스 (landmark #1)
TARGET_Z_STD_MM = 25.0                               # 얼굴 깊이 표준편차 목표(≈2.5 cm)

# ------------------------------------------------------------------------------------------
# EN: (0) Load MiDaS depth model – DPT-Hybrid variant
# KO: (0) MiDaS DPT-Hybrid 모델 로드
# ------------------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
midas = torch.hub.load(
    "intel-isl/MiDaS", "DPT_Hybrid",
    trust_repo=True, skip_validation=True, force_reload=False
).to(device).eval()

midas_tfms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform = midas_tfms.dpt_transform   # PIL → tensor 변환

# ------------------------------------------------------------------------------------------
# EN: (1) Initialise Mediapipe FaceMesh (single image, one face)
# KO: (1) Mediapipe FaceMesh 초기화 (단일 이미지, 얼굴 1개)
# ------------------------------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

# ------------------------------------------------------------------------------------------
# EN: (2) Read image and convert to PIL for MiDaS
# KO: (2) 이미지 읽고 PIL 형식으로 변환
# ------------------------------------------------------------------------------------------
bgr = cv2.imread(IMG_PATH)
if bgr is None:
    raise FileNotFoundError(f"Image not found: {IMG_PATH}")

rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
pil = Image.fromarray(rgb)
H, W = rgb.shape[:2]

# ------------------------------------------------------------------------------------------
# EN: (3) Detect 2-D landmarks (478 pts) with FaceMesh
# KO: (3) FaceMesh로 2-D 랜드마크 검출
# ------------------------------------------------------------------------------------------
res = face_mesh.process(rgb)
if not res.multi_face_landmarks:
    raise RuntimeError("❌ Face not detected.")

all_xy = np.array(
    [[lm.x * W, lm.y * H] for lm in res.multi_face_landmarks[0].landmark],
    dtype=np.float32
)  # shape = (478,2)

# ------------------------------------------------------------------------------------------
# EN: (4) Load 105-point embedding and pick those 2-D points
# KO: (4) 105포인트 임베딩 로드 및 좌표 추출
# ------------------------------------------------------------------------------------------
emb          = np.load(EMB_PATH)
face_idx     = emb["lmk_face_idx"]      # 삼각형 face index
bary_coords  = emb["lmk_b_coords"]      # barycentric weights
idx105       = emb["landmark_indices"]  # Mediapipe 인덱스 105개
xy105_px     = all_xy[idx105]           # (105,2) 픽셀 좌표

# ------------------------------------------------------------------------------------------
# EN: (5) Run MiDaS on a cropped face
# KO: (5) 얼굴 영역 크롭 후 MiDaS 추론
# ------------------------------------------------------------------------------------------
padding = 60
xmin, ymin = xy105_px.min(0).astype(int) - padding
xmax, ymax = xy105_px.max(0).astype(int) + padding
xmin, ymin = np.clip([xmin, ymin], 0, [W - 1, H - 1])
xmax, ymax = np.clip([xmax, ymax], 0, [W - 1, H - 1])

crop_pil  = pil.crop((xmin, ymin, xmax, ymax))
crop_rgb  = np.asarray(crop_pil)

with torch.no_grad():
    tf_out = transform(crop_rgb)
    input_tensor = tf_out["image"] if isinstance(tf_out, dict) else tf_out
    if input_tensor.dim() == 3:  # (C,H,W) → (1,C,H,W)
        input_tensor = input_tensor.unsqueeze(0)
    depth_crop = midas(input_tensor.to(device))
    depth_crop = torch.nn.functional.interpolate(
        depth_crop.unsqueeze(1),
        size=crop_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

# 원본 캔버스에 붙이기 (간단한 덮어쓰기 + 노이즈 완화)
depth = np.full((H, W), depth_crop.mean(), np.float32)
depth[ymin:ymax, xmin:xmax] = depth_crop
depth = median_filter(depth, size=5)  # 미디언 필터로 경계 노이즈 완화

# ------------------------------------------------------------------------------------------
# EN: (6) Sample Z at 105 points (bilinear)
# KO: (6) 105지점에서 Z 샘플링
# ------------------------------------------------------------------------------------------
coords   = np.vstack([xy105_px[:, 1], xy105_px[:, 0]])  # (y,x) order
z105_raw = map_coordinates(depth, coords, order=1)

# ------------------------------------------------------------------------------------------
# EN: (7) Pixel (x,y,z) → metric 3D
# KO: (7) 픽셀 (x,y,z) → 미터 단위 3D 변환
# ------------------------------------------------------------------------------------------
# (a) 눈-눈 거리로 스케일 추정 (pixel ↔ m)
eye_px  = np.linalg.norm(all_xy[LEFT_EYE] - all_xy[RIGHT_EYE])
px2m    = (63.0 / 1000.0) / eye_px   # 63 mm 평균

# (b) (x,y) 중심화·스케일 + Y축 뒤집기 (코 위치 기준)
center_xy = all_xy[NOSE_TIP]          # 코 끝을 기준으로 중앙 정렬
xy_m = (xy105_px - center_xy) * px2m
xy_m[:, 1] *= -1                      # 픽셀 아래→ FLAME 위 방향

# (c) Z 부호 반전
z_tmp = -(z105_raw - np.median(z105_raw)) * px2m

# (d) 얼굴 별 Z 표준편차 맞추기 (25 mm)
z_std = np.std(z_tmp)
target_std = (TARGET_Z_STD_MM / 1000.0)
scale_factor = np.clip(target_std / (z_std + 1e-9), 0.5, 3.0)  # 안전 범위
print(f"[INFO] z_std={z_std:.5f} m → scale ×{scale_factor:.2f}")
z_centered = z_tmp * scale_factor

# (e) (105,3) 스택
lmk3d_m = np.column_stack((xy_m[:, 0], xy_m[:, 1], z_centered)).astype(np.float32)

# ------------------------------------------------------------------------------------------
# EN: (8) Save results – ready for FLAME fitting
# KO: (8) 결과 저장 – FLAME 피팅 준비 완료
# ------------------------------------------------------------------------------------------
Path(SAVE_LMK).parent.mkdir(parents=True, exist_ok=True)
np.save(SAVE_LMK, lmk3d_m)
np.savez(SAVE_EMB, lmk_face_idx=face_idx, lmk_b_coords=bary_coords)

print(f"✅ 3D landmarks saved ▶ {SAVE_LMK}  (shape {lmk3d_m.shape})")
print(f"✅ embedding copy saved ▶ {SAVE_EMB}")
