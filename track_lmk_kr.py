"""
track_lmk_kr.py
────────────────────────────────────────────────────────────
English ⇢ Korean detailed inline comment version
"""

# ------------------------------------------------------------------------------------------
# EN: Imports – bring in all libraries we need
# KO: 필요한 라이브러리 임포트
# ------------------------------------------------------------------------------------------
from pathlib import Path

import cv2                    # EN: OpenCV for image I/O   | KO: 이미지 입출력용 OpenCV
import numpy as np            # EN: Numerical operations   | KO: 수치 계산 라이브러리
import torch                  # EN: PyTorch for MiDaS       | KO: MiDaS 추론용 PyTorch
import mediapipe as mp        # EN: Google FaceMesh         | KO: 구글 FaceMesh
from PIL import Image         # EN: Pillow for MiDaS input  | KO: MiDaS 입력용 Pillow
from scipy.ndimage import map_coordinates, median_filter  # EN: interpolation & denoise
                                                             # KO: 보간‧노이즈 제거

# ------------------------------------------------------------------------------------------
# EN: User-defined paths / parameters
# KO: 사용자 설정 (경로 및 파라미터)
# ------------------------------------------------------------------------------------------
IMG_PATH = "./data/test.jpg"                       # EN: path to input photo
                                                  # KO: 분석할 이미지 경로
EMB_PATH = "./models/mediapipe_landmark_embedding.npz"  # EN: 105-point embedding file
                                                  # KO: 105포인트 임베딩 NPZ
SAVE_LMK = "./data/scan_lmks.npy"                 # EN: output .npy (105,3)
                                                  # KO: 결과 랜드마크 저장 경로
SAVE_EMB = "./data/embedding_105.npz"             # EN: copy embedding for FLAME
                                                  # KO: 임베딩 복사본 저장
LEFT_EYE, RIGHT_EYE = 33, 263                     # EN/KO: FaceMesh eye indices

# ------------------------------------------------------------------------------------------
# EN: (0) Load MiDaS depth model – DPT-Hybrid variant
# KO: (0) MiDaS DPT-Hybrid 모델 로드
# ------------------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid",
                       trust_repo=True).to(device).eval()

midas_tfms = torch.hub.load("intel-isl/MiDaS", "transforms",
                            trust_repo=True)
transform = midas_tfms.dpt_transform   # EN: PIL → tensor transform
                                       # KO: PIL 이미지를 텐서로 변환

# ------------------------------------------------------------------------------------------
# EN: (1) Initialise Mediapipe FaceMesh (single image, one face)
# KO: (1) Mediapipe FaceMesh 초기화 (단일 이미지, 얼굴 1개)
# ------------------------------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,     # EN: no tracking, independent frame
                                # KO: 동영상 추적 대신 정적 이미지 모드
    max_num_faces=1,            # EN: detect at most one face
                                # KO: 최대 한 얼굴만 찾기
    refine_landmarks=True       # EN: more accurate eye/iris landmarks
                                # KO: 눈/홍채 랜드마크 정밀도 향상
)

# ------------------------------------------------------------------------------------------
# EN: (2) Read the image with OpenCV and convert to PIL for MiDaS
# KO: (2) 이미지를 읽어 Pillow 형식으로 변환 (MiDaS 입력용)
# ------------------------------------------------------------------------------------------
bgr = cv2.imread(IMG_PATH)
if bgr is None:
    raise FileNotFoundError(f"Image not found: {IMG_PATH}")

rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)   # EN: OpenCV BGR → RGB
                                             # KO: BGR에서 RGB로 변환
pil = Image.fromarray(rgb)                   # EN: make PIL image
                                             # KO: PIL 이미지 객체 생성
H, W = rgb.shape[:2]

# ------------------------------------------------------------------------------------------
# EN: (3) Detect 2-D landmarks (478 pts) with FaceMesh
# KO: (3) FaceMesh로 478개 2-D 랜드마크 검출
# ------------------------------------------------------------------------------------------
res = face_mesh.process(rgb)
if not res.multi_face_landmarks:
    raise RuntimeError("❌ Face not detected.")

all_xy = np.array(
    [[lm.x * W, lm.y * H]                     # EN: scale to pixel space
                                             # KO: 정규화 좌표 → 픽셀 단위
     for lm in res.multi_face_landmarks[0].landmark],
    dtype=np.float32
)                                            # shape = (478,2)

# ------------------------------------------------------------------------------------------
# EN: (4) Load 105-point embedding and extract those 2-D points
# KO: (4) 105포인트 임베딩 로드 후 해당 2-D 좌표 추출
# ------------------------------------------------------------------------------------------
emb          = np.load(EMB_PATH)
face_idx     = emb["lmk_face_idx"]     # EN/KO: 삼각형 face index
bary_coords  = emb["lmk_b_coords"]     # EN/KO: barycentric weights
idx105       = emb["landmark_indices"] # EN/KO: 105개 인덱스
xy105_px     = all_xy[idx105]          # EN: selected 105 (x,y) pixels
                                        # KO: 선택 105개 픽셀 좌표

# ------------------------------------------------------------------------------------------
# EN: (5) Run MiDaS – crop face for cleaner depth, then merge back
# KO: (5) MiDaS 추론 – 얼굴만 잘라 깊이 예측 후 원본 크기로 삽입
# ------------------------------------------------------------------------------------------
# (a) compute loose bounding box around landmarks
xmin, ymin = xy105_px.min(0).astype(int) - 60
xmax, ymax = xy105_px.max(0).astype(int) + 60
xmin, ymin = np.clip([xmin, ymin], 0, [W-1, H-1])
xmax, ymax = np.clip([xmax, ymax], 0, [W-1, H-1])

crop_pil  = pil.crop((xmin, ymin, xmax, ymax))
crop_rgb  = np.asarray(crop_pil)

# (b) MiDaS inference on the cropped image
with torch.no_grad():
    depth_crop = midas(transform(crop_pil).to(device))
    depth_crop = torch.nn.functional.interpolate(
        depth_crop.unsqueeze(1),
        size=crop_rgb.shape[:2],          # EN: back to crop size
                                         # KO: crop 해상도로 리사이즈
        mode="bicubic", align_corners=False
    ).squeeze().cpu().numpy()            # shape (cropH, cropW)

# (c) paste depth back to full image canvas
depth = np.full((H, W), depth_crop.mean(), np.float32)  # EN: init with mean
                                                        # KO: 평균값으로 채움
depth[ymin:ymax, xmin:xmax] = depth_crop
depth = median_filter(depth, size=5)  # EN: reduce noise | KO: 미디언 필터

# ------------------------------------------------------------------------------------------
# EN: (6) Sample Z at 105 points with bilinear interpolation
# KO: (6) 105지점 Z값을 Bilinear 보간으로 샘플링
# ------------------------------------------------------------------------------------------
coords   = np.vstack([xy105_px[:, 1], xy105_px[:, 0]])   # (2,N)= (y,x)
z105_raw = map_coordinates(depth, coords, order=1)       # bilinear

# ------------------------------------------------------------------------------------------
# EN: (7) Convert pixel-space (x,y) and raw z → metric 3-D
# KO: (7) 픽셀 (x,y) + 상대 z → 미터 단위 3-D 변환
# ------------------------------------------------------------------------------------------
# (a) Eye-to-eye pixel distance
eye_px  = np.linalg.norm(all_xy[LEFT_EYE] - all_xy[RIGHT_EYE])
avg_mm  = 63.0                    # EN: average 63 mm | KO: 평균 63mm
px2m    = (avg_mm/1000.0) / eye_px  # pixel → meter scale

# (b) center each (x,y) on face centroid and scale
center_xy = xy105_px.mean(0)
xy_m = (xy105_px - center_xy) * px2m

# (c) normalize & scale z
z_centered = (z105_raw - np.median(z105_raw)) * px2m

# (d) stack into final (105,3) array
lmk3d_m = np.stack([xy_m[:,0], xy_m[:,1], z_centered], axis=1).astype(np.float32)

# ------------------------------------------------------------------------------------------
# EN: (8) Save results (.npy & .npz) – ready for FLAME fitting
# KO: (8) 결과 저장 – FLAME 피팅에 바로 사용 가능
# ------------------------------------------------------------------------------------------
Path(SAVE_LMK).parent.mkdir(parents=True, exist_ok=True)
np.save(SAVE_LMK, lmk3d_m)
np.savez(SAVE_EMB, lmk_face_idx=face_idx, lmk_b_coords=bary_coords)

print(f"✅ 3-D landmarks saved ▶ {SAVE_LMK}  (shape {lmk3d_m.shape})")
print(f"✅ embedding copy saved ▶ {SAVE_EMB}")
