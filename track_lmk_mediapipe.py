import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp

# --------------------
# 경로 설정: coefficent
# --------------------
IMG_PATH = "./data/test.jpg"
IMG_PATH = "./data/1_Base.png"
EMB_PATH = "./models/mediapipe_landmark_embedding.npz"  # 105 인덱스pip install -r requirements.txt
SAVE_LMK_2D = "./data/my_scan_lmks.npy"                 # (105,2) – Fitting용
SAVE_LMK_3D = "./data/my_scan_lmks_105x3.npy"           # (105,3) – 참고용
DEBUG_PNG   = "./data/my_scan_lmks_debug.png"

# --------------------
# 105 인덱스 로더 (키 유연 처리)
# --------------------
def load_105_indices(npz_path: str) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    # 가장 흔한 키 우선
    for k in ["landmark_indices", "mediapipe_indices", "indices", "lmk_idx", "lmk_ind"]:
        if k in data and data[k].ndim == 1 and data[k].size == 105:
            return data[k].astype(np.int32)
    # 폴백: 길이 105인 1D 배열을 아무거나 사용
    for k in data.files:
        v = data[k]
        if isinstance(v, np.ndarray) and v.ndim == 1 and v.size == 105:
            return v.astype(np.int32)
    raise KeyError(f"{npz_path}에서 길이 105인 인덱스 배열을 찾지 못했습니다. 키들: {list(data.files)}")

# --------------------
# 메인
# --------------------
def main():
    # 이미지 읽기
    img_bgr = cv2.imread(IMG_PATH)
    if img_bgr is None:
        raise FileNotFoundError(f"이미지 없음: {IMG_PATH}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    # 임베딩(105 인덱스) 로드
    idx_105 = load_105_indices(EMB_PATH)

    # MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True
    ) as face_mesh:

        res = face_mesh.process(img_rgb)
        if not res.multi_face_landmarks:
            raise RuntimeError("❌ 얼굴을 찾을 수 없습니다.")

        lmk = res.multi_face_landmarks[0].landmark  # 478개
        # (478,3): x,y는 픽셀로 변환, z는 MP 상대깊이 그대로
        all_xyz = np.array([[p.x * W, p.y * H, p.z] for p in lmk], dtype=np.float32)

        # 105 포인트만 추출
        lmk_3d_105 = all_xyz[idx_105]          # (105,3)
        pts2d = lmk_3d_105[:, :2].copy()        # (105,2) 픽셀 좌표
        # 이미지 경계 클리핑
        pts2d[:, 0] = np.clip(pts2d[:, 0], 0, W - 1)
        pts2d[:, 1] = np.clip(pts2d[:, 1], 0, H - 1)

        # 저장
        Path(SAVE_LMK_2D).parent.mkdir(parents=True, exist_ok=True)
        np.save(SAVE_LMK_2D, pts2d.astype(np.float32))
        np.save(SAVE_LMK_3D, lmk_3d_105.astype(np.float32))

        # 디버그 오버레이 (녹색 점)
        vis = img_bgr.copy()
        for (x, y) in pts2d.astype(int):
            cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
        cv2.imwrite(DEBUG_PNG, vis)

        print(f"✅ 저장 완료: {SAVE_LMK_2D}  (shape {pts2d.shape})")
        print(f"ℹ️ 참고 저장: {SAVE_LMK_3D}  (shape {lmk_3d_105.shape})")
        print(f"🖼️ 디버그:   {DEBUG_PNG}")

if __name__ == "__main__":
    main()
