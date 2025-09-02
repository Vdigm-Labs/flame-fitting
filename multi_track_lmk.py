import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp

# --------------------
# 경로 설정
# --------------------
IMG_DIR   = Path("./data/BlenderFaceImage")
OUT_DIR   = Path("./data/BlenderOutput")
EMB_PATH  = "./models/mediapipe_landmark_embedding.npz"  # (길이 105 인덱스 배열 포함)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------
# 105 인덱스 로더 (키 유연 처리)
# --------------------
def load_105_indices(npz_path: str) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    # 가장 흔한 키 우선
    for k in ["landmark_indices", "mediapipe_indices", "indices", "lmk_idx", "lmk_ind"]:
        if k in data and isinstance(data[k], np.ndarray) and data[k].ndim == 1 and data[k].size == 105:
            return data[k].astype(np.int32)
    # 폴백: 길이 105인 1D 배열을 아무거나 사용
    for k in data.files:
        v = data[k]
        if isinstance(v, np.ndarray) and v.ndim == 1 and v.size == 105:
            return v.astype(np.int32)
    raise KeyError(f"{npz_path}에서 길이 105인 인덱스 배열을 찾지 못했습니다. 키들: {list(data.files)}")

# --------------------
# 단일 이미지 처리
# --------------------
def process_image(img_path: Path, idx_105: np.ndarray, face_mesh: mp.solutions.face_mesh.FaceMesh) -> None:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[skip] 이미지 로드 실패: {img_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    res = face_mesh.process(img_rgb)
    if not res.multi_face_landmarks:
        print(f"[skip] 얼굴 미검출: {img_path.name}")
        return

    lmk = res.multi_face_landmarks[0].landmark  # 478개
    all_xyz = np.array([[p.x * W, p.y * H, p.z] for p in lmk], dtype=np.float32)  # (478,3)

    # 105 포인트만 추출
    lmk_3d_105 = all_xyz[idx_105]           # (105,3)
    pts2d = lmk_3d_105[:, :2].copy()        # (105,2) 픽셀 좌표

    # 이미지 경계 클리핑
    pts2d[:, 0] = np.clip(pts2d[:, 0], 0, W - 1)
    pts2d[:, 1] = np.clip(pts2d[:, 1], 0, H - 1)

    # 저장 파일명 구성
    stem = img_path.stem  # 확장자 제외한 원본 파일명
    save_lmk_2d = OUT_DIR / f"lmk_2d_{stem}.npy"
    save_lmk_3d = OUT_DIR / f"lmk_3d_{stem}.npy"
    debug_png    = OUT_DIR / f"debug_{stem}.png"

    # 저장
    np.save(str(save_lmk_2d), pts2d.astype(np.float32))
    np.save(str(save_lmk_3d), lmk_3d_105.astype(np.float32))

    # 디버그 오버레이 (녹색 점)
    vis = img_bgr.copy()
    for (x, y) in pts2d.astype(int):
        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
    cv2.imwrite(str(debug_png), vis)

    print(f"✅ {img_path.name} → "
          f"{save_lmk_2d.name} (shape {pts2d.shape}), "
          f"{save_lmk_3d.name} (shape {lmk_3d_105.shape}), "
          f"{debug_png.name}")

# --------------------
# 메인: 디렉토리 전체 처리
# --------------------
def main():
    # 이미지 확장자들
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    img_paths = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in exts])

    if not img_paths:
        print(f"이미지 없음: {IMG_DIR.resolve()}")
        return

    # 105 인덱스 로드
    idx_105 = load_105_indices(EMB_PATH)

    # MediaPipe FaceMesh 초기화
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True
    ) as face_mesh:

        for p in img_paths:
            try:
                process_image(p, idx_105, face_mesh)
            except Exception as e:
                print(f"[error] {p.name}: {e}")

if __name__ == "__main__":
    main()
