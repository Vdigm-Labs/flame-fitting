"""
compare_lmks.py
────────────────────────────────────────────────────────
두 개의 3D 얼굴 랜드마크(.npy) 파일을 불러와
① 2D 투영(x-y 평면) 비교
② 3D 산점도(x, y, z) 비교
한 화면에 시각화하고 PNG로 저장합니다.
────────────────────────────────────────────────────────
사용 예:
$ python compare_lmks.py --file_a ./data/jinju_lmks.npy \
                         --file_b ./data/madong_lmks.npy \
                         --label_a Jinju --label_b MaDong
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ────── 1. 인자 파싱 ──────
parser = argparse.ArgumentParser(description="Compare two 3D landmark npy files visually.")
parser.add_argument("--file_a", required=True, help="첫 번째 .npy 경로")
parser.add_argument("--file_b", required=True, help="두 번째 .npy 경로")
parser.add_argument("--label_a", default="A", help="첫 번째 데이터 라벨")
parser.add_argument("--label_b", default="B", help="두 번째 데이터 라벨")
parser.add_argument("--output", default="landmark_compare.png", help="저장할 이미지 이름")
args = parser.parse_args()

# ────── 2. 데이터 로드 ──────
lmk_a = np.load(args.file_a)
lmk_b = np.load(args.file_b)

if lmk_a.shape != lmk_b.shape:
    print(f"⚠️  Warning: shape mismatch → {lmk_a.shape} vs {lmk_b.shape}")

# ────── 3. 시각화 시작 ──────
fig = plt.figure(figsize=(14, 6))

# (1) 2D 시각화 (x, y)
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(lmk_a[:, 0], lmk_a[:, 1], s=10, c="dodgerblue", label=args.label_a, alpha=0.7)
ax1.scatter(lmk_b[:, 0], lmk_b[:, 1], s=10, c="crimson", label=args.label_b, alpha=0.7)
ax1.set_title("2D Projection (x, y)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.axis("equal")
ax1.invert_yaxis()
ax1.legend()

# (2) 3D 시각화
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.scatter(lmk_a[:, 0], lmk_a[:, 1], lmk_a[:, 2], s=8, c="dodgerblue", label=args.label_a, alpha=0.7)
ax2.scatter(lmk_b[:, 0], lmk_b[:, 1], lmk_b[:, 2], s=8, c="crimson", label=args.label_b, alpha=0.7)
ax2.set_title("3D Landmarks (x, y, z)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("z (depth)")
ax2.view_init(elev=25, azim=-75)
ax2.legend()

plt.tight_layout()
plt.savefig(args.output, dpi=300)
print(f"✅ 이미지 저장 완료: {args.output}")

# python compare_lmk3d.py --file_a ./data/jinju_lmks.npy --file_b ./data/madong_lmks.npy --label_a Jinju --label_b MaDong --output ./data/landmark_compare.png
# python compare_lmk3d.py \
#   --file_a ./data/jinju_lmks.npy \
#   --file_b ./data/madong_lmks.npy \
#   --label_a Jinju \
#   --label_b MaDong \
#   --output ./data/landmark_compare.png
# python compare_lmk3d.py \
#   --file_a ./data/test_lmks.npy \
#   --file_b ./data/madong_lmks.npy \
#   --label_a test \
#   --label_b MaDong \
#   --output ./data/landmark_compare_TM.png
# python compare_lmk3d.py \
#   --file_a ./data/origin_scan_lmks.npy \
#   --file_b ./data/scan_lmks.npy \
#   --label_a Origin \
#   --label_b Test \
#   --output ./data/landmark_compare_TM.png