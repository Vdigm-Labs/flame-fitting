import numpy as np
import matplotlib.pyplot as plt

# Load files
gt = np.load("scan_lmks.npy")         # GT landmarks (51, 3)
pred = np.load("my_scan_lmks.npy")    # Predicted landmarks (105, 3)

# Print landmark coordinates
print("🔵 GT landmarks (51):")
for i, pt in enumerate(gt):
    print(f"GT[{i:02}] = {pt}")

print("\n🔴 Predicted landmarks (105):")
for i, pt in enumerate(pred):
    print(f"Pred[{i:02}] = {pt}")

# Plot
fig = plt.figure(figsize=(12, 5))

# 3D Scatter Plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title("GT (blue, 51) vs Pred (red, 105) - 3D")
ax1.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c='blue', label='GT (51)')
ax1.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='red', marker='^', label='Pred (105)')
ax1.legend()

# XY Plane Projection
ax2 = fig.add_subplot(122)
ax2.set_title("XY Projection (2D View)")
ax2.scatter(gt[:, 0], gt[:, 1], c='blue', label='GT (51)')
ax2.scatter(pred[:, 0], pred[:, 1], c='red', marker='^', label='Pred (105)')
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.axis('equal')
ax2.legend()

# Add text labels to GT landmarks
for i, (x, y) in enumerate(gt[:, :2]):
    ax2.text(x, y, str(i), fontsize=6, color='black')

# Add text labels to Pred landmarks
for i, (x, y) in enumerate(pred[:, :2]):
    ax2.text(x, y, str(i), fontsize=6, color='black')

plt.tight_layout()
plt.savefig("landmark_comparison_clean.png")
print("\n📸 Image saved to landmark_comparison_clean.png")
