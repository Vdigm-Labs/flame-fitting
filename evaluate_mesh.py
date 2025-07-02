import numpy as np
import trimesh


def compute_mple(pred_landmarks, gt_landmarks):
    """
    Mean Per-Landmark Error (MPLE)
    :param pred_landmarks: (N, 3)
    :param gt_landmarks: (N, 3)
    :return: scalar mean error
    """
    errors = np.linalg.norm(pred_landmarks - gt_landmarks, axis=-1)  # (N,)
    mple = errors.mean()
    return mple


def compute_nle(pred_landmarks, gt_landmarks, normalization_pair):
    """
    Normalized Landmark Error (NLE)
    :param pred_landmarks: (N, 3)
    :param gt_landmarks: (N, 3)
    :param normalization_pair: tuple of 2 indices (e.g. (36, 45))
    :return: scalar normalized error
    """
    mple = compute_mple(pred_landmarks, gt_landmarks)
    inter_ocular_dist = np.linalg.norm(
        gt_landmarks[normalization_pair[0]] - gt_landmarks[normalization_pair[1]]
    )
    nle = mple / (inter_ocular_dist + 1e-8)  # avoid division by zero
    return nle


if __name__ == "__main__":
    # === Load predicted and ground-truth .obj files ===
    pred_mesh = trimesh.load("./data/fit_lmk3d_result_jinju.obj", process=False)
    gt_mesh = trimesh.load("./data/fit_lmk3d_result_madong.obj", process=False)

    # === Extract vertices ===
    pred_vertices = pred_mesh.vertices  # (N, 3)
    gt_vertices = gt_mesh.vertices      # (N, 3)

    # === Check same number of vertices ===
    assert pred_vertices.shape == gt_vertices.shape, "Meshes have different number of vertices."

    # === Example: eye corner indices for normalization ===
    left_eye_idx = 36
    right_eye_idx = 45

    mple = compute_mple(pred_vertices, gt_vertices)
    nle = compute_nle(pred_vertices, gt_vertices, normalization_pair=(left_eye_idx, right_eye_idx))

    print(f"MPLE: {mple:.4f} meters")
    print(f"NLE : {nle:.4f}")
