import numpy as np


def compute_mple(pred_landmarks, gt_landmarks):
    """
    Mean Per-Landmark Error (MPLE)
    :param pred_landmarks: (B, N, 3)
    :param gt_landmarks: (B, N, 3)
    :return: (B,) mean error per sample
    """
    errors = np.linalg.norm(pred_landmarks - gt_landmarks, axis=-1)  # (B, N)
    mple = errors.mean(axis=1)  # (B,)
    return mple


def compute_nle(pred_landmarks, gt_landmarks, normalization_pairs):
    """
    Normalized Landmark Error (NLE)
    :param pred_landmarks: (B, N, 3)
    :param gt_landmarks: (B, N, 3)
    :param normalization_pairs: list of 2 indices (e.g. [left_eye_idx, right_eye_idx])
    :return: (B,) normalized error per sample
    """
    mple = compute_mple(pred_landmarks, gt_landmarks)  # (B,)
    inter_ocular_dists = np.linalg.norm(
        gt_landmarks[:, normalization_pairs[0]] - gt_landmarks[:, normalization_pairs[1]], axis=1
    )  # (B,)
    nle = mple / (inter_ocular_dists + 1e-8)  # avoid division by zero
    return nle


def compute_parameter_norms(pose_params, shape_params):
    """
    Compute L2 norms of pose and shape parameters
    :param pose_params: (B, P)
    :param shape_params: (B, S)
    :return: dict with pose and shape norms
    """
    pose_norms = np.linalg.norm(pose_params, axis=1)  # (B,)
    shape_norms = np.linalg.norm(shape_params, axis=1)  # (B,)
    return {
        'pose_norm_mean': pose_norms.mean(),
        'pose_norm_std': pose_norms.std(),
        'shape_norm_mean': shape_norms.mean(),
        'shape_norm_std': shape_norms.std()
    }


def summarize_metrics(mple, nle):
    """
    Print MPLE and NLE with mean ± std
    """
    print(f"MPLE: {mple.mean():.4f} ± {mple.std():.4f}")
    print(f"NLE : {nle.mean():.4f} ± {nle.std():.4f}")


# === Example usage ===
if __name__ == "__main__":
    B, N = 16, 68  # batch size, number of landmarks
    pred = np.random.rand(B, N, 3)
    gt = np.random.rand(B, N, 3)

    pose = np.random.randn(B, 72)
    shape = np.random.randn(B, 10)

    mple = compute_mple(pred, gt)
    nle = compute_nle(pred, gt, normalization_pairs=[36, 45])  # eye corner indices

    norms = compute_parameter_norms(pose, shape)

    summarize_metrics(mple, nle)
    print(norms)

