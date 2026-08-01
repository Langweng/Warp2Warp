from __future__ import annotations

# preprocess/homography_utils.py
import torch
import numpy as np
import kornia.geometry.transform as tgm # 用于单应性矩阵计算
import cv2 # 用于图像透视变换


def is_homography_well_conditioned(
    homography: np.ndarray,
    min_abs_det: float = 1e-8,
    max_condition: float = 1e8,
) -> bool:
    """Return whether a homography is finite, invertible, and not extremely ill-conditioned."""
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return False

    det = np.linalg.det(matrix)
    if not np.isfinite(det) or abs(det) < min_abs_det:
        return False

    condition = np.linalg.cond(matrix)
    return bool(np.isfinite(condition) and condition <= max_condition)


def safe_inverse_homography(
    homography: np.ndarray,
    context: str = "homography",
    min_abs_det: float = 1e-8,
    max_condition: float = 1e8,
) -> np.ndarray:
    """Invert a homography after checking for the degenerate matrices that stop long runs."""
    matrix = np.asarray(homography, dtype=np.float64)
    if not is_homography_well_conditioned(matrix, min_abs_det=min_abs_det, max_condition=max_condition):
        det = np.linalg.det(matrix) if matrix.shape == (3, 3) and np.isfinite(matrix).all() else float("nan")
        condition = np.linalg.cond(matrix) if matrix.shape == (3, 3) and np.isfinite(matrix).all() else float("nan")
        raise ValueError(
            f"Cannot invert {context}: det={det:.6e}, condition={condition:.6e}, matrix={matrix.tolist()}"
        )

    inverse = np.linalg.inv(matrix)
    if abs(inverse[2, 2]) > min_abs_det:
        inverse = inverse / inverse[2, 2]
    return inverse.astype(np.float32)


def resolve_reflect_padding(config: dict, crop_h: int, crop_w: int, default_pad: int = 32) -> int:
    """
    为小分辨率裁剪场景返回镜像 padding 大小。

    默认规则：
    - 当 crop size 为 128x128 时，自动启用 32 像素 reflect padding。
    - 也支持通过 data.reflect_padding.enabled / size 显式覆盖。
    """
    reflect_cfg = config.get('data', {}).get('reflect_padding')
    auto_pad = default_pad if crop_h == 128 and crop_w == 128 else 0

    if reflect_cfg is None:
        return auto_pad

    if not bool(reflect_cfg.get('enabled', False)):
        return 0

    return int(reflect_cfg.get('size', auto_pad or default_pad))


def warp_perspective_with_reflect_padding(
    img_np: np.ndarray,
    homography: np.ndarray,
    output_hw: tuple[int, int],
    reflect_pad: int = 0,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    对输入图像先做 reflect padding，再执行透视变换，并裁回原始输出尺寸。

    这样在 128x128 crop 场景下，warp 后中心 crop 更不容易出现黑边。
    """
    out_h, out_w = int(output_hw[0]), int(output_hw[1])
    homography = np.asarray(homography, dtype=np.float32)

    if reflect_pad <= 0:
        return cv2.warpPerspective(img_np, homography, (out_w, out_h), flags=interpolation)

    reflect_pad = int(reflect_pad)
    padded_img = cv2.copyMakeBorder(
        img_np,
        reflect_pad,
        reflect_pad,
        reflect_pad,
        reflect_pad,
        borderType=cv2.BORDER_REFLECT_101,
    )

    translation = np.array(
        [[1.0, 0.0, reflect_pad], [0.0, 1.0, reflect_pad], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    translation_inv = np.array(
        [[1.0, 0.0, -reflect_pad], [0.0, 1.0, -reflect_pad], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    homography_padded = translation @ homography @ translation_inv

    warped_padded = cv2.warpPerspective(
        padded_img,
        homography_padded,
        (out_w + 2 * reflect_pad, out_h + 2 * reflect_pad),
        flags=interpolation,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return warped_padded[reflect_pad:reflect_pad + out_h, reflect_pad:reflect_pad + out_w]


def _roi_corners(H: int, W: int, marginal: int) -> np.ndarray:
    return np.array([
        [marginal, marginal],
        [W - 1 - marginal, marginal],
        [marginal, H - 1 - marginal],
        [W - 1 - marginal, H - 1 - marginal],
    ], dtype=np.float32)


def _coord_scale(src_size: int, dst_size: int, scale_mode: str) -> float:
    if scale_mode == 'align_corners':
        if src_size <= 1 or dst_size <= 1:
            raise ValueError(f"align_corners scaling requires sizes > 1, got {src_size}, {dst_size}")
        return float(dst_size - 1) / float(src_size - 1)
    if scale_mode == 'size':
        if src_size <= 0 or dst_size <= 0:
            raise ValueError(f"size scaling requires sizes > 0, got {src_size}, {dst_size}")
        return float(dst_size) / float(src_size)
    raise ValueError(f"Unsupported scale_mode {scale_mode!r}. Expected 'align_corners' or 'size'.")


def resolve_homography_adaptation(config: dict) -> dict | None:
    adaptation_cfg = config.get('data', {}).get('homography_adaptation')
    if not adaptation_cfg or not bool(adaptation_cfg.get('enabled', False)):
        return None

    source_image_size = adaptation_cfg.get('source_image_size')
    if source_image_size is None or len(source_image_size) != 2:
        raise ValueError(
            "data.homography_adaptation.source_image_size must be [H, W] when adaptation is enabled."
        )

    source_h = int(source_image_size[0])
    source_w = int(source_image_size[1])
    source_marginal = int(adaptation_cfg.get('source_marginal', config['data']['marginal']))
    scale_mode = str(adaptation_cfg.get('scale_mode', 'align_corners')).lower()
    if scale_mode not in {'align_corners', 'size'}:
        raise ValueError(
            f"Unsupported data.homography_adaptation.scale_mode={scale_mode!r}. "
            "Expected 'align_corners' or 'size'."
        )

    source_crop_h = source_h - 2 * source_marginal
    source_crop_w = source_w - 2 * source_marginal
    if source_crop_h <= 0 or source_crop_w <= 0:
        raise ValueError(
            "data.homography_adaptation defines a non-positive source crop size: "
            f"image_size=({source_h}, {source_w}), marginal={source_marginal}."
        )

    return {
        'source_h': source_h,
        'source_w': source_w,
        'source_marginal': source_marginal,
        'source_crop_h': source_crop_h,
        'source_crop_w': source_crop_w,
        'scale_mode': scale_mode,
    }


def map_points_between_rois(
    points: np.ndarray,
    source_hw: tuple[int, int],
    source_marginal: int,
    target_hw: tuple[int, int],
    target_marginal: int,
    scale_mode: str = 'align_corners',
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    source_h, source_w = int(source_hw[0]), int(source_hw[1])
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    source_crop_h = source_h - 2 * source_marginal
    source_crop_w = source_w - 2 * source_marginal
    target_crop_h = target_h - 2 * target_marginal
    target_crop_w = target_w - 2 * target_marginal

    scale_x = _coord_scale(source_crop_w, target_crop_w, scale_mode)
    scale_y = _coord_scale(source_crop_h, target_crop_h, scale_mode)
    offset = np.array([source_marginal, source_marginal], dtype=np.float32)
    target_offset = np.array([target_marginal, target_marginal], dtype=np.float32)
    mapped = points - offset
    mapped[:, 0] = mapped[:, 0] * scale_x
    mapped[:, 1] = mapped[:, 1] * scale_y
    return mapped + target_offset


def scale_delta_between_rois(
    delta: np.ndarray,
    source_crop_hw: tuple[int, int],
    target_crop_hw: tuple[int, int],
    scale_mode: str = 'align_corners',
) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float32).copy()
    source_crop_h, source_crop_w = int(source_crop_hw[0]), int(source_crop_hw[1])
    target_crop_h, target_crop_w = int(target_crop_hw[0]), int(target_crop_hw[1])
    scale_x = _coord_scale(source_crop_w, target_crop_w, scale_mode)
    scale_y = _coord_scale(source_crop_h, target_crop_h, scale_mode)
    delta[:, 0] *= scale_x
    delta[:, 1] *= scale_y
    return delta


def corners_to_H(d_hat: np.ndarray, marginal: int, H: int, W: int) -> np.ndarray:
    """
    将预测位移 d_hat 转换为全局单应性矩阵 H (3x3)。
    
    Args:
        d_hat (np.ndarray): 预测的位移，形状为 (4, 2)。
        marginal (int): 边缘间距。
        H (int): 图像高度。
        W (int): 图像宽度。
        
    Returns:
        np.ndarray: 全局单应性矩阵 H (3, 3)。
    """
    # 1. 确定 ROI 源角点 p_src (在全局坐标系)
    p_src = _roi_corners(H, W, marginal)

    # 2. 目标角点 p_tar
    p_tar = p_src + d_hat

    # 3. 计算全局单应性矩阵 H_global
    H_global = tgm.get_perspective_transform(torch.tensor(p_src).unsqueeze(0), torch.tensor(p_tar).unsqueeze(0))
    
    return H_global.squeeze().numpy()


def delta_to_homography_matrices(
    delta_p: np.ndarray,
    H: int,
    W: int,
    marginal: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    根据四角点位移恢复局部/全局单应性矩阵。

    Args:
        delta_p: 形状为 (4, 2) 的角点位移，定义在当前 ROI 坐标系对应的全局坐标下。
        H: 全图高度。
        W: 全图宽度。
        marginal: ROI 边距。

    Returns:
        tuple[np.ndarray, np.ndarray]: (H_local, H_global)
    """
    delta_p = np.asarray(delta_p, dtype=np.float32)
    h, w = H - 2 * marginal, W - 2 * marginal

    p_src = _roi_corners(H, W, marginal)
    p_tar = p_src + delta_p
    H_global = tgm.get_perspective_transform(
        torch.tensor(p_src).unsqueeze(0),
        torch.tensor(p_tar).unsqueeze(0),
    )

    p_src_local = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=np.float32)
    p_tar_local = p_tar - np.array([marginal, marginal], dtype=np.float32)
    H_local = tgm.get_perspective_transform(
        torch.tensor(p_src_local).unsqueeze(0),
        torch.tensor(p_tar_local).unsqueeze(0),
    )

    return H_local.squeeze().numpy(), H_global.squeeze().numpy()


def get_homography(H: int, W: int, marginal: int, perturb_range: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成一个随机的单应性矩阵，及其对应的全局和局部角点。
    
    Args:
        H (int): 原始图像高度。
        W (int): 原始图像宽度。
        marginal (int): 边缘间距。
        perturb_range (int): 角点扰动范围 U[-r, r]。
        
    Returns:
        tuple: (H_local, H_global, delta_p)
    """
    return get_homography_with_adaptation(H, W, marginal, perturb_range, adaptation=None)


def get_homography_with_adaptation(
    H: int,
    W: int,
    marginal: int,
    perturb_range: int,
    adaptation: dict | None = None,
    max_attempts: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成随机单应性。若 adaptation 不为空，则先在 source ROI 坐标系采样角点扰动，
    再严格映射到当前 ROI 坐标系。
    """
    p_src = _roi_corners(H, W, marginal)

    last_delta_p = None
    last_H_global = None
    last_error = None
    for _ in range(max(1, int(max_attempts))):
        delta_x = np.random.uniform(-perturb_range, perturb_range, size=(4, 1)).astype(np.int16)
        delta_y = np.random.uniform(-perturb_range, perturb_range, size=(4, 1)).astype(np.int16)
        delta_source = np.hstack((delta_x, delta_y)).astype(np.float32)

        if adaptation is None:
            delta_p = delta_source
        else:
            source_h = int(adaptation['source_h'])
            source_w = int(adaptation['source_w'])
            source_marginal = int(adaptation['source_marginal'])
            scale_mode = adaptation['scale_mode']

            p_src_source = _roi_corners(source_h, source_w, source_marginal)
            p_tar_source = p_src_source + delta_source
            p_tar = map_points_between_rois(
                p_tar_source,
                source_hw=(source_h, source_w),
                source_marginal=source_marginal,
                target_hw=(H, W),
                target_marginal=marginal,
                scale_mode=scale_mode,
            )
            delta_p = p_tar - p_src

        last_delta_p = delta_p
        try:
            H_local, H_global = delta_to_homography_matrices(delta_p, H, W, marginal)
        except RuntimeError as exc:
            # Kornia raises when the sampled corner configuration is singular.
            # Treat it like any other invalid random homography and resample.
            last_error = exc
            last_H_global = None
            continue

        last_H_global = H_global
        if is_homography_well_conditioned(H_global) and is_homography_well_conditioned(H_local):
            return H_local.astype(np.float32), H_global.astype(np.float32), delta_p.astype(np.float32)

    raise ValueError(
        "Failed to sample a valid homography after "
        f"{max_attempts} attempts; last delta={None if last_delta_p is None else last_delta_p.tolist()}, "
        f"last global H={None if last_H_global is None else last_H_global.tolist()}, "
        f"last error={None if last_error is None else str(last_error)}"
    )
