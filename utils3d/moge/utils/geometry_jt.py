# geometry_jittor.py
from typing import Union
import jittor as jt
import math
from typing import Tuple, Optional
import numpy as np
from .geometry_numpy import solve_optimal_focal_shift, solve_optimal_shift
from utils3d.numpy import image_pixel_center, sliding_window_2d, image_uv


def weighted_mean(x: jt.Var, w: jt.Var = None, dim: Union[int, tuple] = None, keepdim: bool = False, eps: float = 1e-7) -> jt.Var: #debug:right
    if w is None:
        return jt.mean(x, dim=dim, keepdims=keepdim)
    else:
        w = w.cast(x.dtype)
        return jt.mean(x * w, dim=dim, keepdims=keepdim) / (jt.mean(w, dim=dim, keepdims=keepdim) + eps)

def harmonic_mean(x: jt.Var, w: jt.Var = None, dim: Union[int, tuple] = None, keepdim: bool = False, eps: float = 1e-7) -> jt.Var:
    if w is None:
         return 1.0 / jt.mean(1.0 / (x + eps), dim=dim, keepdims=keepdim)
    else:
        w = w.cast(x.dtype)
        return 1.0 / weighted_mean(1.0 / (x + eps), w, dim=dim, keepdim=keepdim, eps=eps)
    
def geometric_mean(x: jt.Var, w: jt.Var = None, dim: Union[int, tuple] = None, keepdim: bool = False, eps: float = 1e-7) -> jt.Var:
    if w is None:
        return jt.exp(jt.mean(jt.log(x + eps), dim=dim, keepdims=keepdim))
    else:
        w = w.cast(x.dtype)
        return jt.exp(weighted_mean(jt.log(x + eps), w, dim=dim, keepdim=keepdim, eps=eps))

def normalized_view_plane_uv(width: int, height: int, aspect_ratio: float = None, dtype: str = "float32", device=None) -> jt.Var:
    # UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom as (width / diagonal, height / diagonal)
    if aspect_ratio is None:
        aspect_ratio = width / height

    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1.0 / (1 + aspect_ratio ** 2) ** 0.5

    u = jt.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width).cast(dtype)
    v = jt.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height).cast(dtype)
    u, v = jt.meshgrid(u, v)  # (W, H)

    u = u.transpose(0, 1)  # -> (H, W)
    v = v.transpose(0, 1)

    uv = jt.stack([u, v], dim=-1)  # (H, W, 2)
    return uv

def gaussian_blur_2d(input: jt.Var, kernel_size: int, sigma: float) -> jt.Var:
    # 1D Gaussian kernel
    half_k = kernel_size // 2
    x = jt.arange(-half_k + 1, half_k + 1).cast(input.dtype)
    kernel_1d = jt.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # 2D Gaussian kernel
    kernel_2d = jt.matmul(kernel_1d.unsqueeze(1), kernel_1d.unsqueeze(0))  # (K, K)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # -> (1, 1, K, K)


    # padding
    pad = kernel_size // 2
    jt.nn.pad(input, [pad, pad, pad, pad], mode='replicate')


    # 2D convolution
    out = jt.nn.conv2d(input, kernel_2d, groups=input.shape[1])
    return out

def focal_to_fov(focal: jt.Var) -> jt.Var:
    return 2 * jt.atan(0.5 / focal)

def fov_to_focal(fov: jt.Var) -> jt.Var:
    return 0.5 / jt.tan(fov / 2)

def intrinsics_to_fov(intrinsics: jt.Var) -> Tuple[jt.Var, jt.Var]:

    focal_x = intrinsics[..., 0, 0]
    focal_y = intrinsics[..., 1, 1]
    fov_x = 2 * jt.atan(0.5 / focal_x)
    fov_y = 2 * jt.atan(0.5 / focal_y)
    return fov_x, fov_y

def point_map_to_depth_legacy(points: jt.Var, eps: float = 1e-6) -> Tuple[jt.Var, jt.Var, jt.Var, jt.Var]:
 
    height, width = points.shape[-3:-1]
    diagonal = (height ** 2 + width ** 2) ** 0.5

    uv = normalized_view_plane_uv(width, height, dtype=points.dtype)  # (H, W, 2)
    uv = uv.broadcast(points[..., 0:2].shape)  # (..., H, W, 2)

    # Compute A and b
    b = (uv * points[..., 2:3]).reshape(*points.shape[:-1], -1)  # (..., H*W*2)
    A = jt.stack([
        points[..., :2],              # (..., H, W, 2)
        -uv                           # (..., H, W, 2)
    ], dim=-1)                        # (..., H, W, 2, 2)
    A = A.reshape(*points.shape[:-1], -1, 2)  # (..., H*W*2, 2)

    # Compute least-squares solution
    A_T = A.transpose(-2, -1)                         # (..., 2, H*W*2)
    M = A_T @ A                                       # (..., 2, 2)
    batch_shape = M.shape[:-2]
    I = jt.nn.init.eye(2).unsqueeze(0).repeat(*batch_shape, 1, 1).cast(M.dtype)
   # (..., 2, 2)
    M_inv = jt.linalg.inv(M + eps * I)                # (..., 2, 2)

    solution = (M_inv @ (A_T @ b.unsqueeze(-1))).squeeze(-1)  # (..., 2)
    focal, shift = solution[..., 0], solution[..., 1]

    # Compute depth map and FoV
    depth = points[..., 2] + shift.unsqueeze(-1).unsqueeze(-1)
    fov_x = 2 * jt.atan(width / diagonal / focal)
    fov_y = 2 * jt.atan(height / diagonal / focal)

    return depth, fov_x, fov_y, shift

def view_plane_uv_to_focal(uv: jt.Var) -> jt.Var:
    normed_uv = normalized_view_plane_uv(width=uv.shape[-2], height=uv.shape[-3], dtype=uv.dtype)
    numerator = (uv * normed_uv).sum()
    denominator = (uv ** 2).sum() + 1e-12
    focal = numerator / denominator
    return focal

def recover_focal_shift(points: jt.Var, mask: Optional[jt.Var] = None, focal: Optional[jt.Var] = None,
                        downsample_size: Tuple[int, int] = (64, 64)) -> Tuple[jt.Var, jt.Var]:
    """
    Recover focal length and shift from a 3D point map (H, W, 3)
    """
    shape = points.shape
    H, W = shape[-3], shape[-2]
    D = (H**2 + W**2)**0.5

    points = points.reshape(-1, H, W, 3)
    mask = None if mask is None else mask.reshape(-1, H, W)
    focal = focal.reshape(-1) if focal is not None else None

    uv = normalized_view_plane_uv(W, H, dtype=points.dtype)

    # Downsample
    points_lr = jt.nn.interpolate(points.permute(0, 3, 1, 2), size=downsample_size, mode='nearest').permute(0, 2, 3, 1)
    uv_lr = jt.nn.interpolate(uv.permute(2, 0, 1).unsqueeze(0), size=downsample_size, mode='nearest').squeeze(0).permute(1, 2, 0)
    mask_lr = None if mask is None else jt.nn.interpolate(mask.float32().unsqueeze(1), size=downsample_size, mode='nearest').squeeze(1) > 0

    points_lr_np = points_lr.numpy()
    uv_lr_np = uv_lr.numpy()
    mask_lr_np = None if mask_lr is None else mask_lr.numpy()
    focal_np = focal.numpy() if focal is not None else None

    optim_focal = []
    optim_shift = []

    for i in range(points.shape[0]):
        p_i = points_lr_np[i]
        uv_i = uv_lr_np
        if mask is not None:
            m_i = mask_lr_np[i]
            p_i = p_i[m_i]
            uv_i = uv_i[m_i]

        if focal is None:
            shift_i, focal_i = solve_optimal_focal_shift(uv_i, p_i)
            optim_shift.append(shift_i)
            optim_focal.append(focal_i)
        else:
            shift_i = solve_optimal_shift(uv_i, p_i, focal_np[i])
            optim_shift.append(shift_i)
            optim_focal.append(focal_np[i])

    optim_focal = jt.array(optim_focal, dtype=points.dtype)
    optim_shift = jt.array(optim_shift, dtype=points.dtype)
    if len(shape) > 3:
        optim_focal = optim_focal.reshape(shape[:-3])
        optim_shift = optim_shift.reshape(shape[:-3])

    return optim_focal, optim_shift

def mask_aware_nearest_resize(mask: jt.Var, target_width: int, target_height: int):
    *batch_shape, height, width = mask.shape
    B = int(np.prod(batch_shape)) if batch_shape else 1
    filter_h_f, filter_w_f = max(1.0, height / target_height), max(1.0, width / target_width)
    filter_h_i, filter_w_i = int(math.ceil(filter_h_f)), int(math.ceil(filter_w_f))
    filter_size = filter_h_i * filter_w_i
    padding_h, padding_w = int(round(filter_h_f / 2.0)), int(round(filter_w_f / 2.0))
    uv_np = image_pixel_center(width, height, dtype=np.float32)
    indices_np = np.arange(height * width, dtype=np.int64).reshape(height, width)
    mask_np = mask.numpy().reshape(B, height, width).astype(np.bool_)
    Hp, Wp = height + 2 * padding_h, width + 2 * padding_w
    padded_uv = np.zeros((Hp, Wp, 2), dtype=np.float32)
    padded_uv[padding_h:padding_h + height, padding_w:padding_w + width] = uv_np
    padded_indices = np.zeros((Hp, Wp), dtype=np.int64)
    padded_indices[padding_h:padding_h + height, padding_w:padding_w + width] = indices_np
    padded_masks = np.zeros((B, Hp, Wp), dtype=np.bool_)
    for b in range(B):
        padded_masks[b, padding_h:padding_h + height, padding_w:padding_w + width] = mask_np[b]
    windowed_uv = sliding_window_2d(padded_uv, (filter_h_i, filter_w_i), 1, axis=(0, 1))
    windowed_indices = sliding_window_2d(padded_indices, (filter_h_i, filter_w_i), 1, axis=(0, 1))
    windowed_mask = np.stack([sliding_window_2d(padded_masks[b], (filter_h_i, filter_w_i), 1, axis=(0, 1)) for b in range(B)], axis=0)
    Ho, Wo = windowed_uv.shape[0], windowed_uv.shape[1]
    target_uv_np = image_uv(target_width, target_height, dtype=np.float32) * np.array([width, height], dtype=np.float32)
    target_corner = np.round(target_uv_np - np.array([filter_w_f / 2.0, filter_h_f / 2.0], dtype=np.float32) - 0.5).astype(np.int64)
    target_corner[..., 0] += padding_w
    target_corner[..., 1] += padding_h
    yy = target_corner[..., 1]
    xx = target_corner[..., 0]
    target_window_uv = windowed_uv[yy, xx].reshape(target_height, target_width, filter_h_i * filter_w_i, 2).transpose(0, 1, 3, 2)
    target_window_indices = windowed_indices[yy, xx].reshape(target_height, target_width, filter_size)
    target_window_mask = np.stack([windowed_mask[b, yy, xx].reshape(target_height, target_width, filter_size) for b in range(B)], axis=0)
    target_window_indices = np.broadcast_to(target_window_indices[None, ...], (B, target_height, target_width, filter_size))
    dist = np.where(target_window_mask, np.linalg.norm(target_window_uv[None, ...] - target_uv_np[..., None], axis=-2), np.inf)
    nearest = np.argmin(dist, axis=-1, keepdims=True)
    nearest_idx = np.take_along_axis(target_window_indices, nearest, axis=-1).squeeze(-1)
    target_mask = np.any(target_window_mask, axis=-1)
    nearest_i = nearest_idx // width
    nearest_j = nearest_idx % width
    final_shape = (*batch_shape, target_height, target_width) if batch_shape else (target_height, target_width)
    nearest_i = jt.array(nearest_i).reshape(final_shape).astype(jt.int64)
    nearest_j = jt.array(nearest_j).reshape(final_shape).astype(jt.int64)
    target_mask = jt.array(target_mask).reshape(final_shape).astype(jt.bool)
    batch_indices = ()
    if batch_shape:
        bis = []
        mdim = len(batch_shape) + 2
        for i, n in enumerate(batch_shape):
            shape = [1] * i + [n] + [1] * (mdim - i - 1)
            bis.append(jt.arange(n).reshape(shape))
        batch_indices = tuple(bis)
    return (*batch_indices, nearest_i, nearest_j), target_mask