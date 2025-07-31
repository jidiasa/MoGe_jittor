import jittor as jt
from utils.geometry_jt import (weighted_mean, harmonic_mean, geometric_mean, 
normalized_view_plane_uv, gaussian_blur_2d, focal_to_fov, 
fov_to_focal, intrinsics_to_fov, point_map_to_depth_legacy, recover_focal_shift, view_plane_uv_to_focal, mask_aware_nearest_resize )
import numpy as np

def test_weighted_mean():
    x = jt.array([[1.0, 2.0], [3.0, 4.0]])
    w = jt.array([[1.0, 0.0], [0.0, 1.0]])

    mean1 = weighted_mean(x, dim=1)
    print("Unweighted mean:", mean1.numpy())

    mean2 = weighted_mean(x, w=w, dim=1)
    print("Weighted mean:", mean2.numpy())

def test_harmonic_mean():
    x = jt.array([[1.0, 2.0], [3.0, 4.0]])
    w = jt.array([[1.0, 0.0], [0.0, 1.0]])

    hmean1 = harmonic_mean(x, dim=1)
    print("Unweighted harmonic mean:", hmean1.numpy())

    hmean2 = harmonic_mean(x, w=w, dim=1)
    print("Weighted harmonic mean:", hmean2.numpy())


def test_geometric_mean():
    x = jt.array([[1.0, 2.0], [3.0, 4.0]])
    w = jt.array([[1.0, 0.0], [0.0, 1.0]])

    gmean1 = geometric_mean(x, dim=1)
    print("Unweighted geometric mean:", gmean1.numpy())

    gmean2 = geometric_mean(x, w=w, dim=1)
    print("Weighted geometric mean:", gmean2.numpy())

def test_normalized_view_plane_uv():
    uv = normalized_view_plane_uv(width=4, height=3)
    print("Normalized UV shape:", uv.shape)
    print("Sample UV[0,0]:", uv[0,0].numpy())
    print("Sample UV[-1,-1]:", uv[-1,-1].numpy())

def test_gaussian_blur():
    image = jt.ones((1, 1, 5, 5))  # 5x5 全为 1
    blurred = gaussian_blur_2d(image, kernel_size=3, sigma=1.0)
    print("Blurred image shape:", blurred.shape)
    print("Blurred image:", blurred[0, 0].numpy())

def test_focal_fov():
    focal = jt.array([1.0, 2.0])
    fov = focal_to_fov(focal)
    recovered_focal = fov_to_focal(fov)
    print("Original focal:", focal.numpy())
    print("FOV:", fov.numpy())
    print("Recovered focal:", recovered_focal.numpy())

def test_intrinsics_to_fov():
    intrinsics = jt.array([
        [[1.0, 0.0, 0.5],
         [0.0, 1.0, 0.5],
         [0.0, 0.0, 1.0]],
        [[2.0, 0.0, 0.5],
         [0.0, 4.0, 0.5],
         [0.0, 0.0, 1.0]]
    ])  # shape: (2, 3, 3)

    fov_x, fov_y = intrinsics_to_fov(intrinsics)
    print("FOV X:", fov_x.numpy())
    print("FOV Y:", fov_y.numpy())

def test_point_map_to_depth_legacy():

    H, W = 2, 2
    uv = normalized_view_plane_uv(W, H)  # (H, W, 2)
    depth_val = 1.0
    points = jt.concat([uv, jt.ones((H, W, 1)) * depth_val], dim=-1)  # (H, W, 3)

    depth, fov_x, fov_y, shift = point_map_to_depth_legacy(points)

    print("Estimated Depth:\n", depth.numpy())
    print("Estimated FoV X:", fov_x.numpy())
    print("Estimated FoV Y:", fov_y.numpy())
    print("Estimated Shift:", shift.numpy())

def test_view_plane_uv_to_focal():
    uv = normalized_view_plane_uv(width=4, height=3)
    focal = view_plane_uv_to_focal(uv)
    print("Estimated focal from UV:", focal.numpy())

def test_recover_focal_shift():
    H, W = 4, 4
    uv = normalized_view_plane_uv(W, H)
    points = jt.concat([uv, jt.ones((H, W, 1))], dim=-1)
    focal, shift = recover_focal_shift(points)
    print("Recovered focal:", focal.numpy())
    print("Recovered shift:", shift.numpy())

def test_mask_aware_nearest_resize():
    print("=== Test: mask_aware_nearest_resize ===")

    mask_np = np.array([
        [0, 0, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0]
    ], dtype=bool)

    mask = jt.array(mask_np)

    tgt_w, tgt_h = 3, 3

    (i_idx, j_idx), tgt_mask = mask_aware_nearest_resize(mask, tgt_w, tgt_h)
    print("Nearest I indices:\n", i_idx.numpy())
    print("Nearest J indices:\n", j_idx.numpy())
    print("Target mask:\n", tgt_mask.numpy().astype(np.uint8))  

    print("\nTarget mask ASCII:")
    ascii_map = np.where(tgt_mask.numpy(), '●', '·')
    for row in ascii_map:
        print(' '.join(row))


if __name__ == "__main__":
    test_weighted_mean()
    test_harmonic_mean()
    test_geometric_mean()
    test_normalized_view_plane_uv()
    test_gaussian_blur()
    test_focal_fov()
    test_intrinsics_to_fov()
    test_point_map_to_depth_legacy()
    test_recover_focal_shift()
    test_view_plane_uv_to_focal()
    test_mask_aware_nearest_resize()