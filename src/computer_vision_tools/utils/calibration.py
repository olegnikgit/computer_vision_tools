#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A collection of utility functions for calibration.

@author: ons
"""
# =============================================================================
import cv2
import numpy as np
import os


# =============================================================================
def draw_reprojection(image, charuco_corners, charuco_ids, rvec, tvec, camera_matrix, dist_coeffs, board):
    """
    Draws the reprojection of the ChArUco corners on the image.

    Parameters:
    image: np.ndarray
        The input image (BGR).
    charuco_corners: np.ndarray
        Detected ChArUco corners of shape (N, 1, 2), dtype=float32.
    charuco_ids: np.ndarray
        Detected ChArUco IDs of shape (N, 1), dtype=int32.
    rvec: np.ndarray
        Rotation vector (3x1) from solvePnP.
    tvec: np.ndarray
        Translation vector (3x1) from solvePnP.
    camera_matrix: np.ndarray
        Camera intrinsic matrix (3x3).
    dist_coeffs: np.ndarray
        Distortion coefficients (k1, k2, p1, p2, etc.).
    board: cv2.aruco.CharucoBoard
        The ChArUco board object used for calibration.

    Returns:
    np.ndarray
        The output image with drawn reprojection.
    """
    img = image.copy()

    obj_points, _ = board.matchImagePoints(charuco_corners, charuco_ids)

    projected, _ = cv2.projectPoints(
        obj_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs
    )

    # detected = green
    for c in charuco_corners:
        x, y = int(c[0][0]), int(c[0][1])
        cv2.circle(img, (x, y), 4, (0, 255, 0), -1)

    # projected = red + error lines
    for c, p in zip(charuco_corners, projected):
        x1, y1 = int(c[0][0]), int(c[0][1])
        x2, y2 = int(p[0][0]), int(p[0][1])

        cv2.circle(img, (x2, y2), 3, (0, 0, 255), -1)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

    return img


def draw_aruco_corners(image, corners, ids, color=(0, 255, 0), font_scale=0.5, thickness=1):
    """
    Draw ArUco corners and their IDs on an image.

    Parameters:
    -----------
    image: numpy.ndarray
        Input BGR image.
    corners: list
        Detected ChArUco corners stacked in the list, each element in the list has a size (1, N, 2).
    ids: list
        Corner IDs stacked in the list, each element in the list is an aray with size (1,).
    color: tuple
        Color for drawing (B, G, R).
    font_scale: float
        Font size for text.
    thickness: int
        Thickness for circles and text.

    Returns:
    --------
    annotated: numpy.ndarray
        Annotated image with drawn corners and IDs.
    """
    if corners is None or ids is None or len(corners[0]) == 0:
        return image

    annotated = image.copy()
    ids = np.array(ids).flatten()

    for idx, aruco_corners in enumerate(corners):

        # the dimension of aruco_corners is (1, N, 2)
        # convert to (N, 2):
        aruco_corners = np.squeeze(aruco_corners)

        for aruco_corner in aruco_corners:
            x, y = int(aruco_corner[0]), int(aruco_corner[1])
            cv2.circle(annotated, (x, y), 4, color, -1)
            cv2.putText(annotated, str(ids[idx]), (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    return annotated


def save_corner_heatmap(all_charuco_corners,
                        image_size,
                        save_path,
                        filename="corner_heatmap.png",
                        overlay_image=None):
    """
    Create and save a smooth corner density heatmap.

    Parameters:
    -----------
    all_charuco_corners : list of np.ndarray
        List of detected charuco corners (N,1,2)
    image_size : tuple
        (width, height)
    save_path : str
        Output directory
    filename : str
        Output filename
    overlay_image : np.ndarray or None
        Optional image to overlay heatmap on (BGR)
    """

    w, h = image_size

    # --- adaptive scale (works for any resolution) ---
    scale = max(w, h) / 1000.0   # ~1.0 for 1MP, ~4 for 4K

    sigma_point = 6 * scale      # size of each corner blob
    sigma_smooth = 12 * scale    # global smoothing

    heatmap = np.zeros((h, w), dtype=np.float32)

    # --- helper: gaussian stamp ---
    def add_gaussian(hmap, x, y, sigma):
        size = int(6 * sigma + 1)
        if size % 2 == 0:
            size += 1

        g = cv2.getGaussianKernel(size, sigma)
        kernel = g @ g.T

        x0 = max(0, x - size // 2)
        y0 = max(0, y - size // 2)
        x1 = min(w, x + size // 2 + 1)
        y1 = min(h, y + size // 2 + 1)

        kx0 = max(0, size // 2 - x)
        ky0 = max(0, size // 2 - y)
        kx1 = kx0 + (x1 - x0)
        ky1 = ky0 + (y1 - y0)

        hmap[y0:y1, x0:x1] += kernel[ky0:ky1, kx0:kx1]

    # --- accumulate ---
    for corners in all_charuco_corners:
        for c in corners:
            x, y = int(c[0][0]), int(c[0][1])
            add_gaussian(heatmap, x, y, sigma_point)

    # --- global smoothing ---
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=sigma_smooth, sigmaY=sigma_smooth)

    # --- normalize ---
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)

    # --- colorize ---
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)

    # --- overlay if image provided ---
    if overlay_image is not None:
        overlay = cv2.addWeighted(overlay_image, 0.6, heatmap_color, 0.4, 0)
        output = overlay
    else:
        output = heatmap_color

    # --- save ---
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(os.path.join(save_path, filename), output)


def draw_checkerboard_corners(image, charuco_corners, charuco_ids, color=(0, 255, 0)):
    """
    Draws checkerboard corners and their IDs on the image.

    Parameters:
    -----------
    image: np.ndarray
        The input image (BGR).
    charuco_corners: np.ndarray
        Detected ChArUco corners of shape (N, 1, 2), dtype=float32.
    charuco_ids: np.ndarray
        Detected ChArUco IDs of shape (N, 1), dtype=int32.
    color: tuple
        BGR color for the corner markers.

    Returns:
    --------
    img_out: np.ndarray
        The output image with drawn corners and IDs.
    """

    img_out = image.copy()

    if charuco_corners is None or charuco_ids is None:
        return img_out

    for i in range(len(charuco_corners)):
        corner = tuple(int(v) for v in charuco_corners[i][0])  # (x, y)
        corner_id = int(charuco_ids[i][0])

        # Draw circle at corner
        cv2.circle(img_out, corner, radius=5, color=color, thickness=-1)

        # Put ID text
        cv2.putText(img_out, str(corner_id), (corner[0] + 10, corner[1] - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

    return img_out


def compute_epipolar_error(ptsL, ptsR, F):
    """
    Compute the epipolar error for each pair of corresponding points given the fundamental matrix F.
    The epipolar error is defined as the distance from the right point to the corresponding epipolar line
    defined by the left point and F.

    Parameters:
    - ptsL: List of points in the left image (Nx2 array)
    - ptsR: List of corresponding points in the right image (Nx2 array)
    - F: Fundamental matrix (3x3)

    Returns:
    - errors: Array of epipolar errors for each point pair
    """
    ptsL = np.array(ptsL, dtype=np.float32)
    ptsR = np.array(ptsR, dtype=np.float32)

    linesR = cv2.computeCorrespondEpilines(
        ptsL.reshape(-1,1,2), 1, F
    ).reshape(-1,3)

    errors = []
    for ptR, line in zip(ptsR, linesR):
        a, b, c = line
        x, y = ptR
        err = abs(a*x + b*y + c) / np.sqrt(a*a + b*b)
        errors.append(err)

    return np.array(errors)


def rectify_and_visualize(imgL, imgR, K1, D1, K2, D2, R, T, save_path, n_lines = 20,
                          file_name="rectified_check.png"):
    """
    Rectify stereo images and visualize with horizontal lines.

    Parameters:
    - imgL, imgR: Input left and right images (BGR)
    - K1, D1: Intrinsic matrix and distortion coefficients for left camera
    - K2, D2: Intrinsic matrix and distortion coefficients for right camera
    - R, T: Rotation and translation between cameras
    - save_path: Directory to save the visualization
    - n_lines: Number of horizontal lines to draw for visualization
    - file_name: Name of the output image file
    """
    h, w = imgL.shape[:2]

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, (w, h), R, T, alpha=0
    )

    map1L, map2L = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1)
    map1R, map2R = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_32FC1)

    rectL = cv2.remap(imgL, map1L, map2L, cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, map1R, map2R, cv2.INTER_LINEAR)

    # Draw horizontal lines
    vis = np.hstack((rectL, rectR))

    # make n_lines lines in the image:
    step = max(1, h // n_lines)

    for y in range(0, h, step):
        # make the line green and 2 pixels thick
        cv2.line(vis, (0, y), (2*w, y), (0,255,0), 2)

    cv2.imwrite(os.path.join(save_path, file_name), vis)

