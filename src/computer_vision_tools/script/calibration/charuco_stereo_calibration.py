#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrate the stereo camera using a ChArUco board.
This script detects the ChArUco corners in the left and right images, matches them,
and performs stereo calibration to compute the rotation and translation between the two cameras.

An example usage:
python charuco_stereo_calibration.py \
    --left_images_path /path/to/left/images \
    --right_images_path /path/to/right/images \
    --left_calibration /path/to/left/calibration.npz \
    --right_calibration /path/to/right/calibration.npz \
    --save_path /path/to/save/stereo/calibration \
    --calibration_filename stereo_calibration.npz \
    --charuco_squares_x 45 \
    --charuco_squares_y 15 \
    --square_length 0.012125 \
    --marker_length 0.007275 \
    --aruco_dict DICT_5X5_1000

Another board example:



@author: ons
"""
# =============================================================================
import cv2
import cv2.aruco as aruco
import numpy as np
import glob
import os
import argparse
from tqdm import tqdm
from csem_vision_tools.utils.io_utils import load_calibration
from csem_vision_tools.utils.calibration import compute_epipolar_error
from csem_vision_tools.utils.calibration import rectify_and_visualize


# =============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Stereo calibration using ChArUco board.")

    parser.add_argument("--left_images_path", type=str, required=True)
    parser.add_argument("--right_images_path", type=str, required=True)

    parser.add_argument("--left_calibration", type=str, required=True)
    parser.add_argument("--right_calibration", type=str, required=True)

    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--calibration_filename", type=str, default="stereo_calibration.npz")

    parser.add_argument("--charuco_squares_x", type=int, default=45)
    parser.add_argument("--charuco_squares_y", type=int, default=15)
    parser.add_argument("--square_length", type=float, default=0.012125)
    parser.add_argument("--marker_length", type=float, default=0.007275)
    parser.add_argument("--aruco_dict", type=str, default="DICT_5X5_1000")

    return parser.parse_args()


# =============================================================================
def evaluate_epipolar_geometry(pairs, charuco_detector, board, F, K1, D1, K2, D2):
    """
    Evaluate the epipolar geometry by computing the epipolar error for all matched
    points across the stereo pairs.

    Parameters:
    - pairs: List of tuples (left_image_path, right_image_path)
    - charuco_detector: An instance of cv2.aruco.CharucoDetector for detecting corners
    - board: The ChArUco board object used for detection
    - F: The fundamental matrix obtained from stereo calibration
    - K1, D1: Intrinsic parameters of the left camera
    - K2, D2: Intrinsic parameters of the right camera

    This function will print the mean epipolar error for each pair and a summary of the overall error statistics.
    """

    all_errors = []

    print("\nEvaluating epipolar geometry...\n")

    for left_path, right_path in pairs:

        imgL = cv2.imread(left_path)
        imgR = cv2.imread(right_path)

        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # Detect board
        _, _, cornersL, idsL = charuco_detector.detectBoard(grayL)
        _, _, cornersR, idsR = charuco_detector.detectBoard(grayR)

        if idsL is None or idsR is None:
            print(f"{os.path.basename(left_path)}: No corners detected in one of the images, skipping.")
            continue

        retL, charuco_cornersL, charuco_idsL = cv2.aruco.interpolateCornersCharuco(
            cornersL, idsL, grayL, board
        )
        retR, charuco_cornersR, charuco_idsR = cv2.aruco.interpolateCornersCharuco(
            cornersR, idsR, grayR, board
        )

        if (
            retL is None or retR is None or
            charuco_idsL is None or charuco_idsR is None
        ):
            print (f"{os.path.basename(left_path)}: Failed to refine corners, skipping.")
            continue

        idsL = charuco_idsL.flatten()
        idsR = charuco_idsR.flatten()

        common_ids = np.intersect1d(idsL, idsR)

        if len(common_ids) < 15:
            print(f"{os.path.basename(left_path)}: Not enough common corners ({len(common_ids)}), skipping.")
            continue

        ptsL = []
        ptsR = []

        for cid in common_ids:
            idxL = np.where(idsL == cid)[0][0]
            idxR = np.where(idsR == cid)[0][0]

            ptsL.append(charuco_cornersL[idxL][0])
            ptsR.append(charuco_cornersR[idxR][0])

        # Inside your loop, after collecting ptsL and ptsR:
        ptsL = np.array(ptsL).reshape(-1, 1, 2)
        ptsR = np.array(ptsR).reshape(-1, 1, 2)

        # Map distorted points to ideal pinhole "pixel" coordinates
        # Passing P=K1 ensures the output is still in pixel units (not normalized space)
        ptsL_undist = cv2.undistortPoints(ptsL, K1, D1, P=K1)
        ptsR_undist = cv2.undistortPoints(ptsR, K2, D2, P=K2)

        # Now evaluate using the undistorted points
        errors = compute_epipolar_error(ptsL_undist.reshape(-1, 2),
                                        ptsR_undist.reshape(-1, 2), F)

        all_errors.extend(errors)

        print(f"{os.path.basename(left_path)} → mean error: {np.mean(errors):.3f} px")
        print(f"{os.path.basename(left_path)} → median error: {np.median(errors):.3f} px")
        print(f"{os.path.basename(left_path)} → std error: {np.std(errors):.3f} px")
        print(f"{os.path.basename(left_path)} → max error: {np.max(errors):.3f} px\n")

    if len(all_errors) == 0:
        print("No valid pairs for epipolar evaluation.")
        return

    all_errors = np.array(all_errors)

    print("\n=== Epipolar Error Summary ===")
    print(f"Mean:   {np.mean(all_errors):.3f} px")
    print(f"Median: {np.median(all_errors):.3f} px")
    print(f"Std:    {np.std(all_errors):.3f} px")
    print(f"Max:    {np.max(all_errors):.3f} px")


# =============================================================================
def get_image_pairs(left_path, right_path):
    left_files = sorted(glob.glob(os.path.join(left_path, "*")))
    pairs = []

    for lf in left_files:
        name = os.path.basename(lf)
        rf = os.path.join(right_path, name)

        if not os.path.exists(rf):
            raise ValueError(f"Missing pair for {lf}")

        pairs.append((lf, rf))

    return pairs


# =============================================================================
def run_stereo_calibration(args):
    """
    The function performs stereo calibration using Charuco boards for both the left and right cameras.
    It detects Charuco corners in the stereo image pairs, matches common points, and performs stereo calibration.
    The resulting stereo parameters can be used for rectification and 3D reconstruction.

    Parameters:
    - args: Command-line arguments containing paths to images, calibration files, and Charuco board parameters.

    Returns:
    - None
    """

    # Load intrinsics
    K1, D1, _, _ = load_calibration(args.left_calibration)
    K2, D2, _, _ = load_calibration(args.right_calibration)

    # Create board
    board = aruco.CharucoBoard(
        (args.charuco_squares_x, args.charuco_squares_y),
        args.square_length,
        args.marker_length,
        aruco.getPredefinedDictionary(getattr(aruco, args.aruco_dict))
    )

    charuco_detector = aruco.CharucoDetector(board)

    # Load image filenames as pairs:
    left_right_pairs = get_image_pairs(args.left_images_path, args.right_images_path)
    left_images = [p[0] for p in left_right_pairs]
    right_images = [p[1] for p in left_right_pairs]

    assert len(left_images) == len(right_images), "Left/right image count mismatch"

    all_obj_points = []
    all_img_points_left = []
    all_img_points_right = []

    image_size = None

    for left_path, right_path in tqdm(zip(left_images, right_images), total=len(left_images)):

        imgL = cv2.imread(left_path)
        imgR = cv2.imread(right_path)

        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = grayL.shape[::-1]

        # Detect ChArUco
        _, _, cornersL, idsL = charuco_detector.detectBoard(grayL)
        _, _, cornersR, idsR = charuco_detector.detectBoard(grayR)

        if idsL is None or idsR is None:
            continue

        # Refine corners
        retL, charuco_cornersL, charuco_idsL = cv2.aruco.interpolateCornersCharuco(
            cornersL, idsL, grayL, board
        )
        retR, charuco_cornersR, charuco_idsR = cv2.aruco.interpolateCornersCharuco(
            cornersR, idsR, grayR, board
        )

        # if retL is None or retR is None:
        #     continue

        if (
            retL is None or retR is None or
            charuco_idsL is None or charuco_idsR is None or
            charuco_cornersL is None or charuco_cornersR is None
        ):
            continue

        if len(charuco_idsL) < 15 or len(charuco_idsR) < 15:
            continue

        # Match common IDs
        idsL = charuco_idsL.flatten()
        idsR = charuco_idsR.flatten()

        common_ids = np.intersect1d(idsL, idsR)

        # increased number of Charuco points to 15 for better estimation
        if len(common_ids) < 15:
            continue

        obj_pts = []
        img_pts_L = []
        img_pts_R = []

        board_corners_3d = board.getChessboardCorners()

        for cid in common_ids:
            idxL = np.where(idsL == cid)[0][0]
            idxR = np.where(idsR == cid)[0][0]

            obj_pts.append(board_corners_3d[cid])
            img_pts_L.append(charuco_cornersL[idxL][0])
            img_pts_R.append(charuco_cornersR[idxR][0])

        all_obj_points.append(np.array(obj_pts, dtype=np.float32))
        all_img_points_left.append(np.array(img_pts_L, dtype=np.float32))
        all_img_points_right.append(np.array(img_pts_R, dtype=np.float32))

    print(f"Valid stereo pairs: {len(all_obj_points)}")

    # Stereo calibration (FIX intrinsics)
    flags = cv2.CALIB_FIX_INTRINSIC
    # flags = 0

    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objectPoints=all_obj_points,
        imagePoints1=all_img_points_left,
        imagePoints2=all_img_points_right,
        cameraMatrix1=K1,
        distCoeffs1=D1,
        cameraMatrix2=K2,
        distCoeffs2=D2,
        imageSize=image_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        flags=flags
    )

    print("\nStereo calibration RMS error:", ret)
    print("\nLeft camera matrix K1:\n", K1)
    print("Left distortion D1:\n", D1)
    print("\nRight camera matrix K2:\n", K2)
    print("Right distortion D2:\n", D2)
    print("Rotation matrix R:\n", R)
    print("Translation vector T:\n", T)
    print("Essential matrix E:\n", E)
    print("Fundamental matrix F:\n", F)

    # Save results
    os.makedirs(args.save_path, exist_ok=True)

    np.savez(
        os.path.join(args.save_path, args.calibration_filename),
        camera_matrix_left=K1,
        dist_coeffs_left=D1,
        camera_matrix_right=K2,
        dist_coeffs_right=D2,
        R=R,
        T=T,
        E=E,
        F=F)

    print("Stereo calibration saved.")


    # Evaluate epipolar geometry:
    evaluate_epipolar_geometry(
        left_right_pairs,
        charuco_detector,
        board,
        F,
        K1, D1, K2, D2
    )


    # Save epipolar visualization:
    test_left = cv2.imread(left_images[0])
    test_right = cv2.imread(right_images[0])

    rectify_and_visualize(
        test_left, test_right,
        K1, D1, K2, D2,
        R, T,
        args.save_path
    )


# =============================================================================
if __name__ == "__main__":

    args = parse_arguments()
    run_stereo_calibration(args)
