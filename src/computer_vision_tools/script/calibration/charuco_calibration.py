#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrate the camera using a charuco board.

The default parameters are set for the long charuco board with 15x45 squares available at CSEM.
Default parameters are:
charuco_squares_x = 45
charuco_squares_y = 15
square_length = 0.012125
marker_length = 0.007275
aruco_dict = "DICT_5X5_1000"

An example to run the script:
$ python charuco_calibration.py \
    --images_path /home/ons/Documents/Databases/R3_MYDAS/calibration/phone_test/frames/ \
    --save_path /home/ons/Documents/Databases/R3_MYDAS/calibration/phone_test/results/ \
    --calibration_filename calibration.npz \
    --charuco_squares_x 45 \
    --charuco_squares_y 15 \
    --square_length 0.012125 \
    --marker_length 0.007275 \
    --aruco_dict "DICT_5X5_1000" \
    --images_step 20 \
    --save_images

Another board example with:
python charuco_calibration.py \
    --images_path /home/ons/Documents/experiments/AITOOL/stereo/4_camera_setup/v1_client_setup/calibration_22_04_2026/input_images/cam0/ \
    --save_path /home/ons/Documents/experiments/AITOOL/stereo/4_camera_setup/v1_client_setup/calibration_22_04_2026/calibration_results/cam0/ \
    --calibration_filename calibration.npz \
    --charuco_squares_x 25 \
    --charuco_squares_y 17 \
    --square_length 0.01053 \
    --marker_length 0.006318 \
    --aruco_dict "DICT_4X4_250" \
    --images_step 1 \
    --save_images


@author: ons
"""

# =============================================================================
import cv2
import cv2.aruco as aruco
import numpy as np
import glob
import os
import argparse
import pandas as pd
from tqdm import tqdm

from csem_vision_tools.utils.io_utils import save_calibration
from csem_vision_tools.utils.io_utils import natural_sort_key
from csem_vision_tools.utils.io_utils import load_calibration

from csem_vision_tools.utils.calibration import draw_reprojection
from csem_vision_tools.utils.calibration import draw_aruco_corners
from csem_vision_tools.utils.calibration import save_corner_heatmap
from csem_vision_tools.utils.calibration import draw_checkerboard_corners


# =============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    # Create an argument parser
    parser = argparse.ArgumentParser(description="The script to calibrate the camera using a charuco board.")

    # add images_path argument:
    parser.add_argument(
        "--images_path",
        type=str,
        default=None,
        help="Path to the folder with images."
    )
    # add save_path argument:
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the calibration results."
    )
    # add calibration_fileanme argument:
    parser.add_argument(
        "--calibration_filename",
        type=str,
        default="calibration.npz",
        help="Filename to save the calibration results."
    )
    # add load_calibration_filename argument, set to None by default:
    parser.add_argument(
        "--load_calibration_filename",
        type=str,
        default=None,
        help="Filename to load the calibration from. If given, only rvecs and tvecs are computed for all board images."
    )
    # add charuco_board argument:
    parser.add_argument(
        "--charuco_squares_x",
        type=int,
        default=45,
        help="Number of squares in X direction, number of columns."
    )
    # add charuco_board argument:
    parser.add_argument(
        "--charuco_squares_y",
        type=int,
        default=15,
        help="Number of squares in Y direction, number of rows."
    )
    # add square_length argument:
    parser.add_argument(
        "--square_length",
        type=float,
        default=0.012125,
        help="Square length in meters."
    )
    # add marker_length argument:
    parser.add_argument(
        "--marker_length",
        type=float,
        default=0.007275,
        help="ArUco marker length in meters."
    )
    # add aruco_dict argument:
    parser.add_argument(
        "--aruco_dict",
        type=str,
        default="DICT_5X5_1000",
        help="ArUco dictionary type."
    )
    # add images step argument:
    parser.add_argument(
        "--images_step",
        type=int,
        default=1,
        help="Step to skip images."
    )
    # save images with detected corners:
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save images with detected corners."
    )

    # Parse arguments
    return parser.parse_args()


def build_calibration_metadata(indices, filenames, rvecs, tvecs):
    data = []
    for idx, filename, rvec, tvec in zip(indices, filenames, rvecs, tvecs):
        rvec = rvec.flatten()
        tvec = tvec.flatten()
        data.append({
            "frame_index": idx,
            "filename": filename,
            "rvec_x": rvec[0],
            "rvec_y": rvec[1],
            "rvec_z": rvec[2],
            "tvec_x": tvec[0],
            "tvec_y": tvec[1],
            "tvec_z": tvec[2],
        })
    return pd.DataFrame(data)


def run_charuco_calibration(images_path,
                            save_path,
                            calibration_filename,
                            load_calibration_filename,
                            charuco_squares_x,
                            charuco_squares_y,
                            square_length,
                            marker_length,
                            aruco_dict,
                            images_step,
                            save_images):

    # Define the charuco board:
    board = aruco.CharucoBoard((charuco_squares_x, charuco_squares_y),
                               square_length,
                               marker_length,
                               aruco.getPredefinedDictionary(getattr(aruco, aruco_dict))
                               )

    detector_params = aruco.DetectorParameters()
    charuco_params = aruco.CharucoParameters()

    charuco_detector = aruco.CharucoDetector(board, charuco_params, detector_params)

    # Find all images in the directory:
    images = glob.glob(os.path.join(images_path, "*.jpg")) + glob.glob(os.path.join(images_path, "*.png")) + glob.glob(os.path.join(images_path, "*.bmp"))
    # Get only the filenames:
    images_filenames = [os.path.basename(fname) for fname in images]
    # Sort the images_filenames:
    images_filenames = sorted(images_filenames, key=natural_sort_key)

    # take every N-th image:
    images_filenames = images_filenames[::images_step]

    print(f"Found {len(images)} images")
    print(f"Using {len(images_filenames)} images")


    # Store all Charuco corners and ids
    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None
    image_filenames_selected = []
    frame_indices = []

    for idx, image_filename in enumerate(tqdm(images_filenames)):

        # load the image and convert to grayscale:
        image = cv2.imread(os.path.join(images_path, image_filename))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect using new CharucoDetector API
        _, _, aruco_corners, aruco_ids = charuco_detector.detectBoard(gray)

        if aruco_ids is None or len(aruco_ids) == 0:
            continue  # Skip images with no detection

        # Refine the charuco corners:
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(markerCorners=aruco_corners,
                                                                                   markerIds=aruco_ids,
                                                                                   image=gray,
                                                                                   board=board)

        # if refined coordinates are available:
        if retval is not None and retval >= 6:  # At least 6 corners needed
            all_charuco_corners.append(charuco_corners.astype(np.float32))
            all_charuco_ids.append(charuco_ids.astype(np.int32))
            if image_size is None:
                image_size = gray.shape[::-1]  # (width, height)
        else:
            continue  # Skip images with no detection

        # Store the image filename and index
        image_filenames_selected.append(image_filename)
        frame_indices.append(idx)

        if save_images:
            aruco_annotated_image = draw_aruco_corners(image, aruco_corners, aruco_ids)
            save_path_aruco = os.path.join(save_path, "aruco")
            if not os.path.exists(save_path_aruco):
                os.makedirs(save_path_aruco)
            # save the image with aruco corners:
            cv2.imwrite(os.path.join(save_path_aruco,
                                     image_filename), aruco_annotated_image)

            checkerboard_annotated_image = draw_checkerboard_corners(image, charuco_corners, charuco_ids)
            save_path_checkerboard = os.path.join(save_path, "checkerboard")
            if not os.path.exists(save_path_checkerboard):
                os.makedirs(save_path_checkerboard)
            # save the image with charuco corners:
            cv2.imwrite(os.path.join(save_path_checkerboard,
                                     image_filename), checkerboard_annotated_image)


    # load calibration parameters if calibration file is given, and estimate rvecs and tvecs:
    if load_calibration_filename is not None:
        # use the intrinsic parameters from the calibration file:
        camera_matrix, dist_coeffs, _, _ = load_calibration(load_calibration_filename)

        rvecs = []
        tvecs = []
        image_filenames_selected_temp = []
        frame_indices_temp = []

        # estimate rvecs and tvecs for each image:
        for charuco_corners, charuco_ids, image_filename, idx in zip(all_charuco_corners,
                                                                     all_charuco_ids,
                                                                     image_filenames_selected,
                                                                     frame_indices):
            # estimate pose for each image:
            success, rvec_frame, tvec_frame = cv2.aruco.estimatePoseCharucoBoard(
                charucoCorners=charuco_corners,
                charucoIds=charuco_ids,
                board=board,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs,
                rvec=np.empty(1),
                tvec=np.empty(1),
            )

            if success:
                rvecs.append(rvec_frame)
                tvecs.append(tvec_frame)
                image_filenames_selected_temp.append(image_filename)
                frame_indices_temp.append(idx)
        # keep only the images with valid pose estimation:
        image_filenames_selected = image_filenames_selected_temp
        frame_indices = frame_indices_temp
        assert len(rvecs) == len(tvecs) == len(image_filenames_selected) == len(frame_indices)
        print("Estimation of rvecs and tvecs successful")


    # If no calibration file is given, perform calibration:
    else:
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=all_charuco_corners,
            charucoIds=all_charuco_ids,
            board=board,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None
        )
        print("Calibration successful")


    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)

    # Save the calibration results:
    print (f"Saving calibration results to {os.path.join(save_path, calibration_filename)}")
    save_calibration(os.path.join(save_path, calibration_filename),
                        camera_matrix,
                        dist_coeffs,
                        rvecs,
                        tvecs)

    """
    For Rendering in Blender use new_camera_matrix.
    To match the real camera intrinsics (especially focal length) accurately in Blender, set alpha = 0.
    alpha = 0 gives you the true, cropped camera matrix where only the valid image region remains.
    This matrix will most closely represent the actual projection used in rendering — it reflects the corrected, usable field of view.
    """
    # compute the new camera matrix:
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix=camera_matrix,
                                                            distCoeffs=dist_coeffs,
                                                            imageSize=image_size,
                                                            alpha=0)
    # print the new focal length:
    print(f"New focal length: {new_camera_matrix[0, 0]:.4f}, {new_camera_matrix[1, 1]:.4f} pixels")
    print("New camera matrix:\n", new_camera_matrix)


    # assert the length of rvecs and tvecs is equal to the number of selected images:
    assert len(rvecs) == len(tvecs) == len(image_filenames_selected) == len(frame_indices)

    # Save the calibration metadata:
    calibration_metadata = build_calibration_metadata(frame_indices, image_filenames_selected, rvecs, tvecs)
    # save the metadata to a hdf5 file:
    calibration_metadata.to_hdf(os.path.join(save_path, "calibration_metadata.h5"),
                                key="calibration_metadata",
                                mode="w",
                                index=False)
    print(f"Calibration metadata saved to {os.path.join(save_path, 'calibration_metadata.h5')}")
    # to read the file:
    # calibration_metadata = pd.read_hdf(os.path.join(save_path, "calibration_metadata.h5"),
    #                                     key="calibration_metadata",
    #                                     mode="r",
    #                                     index=False)


    # save the undistorted images, the reprojection images, and the heatmap of the detected corners if save_images is True:
    if save_images:
        # save the undistorted images:
        undistort_path = os.path.join(save_path, "undistorted")
        os.makedirs(undistort_path, exist_ok=True)

        for image_filename in image_filenames_selected:
            img = cv2.imread(os.path.join(images_path, image_filename))

            undistorted = cv2.undistort(
                img,
                camera_matrix,
                dist_coeffs,
                None,
                new_camera_matrix
            )

            cv2.imwrite(os.path.join(undistort_path, image_filename), undistorted)


        # save the reprojection images:
        reprojection_path = os.path.join(save_path, "reprojection")
        os.makedirs(reprojection_path, exist_ok=True)

        for charuco_corners, charuco_ids, rvec, tvec, image_filename in zip(all_charuco_corners,
                                                                            all_charuco_ids,
                                                                            rvecs,
                                                                            tvecs,
                                                                            image_filenames_selected):
            img = cv2.imread(os.path.join(images_path, image_filename))

            reprojection_img = draw_reprojection(img, charuco_corners, charuco_ids, rvec, tvec, camera_matrix, dist_coeffs, board)

            cv2.imwrite(os.path.join(reprojection_path, image_filename), reprojection_img)


        # save the heatmap of the detected corners:
        save_corner_heatmap(all_charuco_corners,
                            image_size,
                            save_path,
                            filename="corner_heatmap.png",
                            overlay_image=None)


if __name__ == "__main__":

    # Parse arguments
    args = parse_arguments()

    # get command line arguments:
    images_path = args.images_path
    save_path = args.save_path
    calibration_filename = args.calibration_filename
    # check if the calibration filename is valid:
    if not calibration_filename.endswith(".npz"):
        calibration_filename = calibration_filename + ".npz"

    load_calibration_filename = args.load_calibration_filename

    charuco_squares_x = args.charuco_squares_x
    charuco_squares_y = args.charuco_squares_y
    square_length = args.square_length
    marker_length = args.marker_length
    aruco_dict = args.aruco_dict

    images_step = args.images_step

    save_images = args.save_images

    # check if the images path exists:
    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Images path {images_path} does not exist.")
    # check if the save path exists:
    if not os.path.exists(save_path):
        # create the save path:
        os.makedirs(save_path)

    # run charuco calibration:
    run_charuco_calibration(images_path,
                            save_path,
                            calibration_filename,
                            load_calibration_filename,
                            charuco_squares_x,
                            charuco_squares_y,
                            square_length,
                            marker_length,
                            aruco_dict,
                            images_step,
                            save_images)
