import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import use
from VanCollections import Frame, MatchDict, TracksData, \
    compute_relative_cov_mat
import gtsam

DATA_PATH = '../../VAN_ex/dataset/sequences/05/'  # path to frames data
CALIB_PATH = os.path.join(DATA_PATH, 'calib.txt')  # path to calibration matrix
GT_PATH = '../../VAN_ex/dataset/poses/05.txt'  # path to ground truth

BA_FACTOR_ERR_DICT = {'optimized': {}, 'pre_opt': {}}



# ------------------------------------- Functionality added for EX1 + EX2

def read_images(idx):
    img_name = '{:06d}.png'.format(idx)
    img1 = cv.imread(DATA_PATH + 'image_0/' + img_name, 0)
    img2 = cv.imread(DATA_PATH + 'image_1/' + img_name, 0)

    return img1, img2


def apply_ratio_test(matches: list, ratio_threshold=0.8):
    """
    This function applies the Lowe's ratio test.

    :param matches: matches list, as returned by CV.
    :param ratio_threshold: the threshold for the ratio.

    The function returns two lists: good and bad matches.
    """
    best_matches_ratio = []
    discarded_matches_ratio = []

    for neighbour1, neighbour2 in matches:
        cur_ratio = neighbour1.distance / neighbour2.distance

        if cur_ratio > ratio_threshold:  # setting the ratio threshold to 0.8
            discarded_matches_ratio.append(neighbour1)
            continue

        best_matches_ratio.append(neighbour1)

    print(f'{len(discarded_matches_ratio)} matches discarded by ratio test')

    return best_matches_ratio, discarded_matches_ratio


def get_frame_matches(matcher, left_values: tuple, right_values: tuple, use_knn=True):
    """
    This function finds matches using knnMatch

    :param matcher: cv2 matcher object.
    :param left_values: tuple of (keyPoints, descriptors) for left image.
    :param right_values: tuple of (keyPoints, descriptors) for right image.
    """
    kps_left, desc_left = left_values
    kps_right, desc_right = right_values

    if use_knn:
        return matcher.knnMatch(desc_left, desc_right, k=2)

    return matcher.match(desc_left, desc_right)


def apply_y_distance_test(matches: list, kp_left, kp_right, y_dist_threshold=2):
    """
    This function filters frame matches by Y distance.
    :param matches: matches
    :param kp_left: left image key points.
    :param kp_right:  right image key points.
    :param y_dist_threshold: distance threshold.
    :return: good matches, discarded matches.
    """
    keep_matches, discard_matches = [], []

    for match in matches:
        kp1_index = match.queryIdx
        kp2_index = match.trainIdx

        # obtaining y coordinate distance
        cur_y_dist = abs(kp_left[kp1_index].pt[1] - kp_right[kp2_index].pt[1])

        if cur_y_dist > y_dist_threshold:
            discard_matches.append(match)
            continue

        keep_matches.append(match)

    return keep_matches, discard_matches


def get_keypoints_idx_from_matches(matches: list):
    """
    This function returns the indices of keypoints from match list.
    :param matches: matches.
    :return: lists of indices.
    """
    left_kps_idx, right_kps_idx = [], []

    for match in matches:
        left_kps_idx.append(match.queryIdx)
        right_kps_idx.append(match.trainIdx)

    return left_kps_idx, right_kps_idx


def get_matches_cords_matrices(matches: list, left_kps, right_kps):
    """
    This function returns the coordinates of keypoints from matches.
    :param matches: matches list.
    :param left_kps: left image key points.
    :param right_kps: right image key points.
    :return: left coordinates, right coordinates.
    """
    left_idx_lst, right_idx_lst = get_keypoints_idx_from_matches(matches)

    left_mat = cv.KeyPoint_convert(left_kps, left_idx_lst)
    right_mat = cv.KeyPoint_convert(right_kps, right_idx_lst)

    return left_mat, right_mat


def compute_triangulation(camera_matrix: np.ndarray, left_ext: np.ndarray,
                          right_ext: np.ndarray, left_cords: tuple, right_cords: tuple):
    """
    This function computes triangulation for a single pair of points.

    :param camera_matrix: intrinsic camera matrix (assumed equal for both cameras).
    :param left_ext: external matrix for left camera.
    :param right_ext: external matrix for right camera.
    :param left_cords: coordinate of left point as tuple.
    :param right_cords: coordinate of right point as tuple.
    """
    left_matrix = camera_matrix @ left_ext
    right_matrix = camera_matrix @ right_ext

    # computing the rows of the design matrix
    row1 = left_matrix[2, :] * left_cords[0] - left_matrix[0, :]
    row2 = left_matrix[2, :] * left_cords[1] - left_matrix[1, :]
    row3 = right_matrix[2, :] * right_cords[0] - right_matrix[0, :]
    row4 = right_matrix[2, :] * right_cords[1] - right_matrix[1, :]

    # stacking rows to a matrix
    design_mat = np.vstack((row1, row2, row3, row4))

    # computing the SVD decomposition to find the solution
    u, s, v_t = np.linalg.svd(design_mat)

    last_vec = v_t.T[:, -1]

    return last_vec / last_vec[-1]


def triangulate_cloud(camera_matrix: np.ndarray, left_ext: np.ndarray,
                      right_ext: np.ndarray, left_points: np.ndarray,
                      right_points: np.ndarray):
    """
        This function computes triangulation for a single pair of points.

        :param camera_matrix: intrinsic camera matrix (assumed equal for both cameras).
        :param left_ext: external matrix for left camera.
        :param right_ext: external matrix for right camera.
        :param left_points: 2D array of points.
        :param right_points: 2D array of points.
        """
    point_cloud = np.empty((3, left_points.shape[1]))

    for cur_col in range(left_points.shape[1]):
        # storing current cords as a tuple
        cur_left_cords = (left_points[0, cur_col], left_points[1, cur_col])
        cur_right_cords = (right_points[0, cur_col], right_points[1, cur_col])

        triangulation_vec = compute_triangulation(camera_matrix, left_ext, right_ext,
                                                  cur_left_cords, cur_right_cords)
        triangulation_vec = triangulation_vec[:-1]

        point_cloud[:, cur_col] = triangulation_vec

    # adding row of ones to the cords table
    ones_row = np.ones_like(point_cloud[0, :])

    return np.vstack([point_cloud, ones_row])


def get_calib():
    """
    Gets cameras m1 and m2 projection matrices and k the intrinsic matrix from CALIB_PATH.
    :return:
            m1 - The projection matrix of left camera.
            m2 - The projection matrix of right camera.
            k - The intrinsic matrix.
    """
    # Read calib.txt data:
    with open(CALIB_PATH, 'r') as calib_file:
        l1 = calib_file.readline().split()[1:]
        l2 = calib_file.readline().split()[1:]

    # Get matrices data:
    l1 = [float(i) for i in l1]
    m1 = np.array(l1).reshape(3, 4)
    l2 = [float(i) for i in l2]
    m2 = np.array(l2).reshape(3, 4)

    # Get k, m1, m2:
    k = m1[:, :3]
    m1 = np.linalg.inv(k) @ m1
    m2 = np.linalg.inv(k) @ m2
    return k, m1, m2


def visualize_point_cloud(cloud_data: np.ndarray, title='', file_name=None):
    """
    This function visualizes a cloud point.
    :param cloud_data: 3D cloud coordinates.
    :param title: figure title.
    :param file_name: name of saved figure file.
    :return: None.
    """
    use('tkAgg')
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    ax.scatter(cloud_data[0, :], cloud_data[2, :], cloud_data[1, :])
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')

    fig.suptitle(title)

    # Show and save plot:
    plt.show()

    if file_name is not None:
        fig.savefig(f"q2.3-2.4i{file_name}.png")


def compute_frame_cloud(left_img_vals: tuple, right_img_vals: tuple):
    """
    This function computes the 3D point cloud coordinates of frame image pair.
    :param left_img_vals: tuple of: (keypoints, descriptors), matching the left image.
    :param right_img_vals: tuple of: (keypoints, descriptors), matching the right image.
    :return: cloud coordinates, kept matches, discarded matches.
    """
    if isinstance(left_img_vals[0], np.ndarray):
        return compute_frame_cloud_cnn(left_img_vals, right_img_vals)

    # setting the cv matching objects

    matcher_obj = cv.BFMatcher_create(crossCheck=True)

    matches_lst = get_frame_matches(matcher_obj, left_img_vals, right_img_vals, False)

    # filtering matches using vertical distance test
    keep_matches, discard_matches = apply_y_distance_test(matches_lst, left_img_vals[0],
                                                          right_img_vals[0])

    # obtaining key points coordinates
    left_cords, right_cords = get_matches_cords_matrices(keep_matches, left_img_vals[0],
                                                         right_img_vals[0])

    left_cords = left_cords.T
    right_cords = right_cords.T

    # compute triangulation cloud
    k, m1, m2 = get_calib()
    # frame_cloud = triangulate_cloud(k, m1, m2, left_cords, right_cords)
    frame_cloud = cv.triangulatePoints(k @ m1, k @ m2, left_cords, right_cords)
    frame_cloud = frame_cloud / frame_cloud[-1, :]

    return frame_cloud, keep_matches, discard_matches


# ------------------------------------- Functionality added for EX3


def get_multi_matches(frame0_matches, frame1_matches, consecutive_matches):
    """
    This function matches key points of 2 consecutive frames (total of 4 images) and
    returns a table with the indices of all the matching key points.
    :param frame0_matches: matches of the first frame.
    :param frame1_matches: matches of the second frame.
    :param consecutive_matches: matches between first left and second left.
    :return: indices table such that: col0 = indices of left0, col1=indices of right0,
    col2=indices of left 1, col3=indices of right1.
    """
    frame0_dict = MatchDict(frame0_matches)
    frame1_dict = MatchDict(frame1_matches)
    consecutive_dict = MatchDict(consecutive_matches)

    ind_rows = []

    for left0_ind in consecutive_dict.keys():
        right0_ind = frame0_dict.get_right_kp(left0_ind)
        left1_ind = consecutive_dict.get_right_kp(left0_ind)
        right1_ind = frame1_dict.get_right_kp(left1_ind)

        if right0_ind is None or right1_ind is None:
            continue

        index_row = np.array([left0_ind, right0_ind, left1_ind, right1_ind])
        index_row.reshape((1, 4))
        ind_rows.append(index_row)

    return np.vstack(ind_rows)


def slice_cloud_by_multi_match(point_cloud: np.ndarray, multi_match_index: np.ndarray,
                               cloud_matches):
    """
    This function makes sure that the cloud points table is arranged in the same order as
    the indices in multi_match index.
    :param point_cloud: 3D points coordinates.
    :param multi_match_index: multi-match indices table.
    :param cloud_matches: matches of first left.
    :return: re-arranged cloud table.
    """
    index_list = np.empty_like(multi_match_index)

    for match_ind in range(len(cloud_matches)):
        kp_ind = cloud_matches[match_ind].queryIdx
        cur_position = np.argwhere(multi_match_index == kp_ind)

        if cur_position.shape[0] == 0:
            continue

        cur_position = cur_position.item()

        index_list[cur_position] = match_ind

    res_cloud = point_cloud[:, index_list]

    return res_cloud


def solve_p3p(ind_table: np.ndarray, left1_kps, world_points: np.ndarray,
              intrinsic_mat: np.ndarray, subset: np.ndarray):
    """
    This function computes P3P.
    :param ind_table: multi match indices table.
    :param left1_kps: second left key points.
    :param world_points: 3D cloud points.
    :param intrinsic_mat: cameras intrinsic matrix.
    :param subset: 4 indices for which to compute the procedure.
    :return: transform matrix as p3p result.
    """
    slice_ind = subset

    cur_ind_table = ind_table[slice_ind, :]
    cur_ind_table = cur_ind_table.astype(int)

    if isinstance(left1_kps, np.ndarray):  # if already np array - just slice
        cons_left1_cords = left1_kps[cur_ind_table[:, 2], :]
    else:
        cons_left1_cords = cv.KeyPoint_convert(left1_kps, list(cur_ind_table[:, 2]))

    world_mat = (world_points[: -1, :]).T
    world_mat = world_mat[slice_ind, :]

    pnp_sol = cv.solvePnPGeneric(world_mat, cons_left1_cords, intrinsic_mat, None,
                                 flags=cv.SOLVEPNP_P3P)

    return np.hstack([cv.Rodrigues(pnp_sol[1][0])[0], pnp_sol[2][0]])


def solve_pnp(ind_table: np.ndarray, left1_kps, world_points: np.ndarray,
              intrinsic_mat: np.ndarray, inliners: np.ndarray):
    """
    This function computes PNP.
    :param ind_table: multi match indices table.
    :param left1_kps: second left key points.
    :param world_points: 3D points coordinates.
    :param intrinsic_mat: cameras intrinsic matrix.
    :param inliners: indices of the inliers on which to compute PNP.
    :return: transformation matrix as PNP result.
    """
    cur_ind_table = ind_table[inliners, :]
    cur_ind_table = cur_ind_table.astype(int)

    if isinstance(left1_kps, np.ndarray):  # if already np array - just slice
        cons_left1_cords = left1_kps[cur_ind_table[:, 2], :]
    else:
        cons_left1_cords = cv.KeyPoint_convert(left1_kps, list(cur_ind_table[:, 2]))

    world_mat = (world_points[: -1, :]).T
    world_mat = world_mat[inliners, :]

    pnp_sol = cv.solvePnPGeneric(world_mat, cons_left1_cords, intrinsic_mat, None)

    return np.hstack([cv.Rodrigues(pnp_sol[1][0])[0], pnp_sol[2][0]])


def project_cloud(extrinsic_matrix: np.ndarray, intrinsic_matrix: np.ndarray,
                  cloud_points):
    """
    This function projects a 3D point cloud in the global coordinates system to pixel
    coordinates for a certain camera.
    :param extrinsic_matrix: camera's extrinsic matrix.
    :param intrinsic_matrix: camera's intrinsic matrix.
    :param cloud_points: 3D points coordinates.
    :return: projected cloud in pixels.
    """
    temp_cloud = np.empty_like(cloud_points)
    temp_cloud[:-1, :] = cloud_points[:-1, :]
    temp_cloud[-1, :] = np.ones_like(cloud_points[-1, :])

    camera_matrix = intrinsic_matrix @ extrinsic_matrix

    return camera_matrix @ temp_cloud


def check_supporters(left0_ext: np.ndarray, right0_ext: np.ndarray, left1_ext: np.ndarray,
                     right1_ext: np.ndarray, left0_pixels: np.ndarray,
                     right0_pixels: np.ndarray, left1_pixels: np.ndarray,
                     right1_pixels: np.ndarray, intrinsic_matrix: np.ndarray,
                     point_cloud: np.ndarray, dist_threshold=2):
    """
    This function finds the supporters of a model.
    :param left0_ext: first left extrinsic matrix.
    :param right0_ext: first right extrinsic matrix.
    :param left1_ext: second left extrinsic matrix.
    :param right1_ext: second right extrinsic matrix.
    :param left0_pixels: pixel projection to first left camera.
    :param right0_pixels: pixel projection to first right camera.
    :param left1_pixels: pixel projection to second left camera.
    :param right1_pixels: pixel projection to second right camera.
    :param intrinsic_matrix: camera's intrinsic matrix.
    :param point_cloud: 3D cloud coordinates.
    :param dist_threshold: max projection error. default is 2.
    :return: indices of supporters.
    """
    # computing projections
    left0_projected = project_cloud(left0_ext, intrinsic_matrix, point_cloud)
    right0_projected = project_cloud(right0_ext, intrinsic_matrix, point_cloud)
    left1_projected = project_cloud(left1_ext, intrinsic_matrix, point_cloud)
    right1_projected = project_cloud(right1_ext, intrinsic_matrix, point_cloud)

    # dividing by last coordinate
    left0_projected = left0_projected / left0_projected[-1, :]
    right0_projected = right0_projected / right0_projected[-1, :]
    left1_projected = left1_projected / left1_projected[-1, :]
    right1_projected = right1_projected / right1_projected[-1, :]

    # computing distance from actual pixels
    left0_pixel_dist = np.abs(left0_projected[:-1, :] - left0_pixels)
    right0_pixel_dist = np.abs(right0_projected[:-1, :] - right0_pixels)
    left1_pixel_dist = np.abs(left1_projected[:-1, :] - left1_pixels)
    right1_pixel_dist = np.abs(right1_projected[:-1, :] - right1_pixels)

    # stacking all distances to make a decision
    overall_dist = np.vstack([left0_pixel_dist, right0_pixel_dist, left1_pixel_dist,
                              right1_pixel_dist])
    overall_dist = np.max(overall_dist, axis=0)

    # returning the indices of the inliers
    return np.argwhere(overall_dist < dist_threshold).flatten()


def get_right1_matrix(right0_mat: np.ndarray, left1_mat: np.ndarray):
    """
    This function computes the extrinsic matrix of the second right camera.
    :param right0_mat: first right.
    :param left1_mat: second left.
    :return: second right extrinsic mat.
    """
    right1_r_mat = right0_mat[:, :-1] @ left1_mat[:, :-1]
    right1_t_vec = right0_mat[:, :-1] @ left1_mat[:, -1].reshape((3, 1)) + \
                   right0_mat[:, -1].reshape((3, 1))

    return np.hstack([right1_r_mat, right1_t_vec])


def get_camera_location(camera_mat: np.ndarray):
    """
    This function computes a camera's location using its extrinsic matrix.
    :param camera_mat: extrinsic matrix.
    :return: camera's coordinates in the global system.
    """
    r_mat = camera_mat[:, :-1]
    t_vec = camera_mat[:, -1]
    camera_point = -r_mat.T @ t_vec
    camera_point = camera_point.reshape((3, 1))

    return camera_point


def update_num_ransac_iterations(outliers_ratio, success_rate, num_samples=4):
    """
    This function computes the remaining number of ransac iterations left.
    :param outliers_ratio: outliers ratio.
    :param success_rate: desired success rate.
    :param num_samples: number of model samples.
    :return: number of remaining iterations.
    """
    upper = np.log(1 - success_rate)

    lower = np.power(1 - outliers_ratio, num_samples)
    lower = np.log(1 - lower)

    return np.ceil(upper / lower)


def compute_ransac(cloud_points: np.ndarray, cloud_ind_table: np.ndarray,
                   left1_keypoints, intrinsic_mat: np.ndarray, left0_ext_mat,
                   right0_ext_mat, left0_kps_cords, right0_kps_cords, left1_kps_cords,
                   right1_kps_cords, min_iter=50):
    """
    This function computes the RANSAC algorithm.
    :param cloud_points: 3D cloud points.
    :param cloud_ind_table: multi match indices.
    :param left1_keypoints: second left key points.
    :param intrinsic_mat: camera's intrinsic matrix.
    :param left0_ext_mat: first left extrinsic matrix.
    :param right0_ext_mat: first right extrinsic matrix.
    :param left0_kps_cords: first left key points coordinates.
    :param right0_kps_cords: first right key points coordinates.
    :param left1_kps_cords: second left key points coordinates.
    :param right1_kps_cords: second right key points coordinates.
    :param min_iter: minimum number of iterations.
    :return: model supporters indices, transform matrix.
    """
    max_supporters_num = 0
    supporters = None
    cur_iteration = 0
    iterations_num = update_num_ransac_iterations(0.9, 0.999999)

    while cur_iteration <= max(iterations_num, min_iter):
        cur_p3p_points = np.random.choice(np.arange(cloud_ind_table.shape[0]),
                                          size=4, replace=False)

        try:
            cur_left1 = solve_p3p(cloud_ind_table, left1_keypoints, cloud_points,
                                  intrinsic_mat, cur_p3p_points)
        except IndexError:
            continue

        cur_right1 = get_right1_matrix(right0_ext_mat, cur_left1)

        cur_supporters = check_supporters(left0_ext_mat, right0_ext_mat, cur_left1,
                                          cur_right1, left0_kps_cords.T,
                                          right0_kps_cords.T, left1_kps_cords.T,
                                          right1_kps_cords.T, intrinsic_mat, cloud_points)

        if cur_supporters.shape[0] > max_supporters_num:
            max_supporters_num = cur_supporters.shape[0]
            supporters = cur_supporters

            supporters_ratio = max_supporters_num / cloud_ind_table.shape[0]
            iterations_num = update_num_ransac_iterations(1 - supporters_ratio, 0.999999)

        cur_iteration += 1

    # print(f'#sup = {supporters.shape[0]}')

    best_left1 = solve_pnp(cloud_ind_table, left1_keypoints, cloud_points, intrinsic_mat,
                           supporters)

    return supporters, best_left1


def get_camera_ground_truth():
    """
    This function loads the ground truth positions of the camera.
    :return: positions list.
    """
    gt_lst = []

    with open(GT_PATH, 'r') as gt_file:
        while gt_file:
            try:
                cur_data = gt_file.readline().split()
                cur_data = [float(val) for val in cur_data]
                cur_data = np.array(cur_data).reshape(3, 4)

                gt_lst.append(cur_data)

            except ValueError:
                break

    return gt_lst


def compose_t_transform(current_mat: np.ndarray, next_step: np.ndarray):
    """
    This function updates the position of the left camera by composing current camera
    with the transform computed by PNP.
    :param current_mat: current camera's extrinsic matrix.
    :param next_step: next step matrix. Obtained by PNP.
    :return: new extrinsic matrix of the camera.
    """
    current_r = current_mat[:, :-1]
    current_t = current_mat[:, -1].reshape(3, 1)

    next_r = next_step[:, :-1]
    next_t = next_step[:, -1].reshape(3, 1)

    comp_r = next_r @ current_r
    comp_t = next_r @ current_t + next_t
    comp_t = comp_t.reshape(3, 1)

    comp_mat = np.hstack([comp_r, comp_t])

    return comp_mat


def convert_multi_kps(left0_kps, right0_kps, left1_kps, right1_kps,
                      multi_match_ind: np.ndarray):
    """
    This function converts key points objects to coordinates table.
    :param left0_kps: first left keypoints.
    :param right0_kps: first right keypoints.
    :param left1_kps: second left keypoints.
    :param right1_kps: second right keypoints.
    :param multi_match_ind: multi match indices.
    :return: first left cords, first right cords, second left cords, second right cords.
    """
    left0_kps_cords = cv.KeyPoint_convert(left0_kps, multi_match_ind[:, 0])
    right0_kps_cords = cv.KeyPoint_convert(right0_kps, multi_match_ind[:, 1])
    left1_kps_cords = cv.KeyPoint_convert(left1_kps, multi_match_ind[:, 2])
    right1_kps_cords = cv.KeyPoint_convert(right1_kps, multi_match_ind[:, 3])

    return left0_kps_cords, right0_kps_cords, left1_kps_cords, right1_kps_cords


def filter_negative_matches(multi_match_ind: np.ndarray, multi_cloud: np.ndarray):
    """
    This function filters indices of matches with negative 3D z-values from the multi
    match table.
    :param multi_match_ind: multi match table.
    :param multi_cloud: 3D points coordinates.
    :return: filtered multi matches, filtered 3D points.
    """
    # finding indices of non-negative z
    keep_ind = np.argwhere(multi_cloud[2, :] > 0).flatten()
    return multi_match_ind[keep_ind, :], multi_cloud[:, keep_ind]


def track_frames(num_frames=2560, saved_data_name=None, save_supporters=False):
    """
    This function performs tracking over a sequence of frames.
    :param num_frames: number of frames to track.
    :param saved_data_name: name of saved data file.
    :param save_supporters: if True - supporters percentage is saved in the data frame.
    :return: computed positions, ground truth positions.
    """
    tracking_res = []
    gt_res = []
    cur_k, cur_m1, cur_m2 = get_calib()
    tracking_detector = cv.SIFT_create()
    tracking_matcher = cv.BFMatcher()

    # read ground truth file
    gt_lst = get_camera_ground_truth()

    # reading frames
    cur_frame0 = Frame(0)
    left0_vals, right0_vals = cur_frame0.get_key_points_and_descriptors(tracking_detector)
    cur_f0_cloud, cur_f0_keep, cur_f0_discard = compute_frame_cloud(left0_vals,
                                                                    right0_vals)

    # creating the tracking matrix
    current_left0_tracking = cur_m1
    current_gt_tracking = gt_lst[0]

    # creating a tracking data frame
    track_df = TracksData(save_supporters)

    for frame_ind in range(1, num_frames):
        print(f'frame ind = {frame_ind}')
        cur_frame1 = Frame(frame_ind)

        # computing detection values for frames
        left1_vals, right1_vals = cur_frame1.get_key_points_and_descriptors(
            tracking_detector)

        # computing frames clouds
        cur_f1_cloud, cur_f1_keep, cur_f1_discard = compute_frame_cloud(left1_vals,
                                                                        right1_vals)

        # computing consecutive matches
        cur_cons_matches = tracking_matcher.match(left0_vals[1], left1_vals[1])

        # crossing matches to get multi cloud
        cur_multi_match_ind = get_multi_matches(cur_f0_keep, cur_f1_keep,
                                                cur_cons_matches)
        cur_multi_cloud = slice_cloud_by_multi_match(cur_f0_cloud,
                                                     cur_multi_match_ind[:, 0],
                                                     cur_f0_keep)
        # cur_multi_cloud = cur_f0_cloud

        # filtering matches with negative z-values in cloud
        cur_multi_match_ind, cur_multi_cloud = filter_negative_matches(
            cur_multi_match_ind, cur_multi_cloud)

        # key points cords table
        left0_kps_cords, right0_kps_cords, left1_kps_cords, right1_kps_cords = \
            convert_multi_kps(left0_vals[0], right0_vals[0], left1_vals[0],
                              right1_vals[0], cur_multi_match_ind)

        # performing P3P ransac
        cur_supporters, cur_left1_ext = compute_ransac(cur_multi_cloud,
                                                       cur_multi_match_ind, left1_vals[0],
                                                       cur_k, cur_m1, cur_m2,
                                                       left0_kps_cords, right0_kps_cords,
                                                       left1_kps_cords, right1_kps_cords)
        # updating track df
        track_df.insert_tracks(frame_ind - 1, cur_multi_match_ind[cur_supporters, :],
                               left0_vals[0], right0_vals[0], left1_vals[0],
                               right1_vals[0])

        # updating supporters data
        track_df.update_supporters_data(frame_ind, cur_supporters.shape[0],
                                        cur_multi_match_ind.shape[0])

        # computing left0 location
        cur_left0_point = get_camera_location(current_left0_tracking)
        tracking_res.append(cur_left0_point)

        # saving current left cam position
        track_df.update_camera_location(frame_ind - 1, current_left0_tracking)

        # computing ground truth location
        cur_gt_point = get_camera_location(current_gt_tracking)
        gt_res.append(cur_gt_point)

        # updating left0 matrix
        current_left0_tracking = compose_t_transform(current_left0_tracking,
                                                     cur_left1_ext)
        current_gt_tracking = gt_lst[frame_ind]

        # updating frame1 to be frame0
        left0_vals, right0_vals = left1_vals, right1_vals
        cur_f0_cloud, cur_f0_keep, cur_f0_discard = cur_f1_cloud, cur_f1_keep, \
            cur_f1_discard

    # saving the tracking data
    if saved_data_name:
        track_df.to_pickle(saved_data_name)

    return np.hstack(tracking_res), np.hstack(gt_res)


def filter_extreme_points(cords_table: np.ndarray, quantile_threshold=0.8):
    """
    This function filter extreme coordinates values.
    :param cords_table: cords table.
    :param quantile_threshold: quantile to threshold.
    :return: filtered cords table.
    """
    # verifying format
    assert cords_table.shape[0] == 3

    abs_data = np.abs(cords_table)

    # obtaining thresholds
    x_threshold = np.quantile(abs_data[0, :], quantile_threshold)
    x_indices = np.argwhere(abs_data[0, :] < x_threshold).flatten()

    z_threshold = np.quantile(abs_data[2, :], quantile_threshold)
    z_indices = np.argwhere(abs_data[2, :] < z_threshold).flatten()

    keep_indices = np.intersect1d(x_indices, z_indices)

    return cords_table[:, keep_indices]


def transform_by_ext_mat(point_3d: np.ndarray, ext_mat: np.ndarray):
    """
    This function transforms a 3d point using ext mat.
    :param point_3d: 3d point.
    :param ext_mat: ext mat.
    :return: transformed point in cam cords.
    """
    cur_rotation = ext_mat[:, :-1]
    cur_translation = ext_mat[:, -1].reshape((3, 1))

    return cur_rotation @ point_3d + cur_translation


def evaluate_frames_distance(tracks_data: TracksData, first_frame: int, last_frame: int):
    """
    This function evaluates the distance between 2 cameras.
    :param tracks_data: tracks data.
    :param first_frame: first frame.
    :param last_frame: last frame.
    :return: distance.
    """
    if first_frame == last_frame:
        return 0

    if last_frame < first_frame:
        raise ValueError(f'invalid dist arguments: kf1: {first_frame}, kf2: {last_frame}')

    first_frame_ext = tracks_data.get_camera_location(first_frame)
    last_frame_ext = tracks_data.get_camera_location(last_frame)

    first_cam_pos = get_camera_location(first_frame_ext)
    last_cam_pos = get_camera_location(last_frame_ext)

    return np.linalg.norm(last_cam_pos - first_cam_pos)


def compute_positions_distance(first_ext: np.ndarray, second_ext: np.ndarray):
    """
    This function computes the distance between 2 poses using ext matrices.
    :param first_ext: first ext mat.
    :param second_ext: second ext mat.
    :return: distance.
    """
    first_pos = get_camera_location(first_ext)
    second_pos = get_camera_location(second_ext)

    return np.linalg.norm(second_pos - first_pos)


def invert_extrinsic(ext_mat: np.ndarray):
    """
    This function inverts the transform of the extrinsic matrix and returns a matrix
    representing the inverse transform.
    :param ext_mat: extrinsic matrix.
    :return: inverse transform matrix.
    """
    r_mat = ext_mat[:, :-1]
    t_vec = ext_mat[:, -1].reshape((3, 1))

    inv_r_mat = r_mat.T
    inv_t_vec = -inv_r_mat @ t_vec

    return np.hstack([inv_r_mat, inv_t_vec])


def compute_relative_ext(first_ext: np.ndarray, new_ext: np.ndarray):
    """
    This function converts extrinsic matrix in the global coordinates to an extrinsic
    matrix in the coordinates of the first frame in a bundle.
    :param first_ext: extrinsic matrix of first camera in the bundle.
    :param new_ext: new matrix to convert.
    :return: new extrinsic matrix converted to relational.
    """
    first_rotation = first_ext[:, :-1]
    first_translation = first_ext[:, -1].reshape((3, 1))

    new_rotation = new_ext[:, :-1]
    new_translation = new_ext[:, -1].reshape((3, 1))

    # computing the relative rotation
    relative_rotation = new_rotation @ first_rotation.T

    # computing the relative translation
    relative_translation = new_translation - relative_rotation @ first_translation

    return np.hstack([relative_rotation, relative_translation])


def get_cameras_ext_from_values(vals: gtsam.Values, pose_keys: dict,
                                global_pose: np.ndarray):
    cam_ext_list = []

    for pose_key in pose_keys.values():
        cur_pose = vals.atPose3(pose_key).matrix()[: -1, :]
        cur_pose = invert_extrinsic(cur_pose)
        cur_pose = compose_t_transform(global_pose, cur_pose)
        cam_ext_list.append(cur_pose)

    return cam_ext_list


def get_camera_locations_from_values(vals: gtsam.Values, pose_keys: dict,
                                     global_pose: np.ndarray):
    """
    This function extracts camera locations from gtsam values.
    :param vals: gtsam Values.
    :param pose_keys: poses symbols.
    :param global_pose: first bundle ext matrix in global cords.
    :return: cameras locations as numpy array.
    """
    cam_ext_list = get_cameras_ext_from_values(vals, pose_keys, global_pose)
    cam_cords_table = None

    for cam_ext in cam_ext_list:
        cur_pose = get_camera_location(cam_ext)

        if cam_cords_table is None:
            cam_cords_table = cur_pose
            continue

        cam_cords_table = np.hstack([cam_cords_table, cur_pose])

    return cam_cords_table, cam_ext_list


# ------------------------------------------------------------ Functionality added for EX5
def update_ba_factor_err(first_kf: int, factor_list: list, initial_vals: gtsam.Values,
                         optimized: gtsam.Values):
    init_err_sum, opt_err_sum = 0, 0

    for cur_factor in factor_list:
        init_err_sum += cur_factor.error(initial_vals)
        opt_err_sum += cur_factor.error(optimized)

    BA_FACTOR_ERR_DICT['pre_opt'].update({first_kf: init_err_sum / len(factor_list)})
    BA_FACTOR_ERR_DICT['optimized'].update({first_kf: opt_err_sum / len(factor_list)})


def compute_bundle_window(first_frame: int, last_frame: int, tracks_data: TracksData):
    """
    This function computes a bundle window.
    :param first_frame: first frame id.
    :param last_frame: last frame if.
    :param tracks_data: tracks data.
    :return: optimization results and symbols.
    """
    tracks_ids = tracks_data.get_overlapping_range_tracks(first_frame, last_frame)
    k, m1, m2 = get_calib()  # calibration data
    k = gtsam.Cal3_S2Stereo(k[0, 0], k[1, 1], 0, k[0, -1], k[1, -1], -m2[0, -1])
    initial_vals = gtsam.Values()
    graph = gtsam.NonlinearFactorGraph()
    factor_lst = []

    # computing last cam pose
    last_cam_pose = tracks_data.get_camera_location(last_frame)
    first_cam_pose = tracks_data.get_camera_location(first_frame)
    pose_keys = {}

    last_cam_pose = compute_relative_ext(first_cam_pose, last_cam_pose)
    last_cam_pose = invert_extrinsic(last_cam_pose)
    last_cam_pose = gtsam.Pose3(gtsam.Rot3(last_cam_pose[:, :-1]), last_cam_pose[:, -1])

    last_stereo_cam = gtsam.StereoCamera(last_cam_pose, k)

    # adding cameras positions to values
    for frame_id in range(first_frame, last_frame + 1):
        cur_cam_pose = np.copy(tracks_data.get_camera_location(frame_id))
        cur_cam_pose = compute_relative_ext(first_cam_pose, cur_cam_pose)
        cur_cam_pose = invert_extrinsic(cur_cam_pose)
        cur_cam_pose = gtsam.Pose3(gtsam.Rot3(cur_cam_pose[:, :-1]), cur_cam_pose[:, -1])

        cur_pos_symbol = gtsam.symbol('c', frame_id)
        initial_vals.insert(cur_pos_symbol, cur_cam_pose)

        pose_keys.update({frame_id: cur_pos_symbol})

    # computing last frame triangulations
    q_keys = []

    for track_id in tracks_ids:
        cur_frames = tracks_data.get_frames_by_track(track_id)

        while cur_frames[-1] > last_frame:
            cur_frames.pop()

        while cur_frames[0] < first_frame:
            cur_frames = cur_frames[1:]

        last_img_cords = tracks_data.get_track_frame_cords(cur_frames[-1], track_id)
        last_x_left, last_x_right, last_y = last_img_cords
        last_stereo_point = gtsam.StereoPoint2(last_x_left, last_x_right, last_y)
        cur_q = last_stereo_cam.backproject(last_stereo_point).astype(float)

        # checking for nan values in qs
        if np.any(np.isnan(cur_q)):
            continue

        # checking for negative z values
        if cur_q[2] < 0 or cur_q[2] > 100:
            continue

        # creating key symbol for the point
        cur_q_symbol = gtsam.symbol('q', track_id)

        initial_vals.insert(cur_q_symbol, cur_q)
        q_keys.append(cur_q_symbol)

        # creating projection factors
        for frame_id in cur_frames:
            cur_img_cords = tracks_data.get_track_frame_cords(frame_id, track_id)
            cur_x_left, cur_x_right, cur_y = cur_img_cords
            cur_stereo_point = gtsam.StereoPoint2(cur_x_left, cur_x_right, cur_y)
            sigma_cov = gtsam.noiseModel.Diagonal.Sigmas(np.ones(3))

            cur_factor = gtsam.GenericStereoFactor3D(cur_stereo_point, sigma_cov,
                                                     pose_keys[frame_id], cur_q_symbol, k)
            factor_lst.append(cur_factor)
            graph.add(cur_factor)

    # adding prior point
    sigma_6d = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6))
    origin = gtsam.Pose3()
    cur_prior_factor = gtsam.PriorFactorPose3(pose_keys[first_frame], origin, sigma_6d)
    graph.add(cur_prior_factor)

    # running optimization
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_vals)
    result = optimizer.optimize()

    update_ba_factor_err(first_frame, factor_lst, initial_vals, result)

    return graph, result, pose_keys, q_keys


# --------------------------------------- Functionality added for EX7 and final submission


def get_global_gtsam_pose(tracks_data: TracksData, frame_id: int):
    """
    This function returns the global position of the left camera in the given frame as
    gtsam Pose3 object.
    :param tracks_data: tracks data object.
    :param frame_id: frame id.
    :return: pose as Pose3.
    """
    global_pose = tracks_data.get_window_optimized_pose(frame_id)
    global_pose = invert_extrinsic(global_pose)

    return gtsam.Pose3(gtsam.Rot3(global_pose[:, :-1]),
                       global_pose[:, -1].reshape((3, 1)))


def convert_pose_to_gtsam(ext_mat: np.ndarray):
    """
    This function converts any pose (extrinsic matrix) to gtsam Pose3 object.
    :param ext_mat: extrinsic matrix.
    :return: Pose3 object.
    """
    new_mat = invert_extrinsic(ext_mat)

    return gtsam.Pose3(gtsam.Rot3(new_mat[:, :-1]), new_mat[:, -1].reshape((3, 1)))


def construct_pose_graph_objects():
    """
    This function constructs and initialized the objects required for pose graph
    optimization.
    :return: pose graph, initial values.
    """
    graph = gtsam.NonlinearFactorGraph()

    c0_symbol = gtsam.symbol('c', 0)
    sigma_6d = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-8)
    origin = gtsam.Pose3()
    cur_prior_factor = gtsam.PriorFactorPose3(c0_symbol, origin, sigma_6d)
    graph.add(cur_prior_factor)

    init_vals = gtsam.Values()
    init_vals.insert(c0_symbol, origin)

    return graph, init_vals


def get_loop_track_vals(src_symbol: str, dst_symbol: str):
    """
    This function returns frame objects (implemented in an earlier stage of this project
    and used mainly in ex3) matching to the symbols passed to the function.
    :param src_symbol: source pose symbol, for example: 'c0'.
    :param dst_symbol: destination pose symbol.
    :return: source frame, destination frame.
    """
    src_frame_id = int(src_symbol[1:])
    dst_frame_id = int(dst_symbol[1:])

    src_frame = Frame(src_frame_id)
    dst_frame = Frame(dst_frame_id)

    return src_frame, dst_frame


def compute_2kf_bundle(tracks_data: TracksData, src_symbol: str, dst_symbol: str,
                       supporters, left_src_kps, right_src_kps, left_dst_kps,
                       right_dst_kps):
    """
    This function performs bundle adjustment optimization for a window consists solely of
    2 key frames.
    :param tracks_data: tracks data object.
    :param src_symbol: source pose symbol, for example: 'c0'.
    :param dst_symbol: destination pose symbol.
    :param supporters: supporters indices as numpy array.
    :param left_src_kps: key points coordinates in the left cam of the source frame.
    :param right_src_kps: key points coordinates in the right cam of the source frame.
    :param left_dst_kps: key points coordinates in the left cam of the destination frame.
    :param right_dst_kps: key points coordinates in the right cam of the destination frame.
    :return: relative covariance between first and last key frames, and relative pose as
             gtsam Pose3 object.
    """
    src_frame_id = int(src_symbol[1:])
    dst_frame_id = int(dst_symbol[1:])

    # getting poses in global cords
    src_global_pose = tracks_data.get_window_optimized_pose(src_frame_id)
    dst_global_pose = tracks_data.get_window_optimized_pose(dst_frame_id)
    pose_keys = {}

    # computing cameras relative poses
    dst_relative_pose = compute_relative_ext(src_global_pose, dst_global_pose)
    dst_relative_pose = invert_extrinsic(dst_relative_pose)
    dst_relative_pose = gtsam.Pose3(gtsam.Rot3(dst_relative_pose[:, :-1]),
                                    dst_relative_pose[:, -1].reshape((3, 1)))
    src_relative_pose = gtsam.Pose3()

    # initializing factor graph for bundle optimization
    k, m1, m2 = get_calib()
    k = gtsam.Cal3_S2Stereo(k[0, 0], k[1, 1], 0, k[0, -1], k[1, -1], -m2[0, -1])
    initial_vals = gtsam.Values()
    graph = gtsam.NonlinearFactorGraph()

    dst_stereo_cam = gtsam.StereoCamera(dst_relative_pose, k)

    # inserting poses
    poses_lst = [src_relative_pose, dst_relative_pose]
    frames_ids = [src_frame_id, dst_frame_id]

    for i in range(len(frames_ids)):
        cur_symbol_obj = gtsam.symbol('c', frames_ids[i])
        pose_keys.update({frames_ids[i]: cur_symbol_obj})
        initial_vals.insert(cur_symbol_obj, poses_lst[i])

    # inserting q points
    q_count, q_keys = 0, []

    for supp_id in supporters:
        # triangulating from dst frame
        left_dst_cords = left_dst_kps[supp_id, :]
        right_dst_cords = right_dst_kps[supp_id, :]
        left_src_cords = left_src_kps[supp_id, :]
        right_src_cords = right_src_kps[supp_id, :]

        q_count += 1

        dst_stereo_point = gtsam.StereoPoint2(left_dst_cords[0], right_dst_cords[0],
                                              left_dst_cords[1])
        cur_q = dst_stereo_cam.backproject(dst_stereo_point).astype(float)

        # avoiding NAN values
        if np.any(np.isnan(cur_q)):
            continue

        # avoiding far or negative z points
        if cur_q[2] < 0 or cur_q[2] > 100:
            continue

        # creating q symbol
        cur_q_symbol = gtsam.symbol('q', q_count)
        q_keys.append(cur_q_symbol)
        initial_vals.insert(cur_q_symbol, cur_q)

        # adding q-dst factor
        sigma_cov = gtsam.noiseModel.Diagonal.Sigmas(np.ones(3))
        dst_factor = gtsam.GenericStereoFactor3D(dst_stereo_point, sigma_cov,
                                                 pose_keys[dst_frame_id], cur_q_symbol, k)

        # adding q-src factor
        src_stereo_point = gtsam.StereoPoint2(left_src_cords[0], right_src_cords[0],
                                              left_src_cords[1])
        sigma_cov = gtsam.noiseModel.Diagonal.Sigmas(np.ones(3))
        src_factor = gtsam.GenericStereoFactor3D(src_stereo_point, sigma_cov,
                                                 pose_keys[src_frame_id], cur_q_symbol, k)

        # adding factors to the graph
        graph.add(dst_factor)
        graph.add(src_factor)

    # adding prior to the first pose
    sigma_6d = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6))
    origin = gtsam.Pose3()
    prior_factor = gtsam.PriorFactorPose3(pose_keys[src_frame_id], origin, sigma_6d)
    graph.add(prior_factor)

    # running optimization
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_vals)
    result = optimizer.optimize()

    # computing the optimized relative transformation
    opt_src_pose = result.atPose3(pose_keys[src_frame_id])
    opt_dst_pose = result.atPose3(pose_keys[dst_frame_id])
    # opt_relative_pose = opt_src_pose.between(opt_dst_pose)
    opt_relative_pose = opt_dst_pose.between(opt_src_pose)

    # returning the relative covariance
    all_cov = gtsam.Marginals(graph, result)
    return compute_relative_cov_mat(all_cov, pose_keys, src_frame_id, dst_frame_id), \
        opt_relative_pose


def filter_y_dist(matches, kps1, kps2, dist_threshold=2):
    """
    This function filters matches by Y cords distance.
    :param matches: matches.
    :param kps1: kps1.
    :param kps2: kps2.
    :param dist_threshold: filter threshold.
    :return: filtered mathces.
    """
    filter_kps1 = np.squeeze(kps1[matches[:, 0], :])
    filter_kps2 = np.squeeze(kps2[matches[:, 1], :])

    y_dist_vec = np.abs(filter_kps1[:, 1] - filter_kps2[:, 1])
    keep_ind = np.argwhere(y_dist_vec <= dist_threshold).flatten()

    return matches[keep_ind, :], filter_kps1[keep_ind, :], filter_kps2[keep_ind, :]


def compute_frame_cloud_cnn(left_img_vals: tuple, right_img_vals: tuple):
    """
    This function computes the 3D point cloud coordinates of frame image pair.
    :param left_img_vals: tuple of: (keypoints, descriptors), matching the left image.
    :param right_img_vals: tuple of: (keypoints, descriptors), matching the right image.
    :return: cloud coordinates, kept matches, discarded matches.
    """
    matches_lst = Frame.cnn_detector.match_descriptors(left_img_vals[1],
                                                       right_img_vals[1])

    # filtering matches using vertical distance test
    keep_matches, left_cords, right_cords = filter_y_dist(
        matches_lst, left_img_vals[0], right_img_vals[0])

    left_cords = left_cords.T
    right_cords = right_cords.T
    discard_matches = None

    # compute triangulation cloud
    k, m1, m2 = get_calib()
    frame_cloud = triangulate_cloud(k, m1, m2, left_cords, right_cords)
    # frame_cloud = cv.triangulatePoints(k @ m1, k @ m2, left_cords, right_cords)
    frame_cloud = frame_cloud / frame_cloud[-1, :]

    return frame_cloud, keep_matches, discard_matches
