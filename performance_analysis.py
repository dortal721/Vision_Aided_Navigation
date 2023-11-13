from VanCollections import TracksData, Frame
from ex_code.slam_ex4 import process_tracks_stats, get_connectivity_data
from matplotlib import pyplot as plt
import numpy as np
import pickle
from functional import VanFunctional as F
import cv2 as cv
from tqdm import tqdm

PNP_REF_ERR = {}
NUM_FRAMES = 2560
TRACKING_IMGS_PATH = 'tracking_imgs'
PLOTS_PATH = 'final_submission_plots'


def create_tracking_video(tracks_data: TracksData):
    """
    This function creates a tracking video following different results throught the
    process.
    :param tracks_data: tracks data object.
    :return: None
    """
    positions_cords, gt_cords = None, None
    gt_lst = F.get_camera_ground_truth()

    for frame_ind in tqdm(range(NUM_FRAMES - 1), desc='processing video frames'):
        # save current
        cur_frame_obj = Frame(frame_ind)
        left_vals, right_vals = cur_frame_obj.get_key_points_and_descriptors()

        left_kps, left_desc = left_vals
        right_kps, right_desc = right_vals

        matches = Frame.cnn_detector.match_descriptors(left_desc, right_desc)
        matches, left_kps, right_kps = F.filter_y_dist(matches, left_kps, right_kps)

        fig, ax = plt.subplots(2, 2)
        fig.tight_layout()
        ax[0][0].imshow(cur_frame_obj.left_frame, cmap='gray')
        ax[0][0].scatter(left_kps[:, 0], left_kps[:, 1], s=0.5)
        ax[0][0].set_title('Left Camera')

        ax[0][1].imshow(cur_frame_obj.right_frame, cmap='gray')
        ax[0][1].scatter(right_kps[:, 0], right_kps[:, 1], s=0.5)
        ax[0][1].set_title('Right Camera')

        # obtaining current position
        cur_pos_mat = tracks_data.get_window_optimized_pose(frame_ind)
        cur_pos = F.get_camera_location(cur_pos_mat)
        cur_gt = F.get_camera_location(gt_lst[frame_ind])

        if positions_cords is None:
            positions_cords = cur_pos
            gt_cords = cur_gt
        else:
            positions_cords = np.hstack([positions_cords, cur_pos])
            gt_cords = np.hstack([gt_cords, cur_gt])

        ax[1][0].plot(positions_cords[0, :], positions_cords[2, :], label='prediction')
        ax[1][0].plot(gt_cords[0, :], gt_cords[2, :], label='ground truth')
        ax[1][0].set_xlim(-300, 300)
        ax[1][0].set_ylim(-160, 160)
        ax[1][0].set_title('Trajectory')
        ax[1][0].legend()

        # obtaining 3d cloud
        cur_3d_cloud, keep, discard = F.compute_frame_cloud(left_vals, right_vals)

        # filter extreme points
        non_negative_z_ind = np.argwhere(cur_3d_cloud[2, :] >= 0).flatten()
        non_extreme_z_ind = np.argwhere(cur_3d_cloud[2, :] <= 300).flatten()
        keep_3d = np.intersect1d(non_negative_z_ind, non_extreme_z_ind)
        cur_3d_cloud = cur_3d_cloud[:, keep_3d]

        ax[1][1].scatter(cur_3d_cloud[0, :], cur_3d_cloud[2, :])
        ax[1][1].set_title('2D points map')

        plt.suptitle(f'frame #{frame_ind}')
        fig.tight_layout()
        plt.savefig(f'{TRACKING_IMGS_PATH}/{frame_ind}.png')
        plt.close()

    # reading all saved images and stacking them to video
    out_video = cv.VideoWriter('tracking_vid.avi', cv.VideoWriter_fourcc(*'DIVX'), 15, (640, 480))

    for frame_ind in range(NUM_FRAMES - 1):
        cur_vid_frame = cv.imread(f'{TRACKING_IMGS_PATH}/{frame_ind}.png')
        out_video.write(cur_vid_frame)

    out_video.release()


def compare_3d_maps():
    """
    This function plots a comparison between the 3d maps of SIFT and ALIKE.
    :return: None.
    """
    frame0 = Frame(0)

    # getting keypoints and descriptors computed by ALIKE
    cnn_left_vals, cnn_right_vals = frame0.get_key_points_and_descriptors()

    # keypoints and descriptors computed by SIFT
    sift_obj = cv.SIFT_create()
    sift_left_vals, sift_right_vals = frame0.get_key_points_and_descriptors(sift_obj)

    # computing 3d point clouds
    sift_cloud, sift_keep_matches, sift_discard = F.compute_frame_cloud(sift_left_vals,
                                                                        sift_right_vals)
    cnn_cloud, cnn_keep_matches, cnn_discard = F.compute_frame_cloud_cnn(cnn_left_vals,
                                                                         cnn_right_vals)
    # filtering extreme 3d points
    non_negative_ind = np.argwhere(sift_cloud[2, :] >= 0).flatten()
    non_extreme_ind = np.argwhere(sift_cloud[2, :] <= 300).flatten()
    keep_ind = np.intersect1d(non_negative_ind, non_extreme_ind)
    sift_cloud = sift_cloud[:, keep_ind]

    non_negative_ind = np.argwhere(cnn_cloud[2, :] >= 0).flatten()
    non_extreme_ind = np.argwhere(cnn_cloud[2, :] <= 300).flatten()
    keep_ind = np.intersect1d(non_negative_ind, non_extreme_ind)
    cnn_cloud = cnn_cloud[:, keep_ind]

    # plotting a comparison of the cloud
    plt.scatter(cnn_cloud[0, :], cnn_cloud[2, :], label='ALIKE')
    plt.scatter(sift_cloud[0, :], sift_cloud[2, :], label='SIFT', alpha=0.5)
    plt.xlabel('X axis')
    plt.ylabel('Z axis')
    plt.legend()

    plt.title('Triangulation results comparison')
    plt.savefig(f'{PLOTS_PATH}/compare_3d_maps.png')
    plt.show()


def compare_statistics(sift_tracks: TracksData, cnn_track: TracksData):
    """
    This function prints a comparison of the tracking statistics between ALIKE and SIFT.
    :param sift_tracks: tracks data created by SIFT.
    :param cnn_track: tracks data created by ALIKE.
    :return: None
    """
    print(f'*** SIFT tracks stats ***')
    process_tracks_stats(sift_tracks)
    print('')

    print(f'*** ALIKE tracks stats ***')
    process_tracks_stats(cnn_track)
    print('')


def compare_connectivity(sift_tracks: TracksData, cnn_track: TracksData):
    """
    This function plots a comparison between the connectivity obtained by sift and ALIKE.
    :param sift_tracks: tracks data created by SIFT.
    :param cnn_track: tracks data created by ALIKE.
    :return: None.
    """
    sift_connectivity = get_connectivity_data(sift_tracks)
    cnn_connectivity = get_connectivity_data(cnn_track)
    x_data = np.arange(len(sift_connectivity))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x_data, sift_connectivity, label='SIFT', alpha=0.5)
    ax.plot(x_data, cnn_connectivity, label='ALIKE')
    plt.title('Connectivity')
    plt.xlabel('Frame id')
    plt.ylabel('outgoing tracks')
    plt.legend()

    fig.tight_layout()
    plt.savefig(f'{PLOTS_PATH}/compare_connectivity.png')
    plt.show()


def compare_inliers_rate(sift_tracks: TracksData, cnn_tracks: TracksData):
    """
    This function plots a comparison between the inliers rate obtained by SIFT and the one
    obtained by ALIKE.
    :param sift_tracks: tracks data created by SIFT.
    :param cnn_tracks: tracks data created by ALIKE.
    :return: None.
    """
    sift_inliers = sift_tracks.get_supporters_data()
    cnn_inliers = cnn_tracks.get_supporters_data()
    x_data = np.array(list(sift_inliers.keys()))

    sift_inliers = np.array(list(sift_inliers.values())) * 100
    cnn_inliers = np.array(list(cnn_inliers.values())) * 100

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x_data, sift_inliers, label='SIFT', alpha=0.5)
    ax.plot(x_data, cnn_inliers, label='ALIKE')
    plt.title('Inliers percentage per frame')
    plt.xlabel('Frame id')
    plt.ylabel('Inliers percentage')
    plt.legend()

    fig.tight_layout()
    plt.savefig(f'{PLOTS_PATH}/compare_inliers_rate.png')
    plt.show()


def compare_track_length(sift_tracks: TracksData, cnn_tracks: TracksData):
    """
    This function plots comparison of the track length histograms obtained by SIFT and
    ALIKE.
    :param sift_tracks: tracks data created by SIFT.
    :param cnn_tracks: tracks data created by ALIKE.
    :return: None.
    """
    sift_df = sift_tracks.cords_data[['TrackId', 'FrameId']]
    sift_df = sift_df.groupby('TrackId').count()
    sift_df.reset_index(None, drop=False, inplace=True)
    sift_df = sift_df.groupby('FrameId').count()

    cnn_df = cnn_tracks.cords_data[['TrackId', 'FrameId']]
    cnn_df = cnn_df.groupby('TrackId').count()
    cnn_df.reset_index(None, drop=False, inplace=True)
    cnn_df = cnn_df.groupby('FrameId').count()

    sift_x, sift_y = [0], [0]
    sift_x.extend(list(sift_df.index))
    sift_y.extend(list(sift_df['TrackId']))

    cnn_x, cnn_y = [0], [0]
    cnn_x.extend(list(cnn_df.index))
    cnn_y.extend(list(cnn_df['TrackId']))
    cnn_y = np.array(cnn_y)

    # padding sift data with zeros
    temp_sift_y = np.zeros_like(cnn_y)
    sift_y = np.array(sift_y)
    temp_sift_y[: sift_y.shape[0]] = sift_y
    sift_y = temp_sift_y

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(cnn_x, sift_y, label='SIFT', alpha=0.5)
    ax.plot(cnn_x, cnn_y, label='ALIKE')

    plt.xlabel('Track length')
    plt.ylabel('Track #')
    plt.title('Track length histogram')
    plt.legend()

    fig.tight_layout()
    plt.savefig(f'{PLOTS_PATH}/compare_track_length.png')
    plt.show()


def plot_ba_window_err(err_dict: dict):
    """
    This function plots the mean factor error per bundle window. The plot compares between
    the results obtained by an optimized and non optimized ALIKE models.
    :param err_dict: dictionary containing pre-computed factor errors.
    :return: None.
    """
    x_data = np.array(list(err_dict['pre_opt'].keys()))
    pre_opt_data = np.array(list(err_dict['pre_opt'].values()))
    opt_data = np.array(list(err_dict['optimized'].values()))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x_data, pre_opt_data, label='pre-opt')
    ax.plot(x_data, opt_data, label='optimized')
    plt.title('BA optimization error')
    plt.xlabel('First keyframe id')
    plt.ylabel('Mean Window factor err')
    plt.legend()

    fig.tight_layout()
    plt.savefig(f'{PLOTS_PATH}/plot_ba_window_err.png')
    plt.show()


def convert_left_ext_to_right(left_ext: np.ndarray, m2_mat: np.ndarray):
    """
    This function computes the extrinsic matrix of a right camera using the extrinsic
    matrix of the left camera.
    :param left_ext: left camera extrinsic matrix.
    :param m2_mat: M2 matrix provided by KITTI.
    :return: right extrinsic matrix.
    """
    cur_right_rotation = left_ext[:, :-1]
    cur_right_translation = (left_ext[:, -1] + m2_mat[:, -1]).reshape(3, 1)

    return np.hstack([cur_right_rotation, cur_right_translation])


def compute_bundle_projection_err(tracks_data: TracksData, first_kf: int, last_kf: int):
    """
    This function computes the median projection error per bundle.
    :param tracks_data: track data object (preferably ALIKE data).
    :param first_kf: bundle's first key frame.
    :param last_kf: bundle's last key frame.
    :return: median initial errors, median optimized errors.
    """
    tracks_id = tracks_data.get_overlapping_range_tracks(first_kf, last_kf)
    init_err_vec, opt_err_vec = [], []
    k, m1, m2 = F.get_calib()
    gt_lst = F.get_camera_ground_truth()

    # obtaining ground truth positions for last kf
    last_gt_left = gt_lst[last_kf]
    last_gt_right = convert_left_ext_to_right(last_gt_left, m2)

    # computing projection errors for each track
    for track_id in tracks_id:
        last_track_cords = tracks_data.get_track_frame_cords(last_kf, track_id)
        last_left_cords = (last_track_cords[0], last_track_cords[2])
        last_right_cords = (last_track_cords[1], last_track_cords[2])

        track_3d_cords = F.compute_triangulation(k, last_gt_left, last_gt_right,
                                                 last_left_cords, last_right_cords)
        track_3d_cords = track_3d_cords.reshape(4, 1)
        track_3d_cords = track_3d_cords / track_3d_cords[-1, :]

        for frame_id in range(first_kf, last_kf + 1):
            # computing left and right positions
            cur_init_left_pos = tracks_data.get_camera_location(frame_id)
            cur_init_right_pos = convert_left_ext_to_right(cur_init_left_pos, m2)

            cur_opt_left_pos = tracks_data.get_window_optimized_pose(frame_id)
            cur_opt_right_pos = convert_left_ext_to_right(cur_opt_left_pos, m2)

            # obtaining track cords
            cur_track_cords = tracks_data.get_track_frame_cords(frame_id, track_id)
            cur_left_img_cords = (cur_track_cords[0], cur_track_cords[2])
            cur_right_img_cords = (cur_track_cords[1], cur_track_cords[2])

            # projecting using init matrix
            left_init_proj = k @ cur_init_left_pos @ track_3d_cords
            left_init_proj = left_init_proj[:-1, :] / left_init_proj[-1, :]
            right_init_proj = k @ cur_init_right_pos @ track_3d_cords
            right_init_proj = right_init_proj[:-1, :] / right_init_proj[-1, :]

            # projecting using optimized matrix
            left_opt_proj = k @ cur_opt_left_pos @ track_3d_cords
            left_opt_proj = left_opt_proj[:-1, :] / left_opt_proj[-1, :]
            right_opt_proj = k @ cur_opt_right_pos @ track_3d_cords
            right_opt_proj = right_opt_proj[:-1, :] / right_opt_proj[-1, :]

            # converting cords to np arrays
            cur_left_img_cords = np.array(cur_left_img_cords).reshape(2, 1)
            cur_right_img_cords = np.array(cur_right_img_cords).reshape(2, 1)

            # computing both errors
            cur_init_err = np.linalg.norm(cur_left_img_cords - left_init_proj) + \
                           np.linalg.norm(cur_right_img_cords - right_init_proj)
            cur_opt_err = np.linalg.norm(cur_left_img_cords - left_opt_proj) + \
                          np.linalg.norm(cur_right_img_cords - right_opt_proj)

            # updating the lists
            init_err_vec.append(cur_init_err / 2)
            opt_err_vec.append(cur_opt_err / 2)

    # computing the medians
    init_median = np.median(np.array(init_err_vec))
    opt_median = np.median(np.array(opt_err_vec))

    return init_median, opt_median


def get_projection_ref_err(tracks_data: TracksData, pos_getter):
    """
    This function computes the projection error of a track as a function of the distance
    from the reference (last) frame.
    :param tracks_data: track data object (preferably ALIKE data).
    :param pos_getter: a callable function (getter) that gets positions from tracks_data.
    :return: dictionary mapping the results by distance.
    """
    num_tracks = len(tracks_data.cords_data['TrackId'].unique())
    # num_tracks = 10000
    tracks_len_df = tracks_data.cords_data[['TrackId', 'FrameId']].groupby('TrackId').count()
    max_len = tracks_len_df.max().item()
    k, m1, m2 = F.get_calib()
    gt_lst = F.get_camera_ground_truth()

    # creating lengths dictionary
    len_dict = {}
    skipped_frames = []

    for i in range(0, max_len + 1):
        len_dict.update({i: {'left': [], 'right': []}})

    # iterating over all tracks to compute the projection errors
    for track_id in tqdm(range(num_tracks), desc='computing proj ref'):
        skip_track = False

        # loading ground truth positions
        frames_lst = tracks_data.get_frames_by_track(track_id)
        last_gt_left = pos_getter(frames_lst[-1])

        if last_gt_left is None:
            continue

        last_gt_right = convert_left_ext_to_right(last_gt_left, m2)

        # computing 3d cords using gt
        last_track_cords = tracks_data.get_track_frame_cords(frames_lst[-1], track_id)
        last_left_cords = (last_track_cords[0], last_track_cords[2])
        last_right_cords = (last_track_cords[1], last_track_cords[2])

        track_3d_cords = F.compute_triangulation(k, last_gt_left, last_gt_right,
                                                 last_left_cords, last_right_cords)
        track_3d_cords = track_3d_cords.reshape(4, 1)
        track_3d_cords = track_3d_cords / track_3d_cords[-1, :]

        for frame_ind in frames_lst:
            cur_ref_dist = frames_lst[-1] - frame_ind
            cur_left_pos = pos_getter(frame_ind)

            if cur_left_pos is None:
                skip_track = True
                break

            cur_right_pos = convert_left_ext_to_right(cur_left_pos, m2)

            # obtaining track cords
            cur_track_cords = tracks_data.get_track_frame_cords(frame_ind, track_id)
            cur_left_img_cords = (cur_track_cords[0], cur_track_cords[2])
            cur_right_img_cords = (cur_track_cords[1], cur_track_cords[2])

            # projecting using pos matrices
            left_proj = k @ cur_left_pos @ track_3d_cords
            left_proj = left_proj[:-1, :] / left_proj[-1, :]
            right_proj = k @ cur_right_pos @ track_3d_cords
            right_proj = right_proj[:-1, :] / right_proj[-1, :]

            # converting cords to np arrays
            cur_left_img_cords = np.array(cur_left_img_cords).reshape(2, 1)
            cur_right_img_cords = np.array(cur_right_img_cords).reshape(2, 1)

            # computing errors
            cur_left_err = np.linalg.norm(cur_left_img_cords - left_proj)
            cur_right_err = np.linalg.norm(cur_right_img_cords - right_proj)

            # updating errors in dict
            cur_sub_dict = len_dict[cur_ref_dist]
            cur_sub_dict['left'].append(cur_left_err)
            cur_sub_dict['right'].append(cur_right_err)

        if skip_track:
            continue

    print(skipped_frames)
    return len_dict


def get_bundle_projection_err(tracks_data: TracksData):
    """
    This function computes the projection error as a function of the key frame after bundle
    optimization and loop closure optimization.
    :param tracks_data: track data object (preferably ALIKE data).
    :return: errors table.
    """
    num_bundle_windows = tracks_data.get_num_bundle_windows()
    err_table = np.empty((num_bundle_windows, 3))

    for bundle_id in tqdm(range(num_bundle_windows), desc='computing bundle proj'):
        first_kf, last_kf = tracks_data.get_bundle_bounds(bundle_id)

        init_median, opt_median = compute_bundle_projection_err(tracks_data, first_kf,
                                                                last_kf)

        err_table[bundle_id, 0] = first_kf
        err_table[bundle_id, 1] = init_median
        err_table[bundle_id, 2] = opt_median

    return err_table


def plot_bundle_projection_err(tracks_data: TracksData):
    """
    This function plots the projection error as a function of the key frame after bundle
    optimization and loop closure optimization.
    :param tracks_data: track data object (preferably ALIKE data).
    :return: None.
    """
    err_table = get_bundle_projection_err(tracks_data)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(err_table[:, 0], err_table[:, 1], label='pre-opt')
    ax.plot(err_table[:, 0], err_table[:, 2], label='optimized')
    plt.title('')
    plt.legend()

    plt.title('Projection error for Bundle window')
    plt.xlabel('Key frame')
    plt.ylabel('Median projection error')

    fig.tight_layout()
    plt.savefig(f'{PLOTS_PATH}/bundle_projection_err.png')
    plt.show()


def plot_compare_projection_err(sift_tracks: TracksData, cnn_tracks: TracksData):
    """
    This function plots a comparison of the median bundle window projection error between
    SIFT and ALIKE.
    :param sift_tracks: tracks data created by SIFT.
    :param cnn_tracks: tracks data created by ALIKE.
    :return: None.
    """
    sift_err_table = get_bundle_projection_err(sift_tracks)
    cnn_err_table = get_bundle_projection_err(cnn_tracks)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(sift_err_table[:, 0], sift_err_table[:, 2], label='SIFT-optimized')
    ax.plot(cnn_err_table[:, 0], cnn_err_table[:, 2], label='ALIKE-optimized')
    plt.title(f'projection error comparison')
    plt.xlabel('Key Frame')
    plt.ylabel('Mean projection error')
    plt.legend()

    fig.tight_layout()
    plt.savefig(f'{PLOTS_PATH}/compare_projection_err.png')
    plt.show()


def plot_pnp_projection_ref(tracks_data: TracksData):
    """
    This function plots a comparison between the median projection error of different
    track links as a function of the distance from the reference frame - both for PnP
    and LC.
    :param tracks_data: track data object (preferably ALIKE data).
    :return: None
    """
    pnp_err = get_projection_ref_err(tracks_data, tracks_data.get_camera_location)
    ba_err = get_projection_ref_err(tracks_data, tracks_data.get_window_optimized_pose)

    pnp_left_medians, pnp_right_medians = {}, {}
    ba_left_medians, ba_right_medians = {}, {}

    # computing medians
    for cur_ref_dist in pnp_err.keys():
        cur_pnp_dict = pnp_err.get(cur_ref_dist)
        left_pnp_err_vec = np.array(cur_pnp_dict['left'])
        right_pnp_err_vec = np.array(cur_pnp_dict['right'])

        cur_ba_dict = ba_err.get(cur_ref_dist)
        left_ba_err_vec = np.array(cur_ba_dict['left'])
        right_ba_err_vec = np.array(cur_ba_dict['right'])

        if len(left_pnp_err_vec) == 0 or len(right_pnp_err_vec) == 0:
            continue

        pnp_left_medians.update({cur_ref_dist: np.median(left_pnp_err_vec)})
        pnp_right_medians.update({cur_ref_dist: np.median(right_pnp_err_vec)})

        ba_left_medians.update({cur_ref_dist: np.median(left_ba_err_vec)})
        ba_right_medians.update({cur_ref_dist: np.median(right_ba_err_vec)})

    # plotting medians
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    ax[0].plot(list(pnp_left_medians.keys()), list(pnp_left_medians.values()), label='left')
    ax[0].plot(list(pnp_right_medians.keys()), list(pnp_right_medians.values()), label='right')
    ax[0].set_title('PnP ref projection')
    ax[0].set_xlabel('Distance from reference (last) frame')
    ax[0].set_ylabel('Median projection error')
    ax[0].legend()

    ax[1].plot(list(ba_left_medians.keys()), list(ba_left_medians.values()), label='left')
    ax[1].plot(list(ba_right_medians.keys()), list(ba_right_medians.values()), label='right')
    ax[1].set_title('BA ref projection')
    ax[1].set_xlabel('Distance from reference (last) frame')
    ax[1].set_ylabel('Median projection error')
    ax[1].legend()

    plt.suptitle('Median projection error VS reference distance')
    fig.tight_layout()
    plt.savefig(f'{PLOTS_PATH}/pnp_projection_ref.png')
    plt.show()


def split_to_sequences(seq_len: int):
    """
    This function splits the trajectory into sequences of desired length.
    :param seq_len: sequences length.
    :return: split frames list.
    """
    frames_lst = [0]
    cur_frame = 0

    while cur_frame < NUM_FRAMES:
        cur_frame += seq_len
        frames_lst.append(min(cur_frame, NUM_FRAMES - 2))

    return frames_lst


def compute_relative_error(tracks_data: TracksData, seq_len: int, pos_getter):
    """
    This function computes the relative error (pnp or ba) split to sequences.
    :param tracks_data: track data object (preferably ALIKE data).
    :param seq_len: sequence length.
    :param pos_getter: callable object (getter) that gets postion from tracks_data.
    :return: relative errors.
    """
    seq_lst = split_to_sequences(seq_len)
    gt_lst = F.get_camera_ground_truth()
    res_table = np.empty((len(seq_lst) - 1, 3))

    for i in range(len(seq_lst) - 1):
        first_frame, last_frame = seq_lst[i], seq_lst[i + 1]

        first_pos = pos_getter(first_frame)
        last_pos = pos_getter(last_frame)
        pnp_relative = F.compute_relative_ext(first_pos, last_pos)

        first_gt = gt_lst[first_frame]
        last_gt = gt_lst[last_frame]
        gt_relative = F.compute_relative_ext(first_gt, last_gt)

        # computing the relative error
        cur_location_err = np.linalg.norm(pnp_relative[:, -1] - gt_relative[:, -1], ord=1)

        cur_angle_err = np.linalg.norm(cv.Rodrigues(pnp_relative[:, :-1])[0] -
                                       cv.Rodrigues(gt_relative[:, :-1])[0], ord=1)

        total_traveled = np.linalg.norm(last_gt[:, -1] - first_gt[:, -1], ord=1)

        # updating the error in the table
        res_table[i, 0] = first_frame
        res_table[i, 1] = cur_location_err / total_traveled
        res_table[i, 2] = cur_angle_err / total_traveled

    return res_table


def plot_relative_pnp_err(tracks_data: TracksData):
    """
    This function plots the relative pnp error.
    :param tracks_data: track data object (preferably ALIKE data).
    :return: None.
    """
    seq_results = {100: 'red', 300: 'orange', 800: 'blue'}
    fig, ax = plt.subplots(1, 2, figsize=(24, 8))

    # computing the results for different lengths
    for cur_len in seq_results.keys():
        cur_color = seq_results.get(cur_len)
        cur_res = compute_relative_error(tracks_data, cur_len,
                                         tracks_data.get_camera_location)

        ax[0].plot(cur_res[:, 0], cur_res[:, 1], label=cur_len, c=cur_color)
        ax[1].plot(cur_res[:, 0], cur_res[:, 2], label=cur_len, c=cur_color)

        cur_location_avg = cur_res[:, 1].mean()
        cur_angle_avg = cur_res[:, 2].mean()

        # plotting the averages
        ax[0].plot(cur_res[:, 0], np.ones_like(cur_res[:, 0]) * cur_location_avg,
                   label=f'{cur_len} AVG', linestyle='dashed', c=cur_color)
        ax[1].plot(cur_res[:, 0], np.ones_like(cur_res[:, 0]) * cur_angle_avg,
                   label=f'{cur_len} AVG', linestyle='dashed', c=cur_color)

    ax[0].set_title('Relative location error')
    ax[0].set_xlabel('Frame')
    ax[0].set_ylabel('Error (m/m)')

    ax[1].set_title('Relative angle error')
    ax[1].set_xlabel('Frame')
    ax[1].set_ylabel('Error (deg/m)')

    ax[0].legend()
    ax[1].legend()

    plt.suptitle('Relative PnP estimation')
    fig.tight_layout()
    plt.savefig(f'{PLOTS_PATH}/relative_pnp_err.png')
    plt.show()


def plot_relative_ba_err(tracks_data: TracksData):
    """
    This function plots the relative BA error.
    :param tracks_data: track data object (preferably ALIKE data).
    :return: None.
    """
    seq_results = {100: 'red', 300: 'orange', 800: 'blue'}
    fig, ax = plt.subplots(1, 2, figsize=(24, 8))

    # computing the results for different lengths
    for cur_len in seq_results.keys():
        cur_color = seq_results.get(cur_len)
        cur_res = compute_relative_error(tracks_data, cur_len,
                                         tracks_data.get_window_optimized_pose)

        ax[0].plot(cur_res[:, 0], cur_res[:, 1], label=cur_len, c=cur_color)
        ax[1].plot(cur_res[:, 0], cur_res[:, 2], label=cur_len, c=cur_color)

        cur_location_avg = cur_res[:, 1].mean()
        cur_angle_avg = cur_res[:, 2].mean()

        # plotting the averages
        ax[0].plot(cur_res[:, 0], np.ones_like(cur_res[:, 0]) * cur_location_avg,
                   label=f'{cur_len} AVG', linestyle='dashed', c=cur_color)
        ax[1].plot(cur_res[:, 0], np.ones_like(cur_res[:, 0]) * cur_angle_avg,
                   label=f'{cur_len} AVG', linestyle='dashed', c=cur_color)

    ax[0].set_title('Relative location error')
    ax[0].set_xlabel('Frame')
    ax[0].set_ylabel('Error (m/m)')

    ax[1].set_title('Relative angle error')
    ax[1].set_xlabel('Frame')
    ax[1].set_ylabel('Error (deg/m)')

    ax[0].legend()
    ax[1].legend()

    plt.suptitle('Relative BA estimation')
    fig.tight_layout()
    plt.savefig(f'{PLOTS_PATH}/relative_ba_err.png')
    plt.show()


def plot_compare_lc_match_count(sift_stats: dict, cnn_stats: dict):
    """
    This function compares loop closure match number between SIFT and ALIKE.
    :param sift_stats: dict created by SIFT.
    :param cnn_stats: dict created by ALIKE.
    :return: None.
    """
    sift_count = sift_stats['match_count']
    cnn_count = cnn_stats['match_count']

    sift_x = np.arange(1, len(sift_count) + 1)
    cnn_x = np.arange(1, len(cnn_count) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(24, 8))
    ax[0].plot(cnn_x, cnn_count)
    ax[0].set_title('ALIKE loop closure match count')
    ax[0].set_xlabel('#Loop closure')
    ax[0].set_ylabel('#Matches')

    ax[1].plot(sift_x, sift_count)
    ax[1].set_title('SIFT loop closure match count')
    ax[1].set_xlabel('#Loop closure')
    ax[1].set_ylabel('#Matches')

    fig.tight_layout()
    plt.savefig(f'{PLOTS_PATH}/compare_lc_match_count.png')
    plt.show()


def plot_compare_lc_inliers(sift_stats: dict, cnn_stats: dict):
    """
    This function compares loop closure inliers rate between SIFT and ALIKE.
    :param sift_stats: dict created by SIFT.
    :param cnn_stats: dict created by ALIKE.
    :return: None.
    """
    sift_rate = np.array(sift_stats['inliers_rate']) * 100
    cnn_rate = np.array(cnn_stats['inliers_rate']) * 100

    sift_x = np.arange(1, len(sift_rate) + 1)
    cnn_x = np.arange(1, len(cnn_rate) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    ax[0].plot(cnn_x, cnn_rate)
    ax[0].set_title('ALIKE loop closure inliers rate')
    ax[0].set_xlabel('#Loop closure')
    ax[0].set_ylabel('Inliers rate')

    ax[1].plot(sift_x, sift_rate)
    ax[1].set_title('SIFT loop closure inliers rate')
    ax[1].set_xlabel('#Loop closure')
    ax[1].set_ylabel('Inliers rate')

    fig.tight_layout()
    plt.savefig(f'{PLOTS_PATH}/compare_lc_inliers.png')
    plt.show()


def plot_compare_lc_stats():
    """
    This function plots a comparison between the loop closure statistics between SIFT and
    ALIKE.
    :return: None.
    """
    with open('ex7_lc_stats.pkl', 'rb') as f1:
        sift_stats = pickle.load(f1)

    with open('final_submission_lc_stats.pkl', 'rb') as f1:
        cnn_stats = pickle.load(f1)

    plot_compare_lc_match_count(sift_stats, cnn_stats)
    plot_compare_lc_inliers(sift_stats, cnn_stats)


def get_abs_estimation_err(tracks_data: TracksData, pos_getter):
    """
    This function computes the absolute error.
    :param tracks_data: track data object (preferably ALIKE data).
    :param pos_getter: callable object (getter) that gets postion from tracks_data.
    :return: error table.
    """
    gt_lst = F.get_camera_ground_truth()
    err_table = None

    for frame_ind in range(NUM_FRAMES - 1):
        cur_gt_mat = gt_lst[frame_ind]
        cur_pos_mat = pos_getter(frame_ind)

        gt_cords = F.get_camera_location(cur_gt_mat)
        pos_cords = F.get_camera_location(cur_pos_mat)
        location_err = np.abs(gt_cords - pos_cords).reshape(3, 1)

        gt_angle = cv.Rodrigues(cur_gt_mat[:, :-1])[0]
        pos_angle = cv.Rodrigues(cur_pos_mat[:, :-1])[0]
        angle_err = np.abs(gt_angle - pos_angle).reshape(3, 1)

        norm_err = np.linalg.norm(cur_gt_mat - cur_pos_mat)

        err_vec = np.vstack([angle_err, location_err, norm_err])

        if err_table is None:
            err_table = err_vec
        else:
            err_table = np.hstack([err_table, err_vec])

    return err_table


def plot_abs_estimation_err(tracks_data: TracksData, pos_getter, loop_closure=True):
    """
    This function plots the absolute estimation error.
    :param tracks_data: track data object (preferably ALIKE data).
    :param pos_getter: callable object (getter) that gets postion from tracks_data.
    :param loop_closure: is loop closure included.
    :return: None.
    """
    plot_type = 'Pose Graph'
    lc = 'without'

    if pos_getter == tracks_data.get_camera_location:
        plot_type = 'PnP'
        lc = ''
        loop_closure = False

    if loop_closure:
        lc = 'with'

    err_table = get_abs_estimation_err(tracks_data, pos_getter)
    x_frames = np.arange(NUM_FRAMES - 1)

    fig, ax = plt.subplots(1, 2, figsize=(24, 8))

    ax[1].plot(x_frames, err_table[0, :], label='alpha-angle')
    ax[1].plot(x_frames, err_table[1, :], label='beta-angle')
    ax[1].plot(x_frames, err_table[2, :], label='gamma-angle')
    ax[0].plot(x_frames, err_table[3, :], label='x')
    ax[0].plot(x_frames, err_table[4, :], label='y')
    ax[0].plot(x_frames, err_table[5, :], label='z')
    ax[0].plot(x_frames, err_table[6, :], label='norm')

    ax[1].set_title('Angle error')
    ax[1].set_xlabel('Frame')
    ax[1].set_ylabel('error')
    ax[1].set_ylim(0, 0.6)
    ax[1]. legend()

    ax[0].set_title('location + norm error')
    ax[0].set_xlabel('Frame')
    ax[0].set_ylabel('error')
    ax[0].legend()

    plt.suptitle(f'Abs {plot_type} {lc} loop closure estimation error')
    fig.tight_layout()
    plt.savefig(f'{PLOTS_PATH}/abs_estimation_err_{plot_type}_{lc}.png')
    plt.show()


def get_uncertainty_data(tracks_data: TracksData):
    """
    This function computes the uncertainty data.
    :param tracks_data: track data object (preferably ALIKE data).
    :return: xticks, location uncertainty, angle uncertainty.
    """
    location_vec, angle_vec = [], []
    frame_x = []

    for frame_ind, cov_mat in tracks_data.frame_marginals.items():
        cov_diag = cov_mat.diagonal().reshape(6, 1)
        frame_x.append(frame_ind)

        cur_location_val = np.linalg.norm(cov_diag[3:])
        cur_angle_val = np.linalg.norm(cov_diag[:3])

        location_vec.append(cur_location_val)
        angle_vec.append(cur_angle_val)

    frame_x = np.array(frame_x)
    location_vec = np.array(location_vec)
    angle_vec = np.array(angle_vec)

    return frame_x, location_vec, angle_vec


def plot_compare_uncertainty(lc_tracks_data: TracksData, non_lc_tracks_data: TracksData,
                             log_scale=True):
    """
    This function plots the comparison of uncertainty data.
    :param lc_tracks_data: tracks data obtained using LC.
    :param non_lc_tracks_data: track data obtained without LC.
    :param log_scale: should scale to log.
    :return: None.
    """
    lc_frame_x, lc_location_vec, lc_angle_vec = get_uncertainty_data(lc_tracks_data)
    non_lc_frame_x, non_lc_location_vec, non_lc_angle_vec = get_uncertainty_data(
        non_lc_tracks_data)
    log_str = ''

    if log_scale:
        lc_location_vec = np.log(lc_location_vec)
        non_lc_location_vec = np.log(non_lc_location_vec)
        lc_angle_vec = np.log(lc_angle_vec)
        non_lc_angle_vec = np.log(non_lc_angle_vec)
        log_str = ' log'

    fig, ax = plt.subplots(1, 2, figsize=(24, 8))
    ax[0].plot(lc_frame_x, lc_location_vec, label='pose graph with loop closure')
    ax[0].plot(non_lc_frame_x, non_lc_location_vec, label='pose graph without loop closure')
    ax[0].set_title('Location')
    ax[0].set_xlabel('Key Frame')
    ax[0].set_ylabel(f'Uncertainty{log_str}')
    ax[0].legend()

    ax[1].plot(lc_frame_x, lc_angle_vec, label='pose graph with loop closure')
    ax[1].plot(non_lc_frame_x, non_lc_angle_vec, label='pose graph without loop closure')
    ax[1].set_title('Angle')
    ax[1].set_xlabel('Key Frame')
    ax[1].set_ylabel(f'Uncertainty{log_str}')
    ax[1].legend()

    plt.suptitle('Uncertainty size vs keyframe')
    fig.tight_layout()
    plt.savefig(f'{PLOTS_PATH}/compare_uncertainty.png')
    plt.show()


def plot_match_count_per_frame():
    """
    This function plots the match count per frame
    :return: None.
    """
    with open('ALIKE_match_count.pkl', 'rb') as f1:
        res_dict = pickle.load(f1)

    x_data = list(res_dict.keys())
    count_data = list(res_dict.values())

    fig, ax = plt.subplots()
    ax.plot(x_data, count_data)
    ax.set_title('ALIKE - Number of matches per frame')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Match count')

    fig.tight_layout()
    plt.savefig(f'{PLOTS_PATH}/plot_match_count_per_frame.png')
    plt.show()


def plot_compare_trajectory(lc_tracks_data: TracksData, non_lc_tracks_data: TracksData):
    """
    This function plots the different trajectory estimations.
    :param lc_tracks_data: tracks data obtained using LC.
    :param non_lc_tracks_data: tracks data obtained without using LC.
    :return: None.
    """
    gt_lst = F.get_camera_ground_truth()
    initial_cords, ba_cords, lc_cords, gt_cords = None, None, None, None

    for frame_ind in range(NUM_FRAMES - 1):
        cur_gt = gt_lst[frame_ind]
        cur_init = lc_tracks_data.get_camera_location(frame_ind)
        cur_ba = non_lc_tracks_data.get_window_optimized_pose(frame_ind)
        cur_lc = lc_tracks_data.get_window_optimized_pose(frame_ind)

        cur_gt = F.get_camera_location(cur_gt)
        cur_init = F.get_camera_location(cur_init)
        cur_ba = F.get_camera_location(cur_ba)
        cur_lc = F.get_camera_location(cur_lc)

        if initial_cords is None:
            initial_cords = cur_init
            ba_cords = cur_ba
            lc_cords = cur_lc
            gt_cords = cur_gt
        else:
            initial_cords = np.hstack([initial_cords, cur_init])
            ba_cords = np.hstack([ba_cords, cur_ba])
            lc_cords = np.hstack([lc_cords, cur_lc])
            gt_cords = np.hstack([gt_cords, cur_gt])

    fig, ax = plt.subplots()
    ax.plot(initial_cords[0, :], initial_cords[2, :], label='PnP')
    ax.plot(ba_cords[0, :], ba_cords[2, :], label='BA')
    ax.plot(lc_cords[0, :], lc_cords[2, :], label='LC')
    ax.plot(gt_cords[0, :], gt_cords[2, :], label='GT')
    ax.set_title('Bird eye view of the trajectory')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Z axis')
    ax.legend()

    fig.tight_layout()
    plt.savefig(f'{PLOTS_PATH}/plot_compare_trajectory.png')
    plt.show()


if __name__ == '__main__':
    sift_data = TracksData.from_pickle('try1.pkl')
    cnn_data = TracksData.from_pickle('tracks_data.pkl')
    no_lc_cnn_data = TracksData.from_pickle('tracks_data_no_lc.pkl')

    create_tracking_video(cnn_data)
    compare_statistics(sift_data, cnn_data)
    compare_connectivity(sift_data, cnn_data)
    compare_inliers_rate(sift_data, cnn_data)
    compare_track_length(sift_data, cnn_data)

    # read errors dict
    with open('errs_dict.pkl', 'rb') as f:
        errors_dict = pickle.load(f)

    plot_ba_window_err(errors_dict['mean_ba_factor'])

    plot_bundle_projection_err(cnn_data)
    plot_compare_projection_err(sift_data, cnn_data)
    plot_pnp_projection_ref(cnn_data)
    plot_relative_pnp_err(cnn_data)
    plot_relative_ba_err(cnn_data)
    plot_compare_lc_stats()
    compare_3d_maps()
    plot_abs_estimation_err(cnn_data, cnn_data.get_camera_location, False)
    plot_abs_estimation_err(cnn_data, cnn_data.get_window_optimized_pose, False)
    plot_abs_estimation_err(cnn_data, cnn_data.get_window_optimized_pose, True)

    plot_compare_uncertainty(cnn_data, no_lc_cnn_data)
    plot_compare_trajectory(cnn_data, no_lc_cnn_data)
    plot_match_count_per_frame()
