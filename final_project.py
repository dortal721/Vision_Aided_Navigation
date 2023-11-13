from functional import VanFunctional, VanFunctional as F, VanCnnFunctional as fnn
from VanCollections import Frame, TracksData, CovGraph
import numpy as np
import time
import gtsam
import pickle

NUM_FRAMES = 2560
MATCH_STATS_DICT = {}  # used for saving match count per frame for analysis
np.random.seed(0)


def main():
    cur_k, cur_m1, cur_m2 = F.get_calib()
    pose_graph, init_vals = F.construct_pose_graph_objects()
    cov_graph = CovGraph()

    # read first frame
    cur_frame0 = Frame(0)
    left0_vals, right0_vals = cur_frame0.get_key_points_and_descriptors()
    cur_f0_cloud, cur_f0_keep, cur_f0_discard = F.compute_frame_cloud(left0_vals,
                                                                      right0_vals)
    MATCH_STATS_DICT.update({0: cur_f0_keep.shape[0]})

    # creating the pos tracking matrix
    current_left0_tracking = cur_m1

    # creating the tracking data frame
    track_df = TracksData(save_supporters_data=True)
    first_bundle_kf = 0

    # iterating over trajectory
    for frame_ind in range(1, NUM_FRAMES):
        start = time.time()

        # computing detection values for current frame
        cur_frame1 = Frame(frame_ind)
        left1_vals, right1_vals = cur_frame1.get_key_points_and_descriptors()
        cur_f1_cloud, cur_f1_keep, cur_f1_discard = F.compute_frame_cloud(left1_vals,
                                                                          right1_vals)
        MATCH_STATS_DICT.update({frame_ind: cur_f1_keep.shape[0]})

        # computing consecutive matches
        cur_cons_matches = Frame.cnn_detector.match_descriptors(
            left0_vals[1], left1_vals[1])

        # filter consecutive matches that don't exist in f0 keep
        cur_cons_matches = fnn.filter_consecutive_matches(cur_cons_matches, cur_f0_keep,
                                                          cur_f1_keep)

        # crossing matches to get multi match indices and matching 3d cloud
        cur_multi_match_ind = fnn.find_multi_matches(cur_cons_matches, cur_f0_keep,
                                                     cur_f1_keep)
        cur_multi_cloud = fnn.slice_cloud_by_multi_matches(
            cur_f0_cloud, cur_multi_match_ind, cur_f0_keep)

        # filtering 3d points with large or negative z values
        cur_multi_match_ind, cur_multi_cloud = fnn.filter_extreme_3d_points(
            cur_multi_match_ind, cur_multi_cloud)

        # slicing and arranging the keypoints cords
        left0_kps_cords, right0_kps_cords, left1_kps_cords, right1_kps_cords = \
            fnn.slice_kps_by_multi_matches(left0_vals[0], right0_vals[0], left1_vals[0],
                                           right1_vals[0], cur_multi_match_ind)
        # performing P3P ransac
        cur_supporters, cur_left1_ext = F.compute_ransac(cur_multi_cloud,
                                                         cur_multi_match_ind,
                                                         left1_vals[0],
                                                         cur_k, cur_m1, cur_m2,
                                                         left0_kps_cords,
                                                         right0_kps_cords,
                                                         left1_kps_cords,
                                                         right1_kps_cords)
        # updating track df
        track_df.insert_tracks(frame_ind - 1, cur_multi_match_ind[cur_supporters, :],
                               left0_vals[0], right0_vals[0], left1_vals[0],
                               right1_vals[0])

        # updating supporters data
        track_df.update_supporters_data(frame_ind, cur_supporters.shape[0],
                                        cur_multi_match_ind.shape[0])

        # saving current left cam position
        track_df.update_camera_location(frame_ind - 1, current_left0_tracking)

        # updating left0 matrix
        current_left0_tracking = F.compose_t_transform(current_left0_tracking,
                                                       cur_left1_ext)

        # updating frame1 to be frame0
        left0_vals, right0_vals = left1_vals, right1_vals
        cur_f0_cloud, cur_f0_keep, cur_f0_discard = cur_f1_cloud, cur_f1_keep, \
            cur_f1_discard

        # checking if bundle adjustment is necessary
        if frame_ind > 1:
            first_bundle_kf = fnn.handle_bundle_window(
                track_df, first_bundle_kf, frame_ind - 1, pose_graph, cov_graph,
                init_vals, bundle_dist=8)

        end = time.time()
        print(f'frame #{frame_ind} ; time = {end - start}')

    # updating marginals for analysis purposes
    track_df.update_pose_marginals_cov(pose_graph, init_vals)

    # plotting lc optimization only
    lc_ind_table = None

    for frame_ind in range(NUM_FRAMES):
        try:
            cur_lc_symbol = gtsam.symbol('c', frame_ind)
            cur_lc_pos = init_vals.atPose3(cur_lc_symbol).matrix()[: -1, :]
            cur_lc_pos = F.invert_extrinsic(cur_lc_pos)
            cur_lc_pos = F.get_camera_location(cur_lc_pos)
        except RuntimeError:
            continue

        if lc_ind_table is None:
            lc_ind_table = cur_lc_pos
        else:
            lc_ind_table = np.hstack([lc_ind_table, cur_lc_pos])

    lc_str = ''

    if not fnn.PERFORM_LOOP_CLOSURE:
        lc_str = f'_no_lc'

    track_df.to_pickle(f'tracks_data{lc_str}.pkl')


if __name__ == '__main__':
    main_start = time.time()
    main()
    main_end = time.time()

    print(f'Total runtime = {(main_end - main_start) / 60} minutes')

    # ----------------------------------------------- saving data for performance analysis
    # saving errors
    errs_dict = {'mean_ba_factor': VanFunctional.BA_FACTOR_ERR_DICT}

    with open('errs_dict.pkl', 'wb') as f:
        pickle.dump(errs_dict, f)

    # saving loop closure stats
    lc_stats = {'match_count': fnn.LOOP_CLOSURE_MATCH_COUNT,
                'inliers_rate': fnn.LOOP_CLOSURE_INLIERS_RATE}

    with open(f'final_submission_lc_stats.pkl', 'wb') as f:
        pickle.dump(lc_stats, f)

    with open(f'ALIKE_match_count.pkl', 'wb') as f:
        pickle.dump(MATCH_STATS_DICT, f)
