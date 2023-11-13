import numpy as np
from VanCollections import Frame, TracksData, CovGraph
from functional.VanFunctional import get_calib, compute_2kf_bundle, compute_ransac, \
    get_loop_track_vals, compute_frame_cloud, get_global_gtsam_pose, \
    evaluate_frames_distance, compute_bundle_window, invert_extrinsic, \
    compose_t_transform, get_camera_locations_from_values
import gtsam
import pandas as pd
from colorama import Style, Fore

LOOP_CLOSURE_MATCH_COUNT = []
LOOP_CLOSURE_INLIERS_RATE = []
PERFORM_LOOP_CLOSURE = True  # control loop closure for analysis


def filter_consecutive_matches(cons_matches: np.ndarray, f0_keep_matches: np.ndarray,
                               f1_keep_matches: np.ndarray):
    """
    This funtion filters consecutive matches.
    :param cons_matches: consecutive matches.
    :param f0_keep_matches: kept f0 matches.
    :param f1_keep_matches: kept f1 matches.
    :return: filtered consecutive matches.
    """
    # finding which consecutive matches exist in both f0, f1
    ext_f0 = set(
        np.argwhere(np.isin(cons_matches[:, 0], f0_keep_matches[:, 0])).flatten())
    ext_f1 = set(
        np.argwhere(np.isin(cons_matches[:, 1], f1_keep_matches[:, 0])).flatten())

    # intersecting the indices
    total_ext = np.array(list(ext_f0.intersection(ext_f1))).flatten()

    return cons_matches[total_ext, :]


def handle_single_loop_cnn(src_symbol: str, dst_symbol: str):
    """
    This function performs consensus matching (expansive stage) for 2 key frames that are
    candidates for loop closure.
    :param src_symbol: source pose symbol, for example: 'c0'.
    :param dst_symbol: destination symbol.
    :return: supporters indices, key points coordinates in the following order:
             source frame left, source frame right, destination frame left,
             destination frame right.
    """
    src_frame, dst_frame = get_loop_track_vals(src_symbol, dst_symbol)
    k, m1, m2 = get_calib()

    # computing src matches
    left_src_vals, right_src_vals = src_frame.get_key_points_and_descriptors()
    cur_src_cloud, cur_src_keep, cur_src_discard = compute_frame_cloud(left_src_vals,
                                                                       right_src_vals)
    # computing dst matches
    left_dst_vals, right_dst_vals = dst_frame.get_key_points_and_descriptors()
    cur_dst_cloud, cur_dst_keep, cur_dst_discard = compute_frame_cloud(left_dst_vals,
                                                                       right_dst_vals)
    # consecutive matches
    cons_matches = Frame.cnn_detector.match_descriptors(left_src_vals[1], left_dst_vals[1])

    # filter consecutive matches that don't exist in f0 keep
    cons_matches = filter_consecutive_matches(cons_matches, cur_src_keep, cur_dst_keep)

    # computing cross-matches
    multi_match_ind = find_multi_matches(cons_matches, cur_src_keep, cur_dst_keep)
    multi_cloud = slice_cloud_by_multi_matches(cur_src_cloud, multi_match_ind,
                                               cur_src_keep)

    # filtering negative z values
    multi_match_ind, multi_cloud = filter_extreme_3d_points(multi_match_ind, multi_cloud)

    # key points cords
    left_src_kps, right_src_kps, left_dst_kps, right_dst_kps = slice_kps_by_multi_matches(
        left_src_vals[0], right_src_vals[0], left_dst_vals[0], right_dst_vals[0],
        multi_match_ind)

    # performing P3P ransac
    supporters, dst_ext_mat = compute_ransac(
        multi_cloud, multi_match_ind, left_dst_vals[0], k, m1, m2, left_src_kps,
        right_src_kps, left_dst_kps, right_dst_kps)

    return supporters, left_src_kps, right_src_kps, left_dst_kps, right_dst_kps


def handle_loop_closure(src_symbol: str, candidates: list):
    """
    This function looks for the best loop closure candidate (if there is any) given a
    list of possible candidates that already passed the cheap stage. The best candidate
    is the one who satisfies the requirement for at least 40% inliers and has the highest
    inliers ratio of all the candidates.
    :param src_symbol: source pose symbol, for example: 'c0'.
    :param candidates: candidates list.
    :return: single best match as symbol str, supporters indices, key points coordinates
             in the following order: source frame left, source frame right, destination
             frame left, destination frame right.
    """
    max_supp_score = 0
    loop_closure_match = None
    res_supporters, res_left_src, res_right_src, res_left_dst, res_right_dst = \
        None, None, None, None, None

    for candidate in candidates:
        try:
            supporters, left_src_kps, right_src_kps, left_dst_kps, right_dst_kps = \
                handle_single_loop_cnn(src_symbol, candidate)  # TODO: check this
        except Exception:
            continue

        cur_supp_score = supporters.shape[0] / left_src_kps.shape[0]

        print(f'expensive supp score for {candidate} = {cur_supp_score}')
        # TODO: modify this threshold too
        if cur_supp_score > 0.75:
            if cur_supp_score >= max_supp_score:
                max_supp_score = cur_supp_score
                loop_closure_match = candidate

                res_supporters = supporters
                res_left_src = left_src_kps
                res_right_src = right_src_kps
                res_left_dst = left_dst_kps
                res_right_dst = right_dst_kps

    if loop_closure_match is not None:
        print(f'found loop: {src_symbol} # {loop_closure_match}')

    return loop_closure_match, res_supporters, res_left_src, res_right_src, res_left_dst, \
        res_right_dst


def find_multi_matches(cons_matches: np.ndarray, f0_keep_matches: np.ndarray,
                       f1_keep_matches: np.ndarray):
    """
    This function finds matches across two consecutive frames (4 images).
    :param cons_matches: consecutive matches.
    :param f0_keep_matches: kept f0 matches.
    :param f1_keep_matches: kept f1 matches.
    :return: multi matches indices.
    """
    # converting tables to df
    cons_df = pd.DataFrame(data=cons_matches)
    f0_keep_df = pd.DataFrame(data=f0_keep_matches, index=f0_keep_matches[:, 0])
    f1_keep_df = pd.DataFrame(data=f1_keep_matches, index=f1_keep_matches[:, 0])

    # filtering f0 and f1 by consecutive keypoints
    f0_keep_df = f0_keep_df.loc[cons_df.iloc[:, 0], :]
    f1_keep_df = f1_keep_df.loc[cons_df.iloc[:, 1], :]

    # converting back to numpy
    f0_keep_df = f0_keep_df.to_numpy()
    f1_keep_df = f1_keep_df.to_numpy()
    cons_df = cons_df.to_numpy()

    # arranging indices in table
    multi_match_table = np.empty((cons_df.shape[0], 4))
    multi_match_table[:, 0] = cons_df[:, 0]
    multi_match_table[:, 1] = f0_keep_df[:, 1]
    multi_match_table[:, 2] = cons_df[:, 1]
    multi_match_table[:, 3] = f1_keep_df[:, 1]

    return multi_match_table.astype(int)


def slice_cloud_by_multi_matches(cloud_3d_cords: np.ndarray, multi_match_ind: np.ndarray,
                                 f0_keep_matches: np.ndarray):
    """
    This function arranges the 3d cloud cords according to the multi match ind.
    :param cloud_3d_cords: 3d cords.
    :param multi_match_ind: multi match indices.
    :param f0_keep_matches: kept f0 matches.
    :return:
    """
    cloud_df = pd.DataFrame(data=cloud_3d_cords.T, index=f0_keep_matches[:, 0])
    cloud_df = cloud_df.loc[multi_match_ind[:, 0], :]

    return cloud_df.to_numpy().T


def filter_extreme_3d_points(multi_match_ind: np.ndarray, multi_cloud: np.ndarray,
                             max_z_dist=100):
    """
    This function is used to filter extreme 3d values such as negative z or over 100.
    :param multi_match_ind: multi match indices.
    :param multi_cloud: multi cloud indices.
    :param max_z_dist:
    :return: filtered arrays.
    """
    # find indices of negative z values
    non_negative_z_ind = np.argwhere(multi_cloud[2, :] >= 0)
    valid_dist_ind = np.argwhere(multi_cloud[2, :] <= max_z_dist)
    overall_valid_ind = np.intersect1d(non_negative_z_ind, valid_dist_ind).flatten()

    return multi_match_ind[overall_valid_ind, :], multi_cloud[:, overall_valid_ind]


def slice_kps_by_multi_matches(left0_kps: np.ndarray, right0_kps: np.ndarray,
                               left1_kps: np.ndarray, right1_kps: np.ndarray,
                               multi_matches_ind: np.ndarray):
    """
    This function slices kps by multi matches.
    :param left0_kps: left0 key points.
    :param right0_kps: right0 key points.
    :param left1_kps: left1 key points.
    :param right1_kps: right1 key points.
    :param multi_matches_ind: multi match indices.
    :return: sliced arrays.
    """
    return left0_kps[multi_matches_ind[:, 0], :], right0_kps[multi_matches_ind[:, 1], :], \
        left1_kps[multi_matches_ind[:, 2], :], right1_kps[multi_matches_ind[:, 3], :]


def detect_loop_closure(tracks_data: TracksData, bundle_id: int,
                        pose_graph: gtsam.NonlinearFactorGraph, init_vals: gtsam.Values,
                        cov_graph: CovGraph):
    """
    This function detects loop closures upon bundle optimization.
    :param tracks_data: tracks data.
    :param bundle_id: bundle id.
    :param pose_graph: pose graph.
    :param init_vals: initial values.
    :param cov_graph: CovGraph.
    :return: True id detected, False otherwise.
    """
    cur_first, cur_last = tracks_data.get_bundle_bounds(bundle_id)
    lp_c, lp_d = cov_graph.loop_closure_candidates(f'c{cur_last}')
    last_symbol = gtsam.symbol('c', cur_last)

    if len(lp_c) > 0:
        # print(lp_c)
        loop_closure_match, cur_supporters, left_src_cords, right_src_cords, \
            left_dst_cords, right_dst_cords = handle_loop_closure(f'c{cur_last}',
                                                                  lp_c)

        while loop_closure_match is not None:
            try:
                print(f'found loop closure!')
                closure_cov, closure_pose = compute_2kf_bundle(
                    tracks_data, loop_closure_match, f'c{cur_last}', cur_supporters,
                    left_src_cords, right_src_cords, left_dst_cords, right_dst_cords)

                cov_graph.add_edge(f'c{cur_last}', loop_closure_match, closure_pose,
                                   closure_cov)

                closure_symbol = gtsam.symbol('c', int(loop_closure_match[1:]))
                closure_cov = gtsam.noiseModel.Gaussian.Covariance(closure_cov)

                closure_factor = gtsam.BetweenFactorPose3(closure_symbol, last_symbol,
                                                          closure_pose, closure_cov)
                pose_graph.add(closure_factor)

                # optimizing the graph
                optimizer = gtsam.LevenbergMarquardtOptimizer(pose_graph, init_vals)
                optimization_res = optimizer.optimize()
                cov_graph.update_optimized_poses(optimization_res)
                init_vals = optimization_res

                LOOP_CLOSURE_MATCH_COUNT.append(left_src_cords.shape[0])
                LOOP_CLOSURE_INLIERS_RATE.append(cur_supporters.shape[0] /
                                                 left_src_cords.shape[0])

                return True

            except Exception:
                print('hit candidate replace exception')
                lp_c.remove(loop_closure_match)

                loop_closure_match, cur_supporters, left_src_cords, right_src_cords, \
                    left_dst_cords, right_dst_cords = handle_loop_closure(f'c{cur_last}',
                                                                          lp_c)

    return False


def update_pose_graph(tracks_data: TracksData, bundle_id: int,
                      pose_graph: gtsam.NonlinearFactorGraph, init_vals: gtsam.Values,
                      cov_graph: CovGraph):
    """
    This function updates the pose graph with the optimized results.
    :param tracks_data: tracks data.
    :param bundle_id: bundle id.
    :param pose_graph: pose graph.
    :param init_vals: initial values.
    :param cov_graph: CovGraph.
    :return: None.
    """
    cur_first, cur_last = tracks_data.get_bundle_bounds(bundle_id)
    first_relative, last_relative = tracks_data.get_bundle_relative(bundle_id)
    cur_relative_cov = tracks_data.get_bundle_cov(bundle_id)
    cur_relative_cov = gtsam.noiseModel.Gaussian.Covariance(cur_relative_cov)

    first_symbol = gtsam.symbol('c', cur_first)
    last_symbol = gtsam.symbol('c', cur_last)

    cur_relative_pose = first_relative.between(last_relative)

    cur_factor = gtsam.BetweenFactorPose3(first_symbol, last_symbol,
                                          cur_relative_pose, cur_relative_cov)

    pose_graph.add(cur_factor)

    # adding positions to values vector
    first_global_pose = get_global_gtsam_pose(tracks_data, cur_first)
    last_global_pose = get_global_gtsam_pose(tracks_data, cur_last)
    init_vals.insert(last_symbol, last_global_pose)

    # extending the covariance graph
    if cur_first == 0:
        cov_graph.add_node('c0', 'c0', first_global_pose, cur_relative_pose,
                           tracks_data.get_bundle_cov(bundle_id))

    cov_graph.add_node(f'c{cur_last}', f'c{cur_first}', last_global_pose,
                       cur_relative_pose, tracks_data.get_bundle_cov(bundle_id))


def handle_bundle_window(tracks_data: TracksData, first_kf: int, last_kf: int,
                         pose_graph: gtsam.NonlinearFactorGraph, cov_graph: CovGraph,
                         init_vals: gtsam.Values, bundle_dist=3):
    """
    This function handles single bundle window optimization by deciding if it is necessary.
    :param tracks_data: tracks data.
    :param first_kf: first key frame.
    :param last_kf: last key frame.
    :param pose_graph: pose graph.
    :param cov_graph: CovGraph.
    :param init_vals: initial values.
    :param bundle_dist: distance between keyframes in a bundle.
    :return: last kf upon optimization, first kf otherwise.
    """
    if last_kf <= first_kf:
        raise RuntimeError('could not resolve bundle choices')

    cur_dist = evaluate_frames_distance(tracks_data, first_kf, last_kf)

    # measuring distance. optimizing only for dist >= 3 meters
    if cur_dist < bundle_dist and last_kf - first_kf < 20:
        return first_kf

    # performing bundle window optimization
    while True:
        try:
            cur_graph, cur_bundle_res, cur_pose_keys, cur_q_keys = compute_bundle_window(
                first_kf, last_kf, tracks_data)
            break
        except Exception:
            print(Fore.RED + f'sth went wrong with BA {last_kf}, trying to lower last kf')
            print(Style.RESET_ALL)

    first_kf_ext = cur_bundle_res.atPose3(cur_pose_keys[first_kf]).matrix()[:-1, :]
    first_kf_ext = invert_extrinsic(first_kf_ext)
    global_ext_mat = tracks_data.get_window_optimized_pose(first_kf)

    if global_ext_mat is None:
        global_ext_mat = tracks_data.get_camera_location(first_kf)

    global_ext_mat = compose_t_transform(global_ext_mat, first_kf_ext)

    # computing the positions cords
    cur_poses_cords, cam_ext_list = get_camera_locations_from_values(
        cur_bundle_res, cur_pose_keys, global_ext_mat)

    # updating the optimized position in the data frame
    tracks_data.update_bundle_window(first_kf, last_kf, cur_bundle_res, cur_graph,
                                     cur_pose_keys, cam_ext_list)

    print(f'computed bundle! first = {first_kf} ;; last = {last_kf}')

    bundle_id = tracks_data.get_num_bundle_windows() - 1
    update_pose_graph(tracks_data, bundle_id, pose_graph, init_vals, cov_graph)

    if PERFORM_LOOP_CLOSURE:
        loop_detected = detect_loop_closure(tracks_data, bundle_id, pose_graph, init_vals,
                                            cov_graph)
    else:
        loop_detected = False

    if loop_detected:
        tracks_data.update_loop_closure_pose(init_vals)

    return last_kf

