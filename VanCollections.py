import os
import cv2 as cv
import numpy as np
import pandas as pd
import pickle
import gtsam
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from CNN_detector import CnnDetector

DATA_PATH = '../VAN_ex/dataset/sequences/05/'  # path to frames data


def read_images(idx):
    """
    This function reads image frame.
    :param idx: frame index.
    :return: left image, right image.
    """
    img_name = '{:06d}.png'.format(idx)
    img1 = cv.imread(DATA_PATH + 'image_0/' + img_name, 0)
    img2 = cv.imread(DATA_PATH + 'image_1/' + img_name, 0)

    return img1, img2


def compute_relative_cov_mat(all_marginals: gtsam.Marginals, pose_keys: dict,
                             first_frame: int, cur_frame: int):
    """
    This function computes the relative covariance matrix.
    :param all_marginals: entire marginal matrix.
    :param pose_keys: pose keys dictionary
    :param first_frame: first frame.
    :param cur_frame: current frame.
    :return: relative covariance matrix.
    """
    keys_vec = gtsam.KeyVector()
    keys_vec.append(pose_keys[first_frame])
    keys_vec.append(pose_keys[cur_frame])

    cur_marginal = all_marginals.jointMarginalCovariance(keys_vec).fullMatrix()

    # checking for NAN values
    if np.any(np.isnan(cur_marginal)):
        print(f'*** found NAN in cov bundle ={first_frame}, {cur_frame}')
        print(f'is ALL NAN? {np.all(np.isnan(cur_marginal))}')
        # raise RuntimeError('Nans in covariance')

    cur_marginal = np.linalg.inv(cur_marginal)
    cur_marginal = np.linalg.inv(cur_marginal[6:, 6:])

    return cur_marginal


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


class Frame:
    """
    A class that represents a stereo frame.
    """
    cnn_detector = CnnDetector()

    def __init__(self, frame_idx):
        self.matches_img = None
        self.frame_idx = frame_idx

        left_img, right_img = read_images(frame_idx)

        self.left_frame = left_img
        self.right_frame = right_img

        self.left_values = None
        self.right_values = None

    def get_key_points_and_descriptors(self, detector_object=None):
        """
        Gets key points and descriptors for both frame's images.
        :param detector_object: An open cv detector object to use for key points detection.
        """
        if detector_object is None:
            # computing detection values using cnn for each image in the frame
            if Frame.cnn_detector.read_memory:
                left_pred, right_pred = Frame.cnn_detector.read_frame_detection(self.frame_idx)
            else:
                left_pred = Frame.cnn_detector.extract_keypoints_descriptors(
                    self.left_frame)
                right_pred = Frame.cnn_detector.extract_keypoints_descriptors(
                    self.right_frame)

            self.left_values = left_pred['keypoints'], left_pred['descriptors']
            self.right_values = right_pred['keypoints'], right_pred['descriptors']

            return self.left_values, self.right_values

        kps_left, desc_left = detector_object.detectAndCompute(self.left_frame, None)
        kps_right, desc_right = detector_object.detectAndCompute(self.right_frame, None)

        left_values = (kps_left, desc_left)
        right_values = (kps_right, desc_right)

        return left_values, right_values

    def display_matches(self, matches: list, detector_vals: tuple, show_size, title,
                        save=True, save_name=None):
        """
        Displays matches between the frame's images key points.
        :param matches: list of matches.
        :param detector_vals: tuple of tuples: ((kp1, des1),(kp2, des2)).
        :param show_size: The number of matches to show
        :param title: The matches image window title.
        :param save: whether to save the matches image or not.
        :param save_name: if matches image is being saved, the name of the file to save it in.
        """
        if save and save_name is None:
            save_name = "matches_img" + '_' + "_".join(title.split(" "))

        match_ind_choice = np.array(np.random.choice(np.arange(len(matches)),
                                                     size=show_size), dtype=int)

        to_display = [matches[i] for i in match_ind_choice]  # list of matches to display

        left_vals, right_vals = detector_vals
        left_kps, left_desc = left_vals
        right_kps, right_desc = right_vals

        matches_img = cv.drawMatchesKnn(self.left_frame, left_kps,
                                        self.right_frame, right_kps,
                                        to_display, None, flags=2)

        matches_img = cv.resize(matches_img, (1635, 245))
        self.matches_img = matches_img

        cv.imshow(title, matches_img)
        cv.waitKey(0)

        # some comment

        if save:
            cv.imwrite(f'{save_name}.png', matches_img)

    def get_left_image(self):
        return self.left_frame

    def get_right_image(self):
        return self.right_frame

    def get_images(self):
        return self.left_frame, self.right_frame


class MatchDict:
    """
    This class implements a match dictionary, used for matching keypoints from consecutive
    frames. 4 images overall.
    """
    def __init__(self, match_list):
        self.access_dict = {}
        self.__arrange_dict(match_list)

    def __arrange_dict(self, match_list):
        """
        This function arranges the dictionary to map a keypoint index in the left image
        to its match object.
        :param match_list: list of matches.
        :return: None.
        """
        for match in match_list:
            left_kp_ind = match.queryIdx

            self.access_dict.update({left_kp_ind: match})

    def get_match(self, left_kp_ind):
        """
        This function returns the match containing the given left keypoint.
        :param left_kp_ind: key point index in the left image.
        :return: match object.
        """
        return self.access_dict.get(left_kp_ind)

    def get_right_kp(self, left_kp_ind):
        """
        This function returns the index of the matching right key point of the given left
        key point.
        :param left_kp_ind: index of left key point.
        :return: index of right key point.
        """
        cur_match = self.get_match(left_kp_ind)

        if cur_match is None:
            return None

        return cur_match.trainIdx

    def keys(self):
        """
        This function returns the keys (left key points indices) of the dictionary.
        :return: keys iterator.
        """
        return self.access_dict.keys()


class Track:
    """
    This class implements a single Track object.
    """
    tracks_count = 0  # counter of total tracks created.

    def __init__(self, start_frame, keypoint_ind):
        self.track_id = Track.tracks_count
        self.start_frame = start_frame
        self.keypoint_ind_list = [keypoint_ind]

        Track.tracks_count += 1

    def add_frame(self, new_keypoint_ind):
        """
        This function adds a frame to track.
        :param new_keypoint_ind: index of key points in the new frame.
        :return: None.
        """
        self.keypoint_ind_list.append(new_keypoint_ind)

    def get_id(self):
        """
        This function returns the track's id.
        :return: track's id.
        """
        return self.track_id

    def get_start_frame(self):
        """
        This function returns the track's start frame.
        :return: start frame index.
        """
        return self.start_frame

    def get_len(self):
        """
        :return: This function returns the length of the track.
        """
        return len(self.keypoint_ind_list)

    def last_seen(self):
        """
        This function returns the track's index in frame in which it was last seen.
        :return: index.
        """
        return self.keypoint_ind_list[-1], \
            self.start_frame + len(self.keypoint_ind_list) - 1

    def get_frames_ids(self):
        """
        This function returns the indices of all the frames in which the track appears.
        :return: frames index list.
        """
        frames = []
        cur_frame = self.start_frame

        for offset in range(len(self.keypoint_ind_list)):
            frames.append(cur_frame + offset)

        return frames

    @staticmethod
    def update_tracks_count(new_num: int):
        """
        This function updates the tracks count. used when loading a pre-computed tracking
        data.
        :param new_num: new tracks count.
        :return: None.
        """
        Track.tracks_count = max(Track.tracks_count, new_num)


class TracksData:
    """
    This class implements the tracks data structure.
    """
    def __init__(self, save_supporters_data=False):
        self.tracks = {}
        self.last_seen = {}
        self.cords_data = None
        self.save_supporters_data = save_supporters_data
        self.supporters_data = {}
        self.left_cam_locations = {}
        self.opt_cam_locations = {}
        self.bundle_windows = []
        self.frame_marginals = {}

    @staticmethod
    def create_cords_dict():
        return {'TrackId': [], 'FrameId': [], 'left0_x': [], 'left0_y': [],
                'right0_x': []}

    @ staticmethod
    def update_cords_dict(cords_dict: dict, cur_row, cur_id, frame_id,
                          left0_kps: np.ndarray, right0_kps: np.ndarray):
        cords_dict.get('TrackId').append(cur_id)
        cords_dict.get('FrameId').append(frame_id)
        cords_dict.get('left0_x').append(left0_kps[cur_row, 0])
        cords_dict.get('left0_y').append(left0_kps[cur_row, 1])
        cords_dict.get('right0_x').append(right0_kps[cur_row, 0])

    def __initialize_data_frame(self, multi_match_ind: np.ndarray, left0_kps: np.ndarray,
                                right0_kps: np.ndarray, left1_kps: np.ndarray,
                                right1_kps: np.ndarray):
        # saving cords data in dict for DataFrame
        cords_dict = TracksData.create_cords_dict()

        # creating track objects
        for cur_row in range(multi_match_ind.shape[0]):
            # updating the track in the tracks dict
            cur_track = Track(0, multi_match_ind[cur_row, 0])
            cur_track.add_frame(multi_match_ind[cur_row, 2])
            cur_id = cur_track.get_id()
            self.tracks.update({cur_id: cur_track})

            # updating cords data for frame 0
            TracksData.update_cords_dict(cords_dict, cur_row, cur_id, 0, left0_kps,
                                         right0_kps)

            # updating cords data for frame 1
            TracksData.update_cords_dict(cords_dict, cur_row, cur_id, 1, left1_kps,
                                         right1_kps)

            # updating the last seen dictionary
            self.last_seen.update({multi_match_ind[cur_row, 2]: cur_id})

        # saving cords data as pd data frame
        cords_data = pd.DataFrame.from_dict(cords_dict)
        cords_data['cordInd'] = cords_data['TrackId'].astype(str) + '#' + \
                                cords_data['FrameId'].astype(str)
        cords_data.set_index('cordInd', inplace=True)

        self.cords_data = cords_data

    def __update_existing_track(self, track_id, last_kp_ind):
        track_obj = self.tracks.get(track_id)
        track_obj.add_frame(last_kp_ind)

    def insert_tracks(self, first_frame_id, multi_match_ind: np.ndarray,
                      left0_kps, right0_kps, left1_kps, right1_kps):
        """
        This function inserts tracks to data frame.
        :param first_frame_id: id of the first frame (the first out of 2 consecutive).
        :param multi_match_ind: multi match indices.
        :param left0_kps: first left key points.
        :param right0_kps: first right key points.
        :param left1_kps: second left key points.
        :param right1_kps: second right key points.
        :return: None.
        """
        # formatting the keypoints to cords
        if isinstance(left0_kps, np.ndarray):  # if already np array - just slice
            left0_kps_cords = left0_kps[multi_match_ind[:, 0], :]
            right0_kps_cords = right0_kps[multi_match_ind[:, 1], :]
            left1_kps_cords = left1_kps[multi_match_ind[:, 2], :]
            right1_kps_cords = right1_kps[multi_match_ind[:, 3], :]
        else:
            left0_kps_cords = cv.KeyPoint_convert(left0_kps, multi_match_ind[:, 0])
            right0_kps_cords = cv.KeyPoint_convert(right0_kps, multi_match_ind[:, 1])
            left1_kps_cords = cv.KeyPoint_convert(left1_kps, multi_match_ind[:, 2])
            right1_kps_cords = cv.KeyPoint_convert(right1_kps, multi_match_ind[:, 3])

        # initializing in case of frame 0
        if first_frame_id == 0:
            self.__initialize_data_frame(multi_match_ind, left0_kps_cords,
                                         right0_kps_cords, left1_kps_cords,
                                         right1_kps_cords)

            return

        cords_dict = TracksData.create_cords_dict()
        cur_last_seen = {}

        # creating track objects
        for cur_row in range(multi_match_ind.shape[0]):
            # checking if this kp was already seen
            cur_id = self.last_seen.get(multi_match_ind[cur_row, 0])
            existing_track = False

            if cur_id is None:
                cur_track = Track(first_frame_id, multi_match_ind[cur_row, 0])
                cur_id = cur_track.get_id()
                self.tracks.update({cur_id: cur_track})
            else:
                cur_track = self.tracks.get(cur_id)
                existing_track = True

            cur_track.add_frame(multi_match_ind[cur_row, 2])

            # updating cords data for frame 0
            if not existing_track:
                TracksData.update_cords_dict(cords_dict, cur_row, cur_id, first_frame_id,
                                             left0_kps_cords, right0_kps_cords)

            # updating cords data for frame 1
            TracksData.update_cords_dict(cords_dict, cur_row, cur_id, first_frame_id + 1,
                                         left1_kps_cords, right1_kps_cords)

            # updating the last seen dictionary
            cur_last_seen.update({multi_match_ind[cur_row, 2]: cur_id})

        # saving the current cords data as pandas df
        cur_cords_data = pd.DataFrame.from_dict(cords_dict)
        cur_cords_data['cordInd'] = cur_cords_data['TrackId'].astype(str) + '#' + \
                                    cur_cords_data['FrameId'].astype(str)
        cur_cords_data.set_index('cordInd', inplace=True)
        # self.cords_data = self.cords_data.append(cur_cords_data)
        self.cords_data = pd.concat([self.cords_data, cur_cords_data])

        # explicitly deleting last seen data and replacing it with new data
        del self.last_seen
        self.last_seen = cur_last_seen

    def get_tracks_by_frame(self, frame_id: int):
        """
        This function returns the tracks of a give frame.
        :param frame_id: frame id.
        :return: list of track indices.
        """
        track_ids_sr = self.cords_data[self.cords_data['FrameId'] == frame_id]['TrackId']
        return list(track_ids_sr.unique())

    def get_frames_by_track(self, track_id: int):
        """
        This function returns the frames in which a track appears.
        :param track_id: track id.
        :return: list of frames indices.
        """
        cur_track = self.tracks.get(track_id)
        return cur_track.get_frames_ids()

    def get_track_frame_cords(self, frame_id: int, track_id: int):
        """
        This function returns the coordinates of a track in a given frame.
        :param frame_id: frame id.
        :param track_id: track id.
        :return: image coordinates.
        """
        search_key = f'{track_id}#{frame_id}'
        cur_row = self.cords_data.loc[search_key, :]
        left_x = cur_row.loc['left0_x']
        right_x = cur_row.loc['right0_x']
        y = cur_row.loc['left0_y']

        return np.round(left_x), np.round(right_x), np.round(y)

    def to_pickle(self, file_name: str):
        """
        This function writes the data structure to a pickle file.
        :param file_name: file name.
        :return: None.
        """
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle(file_name: str) -> 'TracksData':
        """
        This function reads data structure from pickle file.
        :param file_name: file name.
        :return: read data.
        """
        # reading the dataframe object
        with open(file_name, 'rb') as f:
            data_obj = pickle.load(f)

        # updating the tracks count according to the maximum
        max_id = int(data_obj.cords_data['TrackId'].max())
        Track.update_tracks_count(max_id)

        return data_obj

    def get_track_object(self, track_id: int):
        """
        This function returns a track object by track id.
        :param track_id: track id.
        :return: track object.
        """
        return self.tracks.get(track_id)

    def get_num_frames(self):
        """
        This function returns the total number of frames in the data.
        :return: frame count.
        """
        return len(self.cords_data['FrameId'].unique())

    def update_supporters_data(self, frame_id, num_supporters, total_samples):
        """
        This function updates supporters data for a frame.
        :param frame_id: frame id.
        :param num_supporters: number of supporters.
        :param total_samples: total number of points.
        :return: None.
        """
        if not self.save_supporters_data:
            return

        cur_supporters_ratio = num_supporters / total_samples
        self.supporters_data.update({frame_id: cur_supporters_ratio})

    def get_supporters_data(self):
        """
        This function returns a dictionary containing supporters data.
        :return: supporters dictionary.
        """
        return self.supporters_data

    def get_track_of_len(self, track_len=10):
        """
        This function returns an id of a track of a given length.
        :param tracks_data: tracks data structure.
        :param track_len: desired length.
        :return: track index.
        """
        tracks_df = self.cords_data
        tracks_df = tracks_df[['TrackId', 'FrameId']].groupby('TrackId').count()
        tracks_df = tracks_df[tracks_df['FrameId'] >= track_len]

        rand_ind = np.random.choice(np.arange(len(tracks_df.index)))

        return tracks_df.index[rand_ind]

    def update_camera_location(self, frame_id, left_ext: np.ndarray):
        """
        This function stores left cam position for frame.
        :param frame_id: frame id.
        :param left_ext: extrinsic matrix.
        :return: None.
        """
        self.left_cam_locations.update({frame_id: left_ext})

    def get_camera_location(self, frame_id):
        """
        This function returns a cameras extrinsic matrix for frame.
        :param frame_id: frame id.
        :return: ext mat as numpy array.
        """
        return self.left_cam_locations.get(frame_id)

    def get_overlapping_range_tracks(self, first_frame: int, last_frame: int):
        """
        This function returns the ids of tracks that appear in the window between frames.
        :param first_frame: first frame id.
        :param last_frame: last frame id.
        :return: set of tracks ids.
        """
        overlapping_tracks = None
        second_tracks = set(self.get_tracks_by_frame(last_frame))

        for first_id in range(first_frame, first_frame + 1):
            first_tracks = set(self.get_tracks_by_frame(first_id))
            cur_intersection = first_tracks.intersection(second_tracks)

            if overlapping_tracks is None:
                overlapping_tracks = cur_intersection
            else:
                overlapping_tracks.update(cur_intersection)

        return overlapping_tracks

    def compute_tracking_ratio(self, first_frame: int, last_frame: int):
        """
        This function computes tracking ratio between 2 frames.
        :param first_frame: first frame.
        :param last_frame: last frame.
        :return: tracking ratio.
        """
        first_track_count = len(self.get_tracks_by_frame(first_frame))
        overlapping_count = len(self.get_overlapping_range_tracks(first_frame, last_frame))

        return overlapping_count / first_track_count

    def update_bundle_window(self, start_frame: int, end_frame: int,
                             bundle_res: gtsam.Values,
                             bundle_graph: gtsam.NonlinearFactorGraph,
                             pose_keys: dict, ext_mat_list: list):
        cur_bundle = BundleWindow(start_frame, end_frame, bundle_res, bundle_graph,
                                  pose_keys)

        # check double
        if len(self.bundle_windows) > 0:
            if start_frame == self.bundle_windows[-1].get_start_id():
                self.bundle_windows.pop()

        self.bundle_windows.append(cur_bundle)

        # updating optimized bundle positions
        cur_frame = start_frame

        for cur_ext_id in range(len(ext_mat_list)):
            self.opt_cam_locations.update({cur_frame: ext_mat_list[cur_ext_id]})
            cur_frame += 1

        print(f'num bundles = {len(self.bundle_windows)}')

    def update_loop_closure_pose(self, optimization_res: gtsam.Values):
        """
        This function updates the optimized position after LC optimization.
        :param optimization_res:
        :return: None.
        """
        for frame_ind in range(0, self.get_num_frames() + 1):
            cur_symbol = gtsam.symbol('c', frame_ind)

            try:
                cur_pos = optimization_res.atPose3(cur_symbol).matrix()[: -1, :]
            except RuntimeError:
                continue

            # print(f'**** found optimized lc in frame {frame_ind}')
            cur_pos = invert_extrinsic(cur_pos)

            self.opt_cam_locations.update({frame_ind: cur_pos})

    def get_num_bundle_windows(self):
        return len(self.bundle_windows)

    def get_bundle_relative(self, bundle_id: int):
        return self.bundle_windows[bundle_id].get_relative_poses()

    def get_bundle_bounds(self, bundle_id: int):
        cur_bundle = self.bundle_windows[bundle_id]

        return cur_bundle.get_start_id(), cur_bundle.get_last_id()

    def get_window_optimized_pose(self, frame_id):
        return self.opt_cam_locations.get(frame_id)

    def get_bundle_cov(self, bundle_id: int):
        return self.bundle_windows[bundle_id].get_relative_cov()

    def get_key_frames_keys(self):
        keys = {}

        for bundle_id in range(len(self.bundle_windows)):
            cur_bundle = self.bundle_windows[bundle_id]
            cur_first = cur_bundle.get_start_id()
            cur_last = cur_bundle.get_last_id()

            first_symbol = gtsam.symbol('c', cur_first)
            keys.update({cur_first: first_symbol})

        return keys

    def get_key_frames_ids(self):
        kf_ids = []

        for bundle_id in range(len(self.bundle_windows)):
            cur_bundle = self.bundle_windows[bundle_id]
            cur_first = cur_bundle.get_start_id()
            cur_last = cur_bundle.get_last_id()

            kf_ids.append(cur_first)

        kf_ids.append(cur_last)

        return kf_ids

    def update_pose_marginals_cov(self, pose_graph: gtsam.NonlinearFactorGraph,
                                  res_values: gtsam.Values):
        """
        This function updates the marginal covariance for a single position.
        :param pose_graph: pose graph.
        :param res_values: result values.
        :return: None.
        """
        all_marginals = gtsam.Marginals(pose_graph, res_values)

        for frame_ind in range(self.get_num_frames()):
            cur_symbol = gtsam.symbol('c', frame_ind)

            try:
                cur_marginal_cov = all_marginals.marginalCovariance(cur_symbol)
            except IndexError:
                continue

            self.frame_marginals.update({frame_ind: cur_marginal_cov})


class BundleWindow:
    """
    This class implements a single bundle adjustment window.
    """
    def __init__(self, start_frame, end_frame, bundle_res: gtsam.Values,
                 bundle_graph: gtsam.NonlinearFactorGraph, pose_keys: dict):
        """
        constructor. used to compute relative postion and covariance.
        :param start_frame: start frame.
        :param end_frame: enf frame.
        :param bundle_res: bundle results.
        :param bundle_graph: bundle graph.
        :param pose_keys: pose keys.
        """
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.first_relative_pose = bundle_res.atPose3(pose_keys[start_frame])
        self.last_relative_pose = bundle_res.atPose3(pose_keys[end_frame])

        # saving relative covariances
        bundle_marginals = gtsam.Marginals(bundle_graph, bundle_res)
        self.relative_cov_mats = {}

        self.relative_cov = compute_relative_cov_mat(bundle_marginals, pose_keys,
                                                     start_frame, end_frame)

    def get_relative_poses(self):
        return self.first_relative_pose, self.last_relative_pose

    def get_start_id(self):
        return self.start_frame

    def get_last_id(self):
        return self.end_frame

    def get_relative_cov(self):
        return self.relative_cov


class CovGraph:
    """
    This class implements the covariance graph that essentially wraps the scipy
    implementation for csgraph and provides a simple API for the purposes of this
    assignment.
    """
    node_count = 0

    def __init__(self):
        self.symbol_converter = {}
        self.id_converter = {}
        self.edge_cov = {}
        self.edge_pose = {}
        self.node_pose = {}
        self.graph_mat = None
        self.paths = None
        self.distances = None
        self.graph = None

    def add_edge(self, new_symbol: str, old_symbol: str, relative_pose: np.ndarray,
                 relative_cov: np.ndarray):
        """
        This function adds an edge between 2 positions in the graph.
        :param new_symbol: symbol of new keyframe. for example 'c280'.
        :param old_symbol: symbol of old keyframe.
        :param relative_pose: relative pose between the keyframes.
        :param relative_cov: relative covariance between the keyframes.
        :return: None.
        """
        new_id = self.id_converter.get(new_symbol)
        old_id = self.id_converter.get(old_symbol)

        cur_weight = np.linalg.norm(relative_cov)
        # cur_weight = np.linalg.det(relative_cov)
        self.graph_mat[new_id, old_id] = cur_weight
        self.graph_mat[old_id, new_id] = cur_weight

        edge_key = f'{new_symbol}#{old_symbol}'
        self.edge_cov.update({edge_key: relative_cov})
        self.edge_pose.update({edge_key: relative_pose})

    def add_node(self, new_symbol: str, old_symbol: str, global_pose: gtsam.Pose3,
                 relative_pose: np.ndarray, relative_cov: np.ndarray):
        """
        This function adds a node (position) to the graph.
        :param new_symbol: symbol of new keyframe. for example 'c280'.
        :param old_symbol: symbol of the predecessor key frame.
        :param global_pose: global pose of the new position.
        :param relative_pose: relative position between new and predecessor.
        :param relative_cov: relative covariance between new and predecessor.
        :return: None.
        """
        if self.id_converter.get(new_symbol) is not None:
            return

        new_id = CovGraph.node_count
        CovGraph.node_count += 1

        self.symbol_converter.update({new_id: new_symbol})
        self.id_converter.update({new_symbol: new_id})
        self.node_pose.update({new_symbol: global_pose})

        if self.graph_mat is None:
            self.graph_mat = np.zeros((1, 1))
        else:
            bottom_row = np.zeros((1, self.graph_mat.shape[1]))
            right_col = np.zeros((self.graph_mat.shape[0] + 1, 1))

            # stacking new columns to the graph matrix
            self.graph_mat = np.vstack([self.graph_mat, bottom_row])
            self.graph_mat = np.hstack([self.graph_mat, right_col])

            self.add_edge(new_symbol, old_symbol, relative_pose, relative_cov)

    def compute_paths(self):
        """
        This function computes the shortest paths in the graph based on the current state.
        :return: None.
        """
        cur_graph = csr_matrix(self.graph_mat)
        cur_dist, cur_predecessors = shortest_path(csgraph=cur_graph, directed=False,
                                                   return_predecessors=True)

        self.distances = cur_dist
        self.paths = cur_predecessors
        self.graph = cur_graph

    def __short_path_helper(self, dst_id: int, src_id: int, path: list):
        """
        This function implements a single iteration in the recursive process for finding
        path.
        :param dst_id: destination frame id.
        :param src_id: source frame id.
        :param path: current nodes in the path.
        :return: None.
        """
        if dst_id == src_id:
            path.append(src_id)
            return

        path.append(dst_id)

        new_dst = self.paths[src_id, dst_id]

        self.__short_path_helper(new_dst, src_id, path)

    def get_shortest_path(self, src_symbol: str, dst_symbol: str):
        """
        This function finds the shortest path in the graph between two keyframes.
        :param src_symbol: source pose symbol, for example: 'c0'.
        :param dst_symbol: destination pose symbol.
        :return: path as list of symbols.
        """
        src_id = self.id_converter.get(src_symbol)
        dst_id = self.id_converter.get(dst_symbol)
        path_lst = []

        self.__short_path_helper(dst_id, src_id, path_lst)

        path_lst = path_lst[::-1]
        symbol_path_lst = []

        for i in range(len(path_lst)):
            cur_symbol = self.symbol_converter.get(path_lst[i])
            symbol_path_lst.append(cur_symbol)

        return symbol_path_lst

    def get_path_cov_matrix(self, src_symbol: str, dst_symbol: str):
        """
        This function returns the covariance estimate for the path between 2 key frames.
        :param src_symbol: source pose symbol, for example: 'c0'.
        :param dst_symbol: destination pose symbol.
        :return: covariance matrix.
        """
        cur_path = self.get_shortest_path(src_symbol, dst_symbol)
        cov_res = None

        for i in range(1, len(cur_path)):
            cur_src = cur_path[i - 1]
            cur_dst = cur_path[i]

            cur_cov = self.edge_cov.get(f'{cur_src}#{cur_dst}')

            if cur_cov is None:
                cur_cov = self.edge_cov.get(f'{cur_dst}#{cur_src}')

                if cur_cov is None:
                    raise KeyError(f'Non existing edge in graph: {cur_dst}#{cur_src}')

            if cov_res is None:
                cov_res = cur_cov
            else:
                cov_res = cov_res + cur_cov

        return cov_res

    # TODO: modify threshold
    def loop_closure_candidates(self, new_symbol: str, dist_threshold=50):
        """
        This function looks for possible loop closure candidates in the graph (cheap stage).
        :param new_symbol: new symbol to search for.
        :param dist_threshold: max mahalanobis distance threshold.
        :return: candidates list, respective distance.
        """
        self.compute_paths()
        new_pose = self.node_pose.get(new_symbol)
        candidates_lst = []
        distances = []

        for cur_symbol in self.id_converter.keys():
            if cur_symbol == new_symbol:
                continue

            cur_relative_cov = self.get_path_cov_matrix(new_symbol, cur_symbol)
            # print(f'relative = {np.linalg.norm(cur_relative_cov)}')
            cur_relative_cov = np.linalg.inv(cur_relative_cov)

            cur_pose = self.node_pose.get(cur_symbol)
            cur_relative_pose = new_pose.between(cur_pose)
            cur_relative_rotation = cur_relative_pose.rotation().ypr().reshape((3, 1))
            cur_relative_translation = cur_relative_pose.translation().reshape((3, 1))

            cur_relative_pose = np.vstack([cur_relative_rotation, cur_relative_translation])

            cur_dist = cur_relative_pose.T @ cur_relative_cov @ cur_relative_pose
            cur_dist = np.sqrt(cur_dist)

            distances.append(cur_dist)

            # print(f'cheap dist = {cur_dist}')

            if cur_dist <= dist_threshold:
                candidates_lst.append(cur_symbol)

        return candidates_lst, np.array(distances).flatten()

    def update_optimized_poses(self, optimization_res: gtsam.Values):
        """
        This function updates the positions for each node in the graph based on the result
        of pose graph optimization.
        :param optimization_res: optimized results vector.
        :return: None.
        """
        temp_dict = {}

        for symbol_str in self.node_pose.keys():
            cur_symbol = gtsam.symbol('c', int(symbol_str[1:]))
            cur_pose = optimization_res.atPose3(cur_symbol)
            temp_dict.update({symbol_str: cur_pose})

        self.node_pose.update(temp_dict)




