import numpy as np

from collections.VanCollections import TracksData, Frame
from matplotlib import pyplot as plt
from functional import VanFunctional as F

LOAD_PRE_COMPUTED_TRACKING = True


def process_tracks_stats(tracks_data: TracksData):
    """
    This function computes tracks statistics.
    :param tracks_data: tracks data structure.
    :return: None.
    """
    tracks_df = tracks_data.cords_data

    # Total number of tracks
    track_count = len(tracks_df['TrackId'].unique())
    print(f'Total number of tracks : {track_count}')

    # Total number of frames
    frame_count = len(tracks_df['FrameId'].unique())
    print(f'Total number of frames: {frame_count}')

    # track length statistics
    tracks_len_df = tracks_df[['TrackId', 'FrameId']].groupby('TrackId').count()
    tracks_len_df.sort_values(by='FrameId', inplace=True, ascending=False)
    print(f'Track minimum length: {tracks_len_df.min().item()}')
    print(f'Track maximum length: {tracks_len_df.max().item()}')
    print(f'Track mean length: {tracks_len_df.mean().item()}')

    # average number of tracks per image
    tracks_count_df = tracks_df[['TrackId', 'FrameId']].groupby('FrameId').count()
    print(f'Mean number of frame links: {tracks_count_df.mean().item()}')


def cut_image_region(left_x, y, region_size=100, img_width=1226, img_height=370):
    """
    This function cuts a region around a track.
    :param left_x: start x.
    :param y: start y.
    :param region_size: region size.
    :param img_width: original image width
    :param img_height: original image height.
    :return: region boundaries.
    """
    offset = np.floor(region_size / 2)
    scatter_x_offset = 0
    scatter_y_offset = 0

    # vertical bounds
    upper_bound = max(y - offset, 0)
    lower_bound = min(y + (region_size - offset), img_height)
    padding_size = region_size - np.abs(lower_bound - upper_bound)

    if upper_bound == 0:
        lower_bound += padding_size
        scatter_y_offset -= padding_size
    elif lower_bound == img_height:
        upper_bound -= padding_size
        scatter_y_offset += padding_size

    # horizontal bounds - left image
    left_bound = max(left_x - offset, 0)
    right_bound = min(left_x + (region_size - offset), img_width)
    padding_size = region_size - np.abs(right_bound - left_bound)

    if left_bound == 0:
        right_bound += padding_size
        scatter_x_offset -= padding_size
    elif right_bound == img_width:
        left_bound -= padding_size
        scatter_x_offset += padding_size

    return int(np.round(left_bound)), int(np.round(right_bound)), \
        int(np.round(upper_bound)), int(np.round(lower_bound)), \
        int(np.round(scatter_x_offset)), int(np.round(scatter_y_offset))


def visualize_track(tracks_data: TracksData, track_id: int, max_seq_len=10):
    """
    This function visualizes a track.
    :param tracks_data: tracks data structure.
    :param track_id: track id.
    :param max_seq_len: maximum frames to visualize.
    :return: None.
    """
    track_obj = tracks_data.get_track_object(track_id)
    start_frame = track_obj.get_start_frame()
    track_len = min(track_obj.get_len(), max_seq_len)

    for frame_id in range(start_frame, start_frame + track_len):
        cur_frame = Frame(frame_id)
        cur_track_cords = tracks_data.get_track_frame_cords(frame_id, track_id)

        # cutting region from left image
        x_min, x_max, y_min, y_max, left_x_offset, left_y_offset = cut_image_region(
            cur_track_cords[0], cur_track_cords[2])

        left_region = cur_frame.get_left_image()[y_min: y_max, x_min: x_max]

        # cutting region from right image
        x_min, x_max, y_min, y_max, right_x_offset, right_y_offset = cut_image_region(
            cur_track_cords[1], cur_track_cords[2])

        right_region = cur_frame.get_right_image()[y_min: y_max, x_min: x_max]

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(left_region, cmap='gray')
        ax[1].imshow(right_region, cmap='gray')

        ax[0].scatter(50 + left_x_offset, 50 + left_y_offset)
        ax[1].scatter(50 + right_x_offset, 50 + right_y_offset)

        ax[0].set_title('left region')
        ax[1].set_title('right region')
        fig.suptitle(f'Frame #{frame_id}')

        plt.savefig(f'{track_id}_f{frame_id}.png')
        plt.show()


def get_connectivity_data(tracks_data: TracksData):
    """
    This function computes the connectivity data.
    :param tracks_data: tracks data structure.
    :return: connectivity data.
    """
    out_going_list = []
    num_frames = tracks_data.get_num_frames()

    cur_frame_tracks = set(tracks_data.get_tracks_by_frame(0))
    next_frame_tracks = set(tracks_data.get_tracks_by_frame(1))

    for next_frame_id in range(1, num_frames - 2):
        cur_out_going = cur_frame_tracks.intersection(next_frame_tracks)
        out_going_list.append(len(cur_out_going))

        cur_frame_tracks = next_frame_tracks
        next_frame_tracks = set(tracks_data.get_tracks_by_frame(next_frame_id + 1))

    return out_going_list


def plot_connectivity(connectivity_data: list):
    """
    This function plots connectivity data.
    :param connectivity_data: connectivity data as list.
    :return: None.
    """
    x_data = np.arange(len(connectivity_data))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x_data, connectivity_data)

    plt.xlabel('frame')
    plt.ylabel('outgoing tracks')
    plt.title('Connectivity')

    plt.show()


def plot_track_length_histogram(tracks_data: TracksData):
    """
    This function plots the track length histogram.
    :param tracks_data: tracks data.
    :return: None.
    """
    tracks_df = tracks_data.cords_data[['TrackId', 'FrameId']]
    tracks_df = tracks_df.groupby('TrackId').count()
    tracks_df.reset_index(None, drop=False, inplace=True)
    tracks_df = tracks_df.groupby('FrameId').count()

    hist_x, hist_y = [0], [0]
    hist_x.extend(list(tracks_df.index))
    hist_y.extend(list(tracks_df['TrackId']))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(hist_x, hist_y)

    plt.xlabel('Track length')
    plt.ylabel('Track #')
    plt.title('Track length histogram')

    plt.show()


def track_re_projection(tracks_data: TracksData, track_id=-1, positions_lst=None):
    """
    This function computes the re-projection error for a track.
    :param tracks_data: tracks data.
    :return: errors list, start frame.
    """
    if track_id < 0:
        track_id = tracks_data.get_track_of_len(10)

    track_obj = tracks_data.get_track_object(track_id)
    last_frame = track_obj.get_start_frame() + track_obj.get_len() - 1

    # getting the track's coordinates
    last_track_cords = tracks_data.get_track_frame_cords(last_frame, track_id)

    # loading the ground truth data
    if positions_lst is None:
        gt_list = F.get_camera_ground_truth()
    else:
        gt_list = positions_lst

    k, m1, m2 = F.get_calib()

    # computing the last camera matrices
    last_left_ext = gt_list[last_frame]
    last_right_rotation = last_left_ext[:, :-1]
    last_right_translation = (last_left_ext[:, -1] + m2[:, -1]).reshape(3, 1)
    last_right_ext = np.hstack([last_right_rotation, last_right_translation])

    # computing triangulation for the last frame
    track_3d_cords = F.compute_triangulation(k, last_left_ext, last_right_ext,
                                             (last_track_cords[0], last_track_cords[2]),
                                             (last_track_cords[1], last_track_cords[2]))
    track_3d_cords = track_3d_cords.reshape(4, 1)
    track_3d_cords = track_3d_cords / track_3d_cords[-1, :]
    dist_list = []

    for frame_id in range(track_obj.get_start_frame(), track_obj.get_start_frame() + track_obj.get_len()):
        cur_track_cords = tracks_data.get_track_frame_cords(frame_id, track_id)
        cur_left_img_cords = [cur_track_cords[0], cur_track_cords[2]]
        cur_right_img_cords = [cur_track_cords[1], cur_track_cords[2]]

        cur_left_img_cords = np.array(cur_left_img_cords).reshape(2, 1)
        cur_right_img_cords = np.array(cur_right_img_cords).reshape(2, 1)

        cur_left_ext = gt_list[frame_id]
        cur_right_r = cur_left_ext[:, :-1]
        cur_right_t = (cur_left_ext[:, -1] + m2[:, -1]).reshape(3, 1)
        cur_right_ext = np.hstack([cur_right_r, cur_right_t])

        # projecting the points to pixels
        cur_left_proj = k @ cur_left_ext @ track_3d_cords
        cur_left_proj = cur_left_proj[:-1, :] / cur_left_proj[-1, :]

        cur_right_proj = k @ cur_right_ext @ track_3d_cords
        cur_right_proj = cur_right_proj[:-1, :] / cur_right_proj[-1, :]

        # computing the average error for current frame
        cur_err = np.linalg.norm(cur_left_img_cords - cur_left_proj) + \
                  np.linalg.norm(cur_right_img_cords - cur_right_proj)
        dist_list.append(cur_err / 2)

    return dist_list, track_obj.get_start_frame()


def plot_re_projection_error(dist_list: list, start_frame: int):
    """
    This function plots the re-projection error.
    :param dist_list: errors list.
    :param start_frame: start frame.
    :return: None.
    """
    x_vals = np.arange(start_frame, start_frame + len(dist_list), 1).astype(int)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x_vals, dist_list)

    plt.xlabel('frame')
    plt.ylabel('average re-projection error')
    plt.title('Re-projection error per frame')

    plt.show()


def plot_inliers_data(tracks_data: TracksData):
    """
    This function plots the inliers data.
    :param tracks_data: tracks data.
    :return: None.
    """
    cur_data = tracks_data.get_supporters_data()
    x_data = np.array(list(cur_data.keys()))
    y_data = np.array(list(cur_data.values())) * 100

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x_data, y_data)

    plt.xlabel('frame')
    plt.ylabel('inliers percentage')
    plt.title('Q5 - inliers percentage per frame')

    plt.show()


if __name__ == '__main__':
    if not LOAD_PRE_COMPUTED_TRACKING:
        F.track_frames(2560, 'try1.pkl', True)

    x = TracksData.from_pickle('try1.pkl')

    # ----------------------------------------------------- Question 2
    process_tracks_stats(x)

    # ----------------------------------------------------- Question 3
    cur_id = x.get_track_of_len(10)
    visualize_track(x, cur_id, max_seq_len=10)

    # ----------------------------------------------------- Question 4
    c_data = get_connectivity_data(x)
    plot_connectivity(c_data)

    # ----------------------------------------------------- Question 5
    plot_inliers_data(x)

    # ----------------------------------------------------- Question 6
    plot_track_length_histogram(x)

    # ----------------------------------------------------- Question 7
    re_projection_errs, start = track_re_projection(x)
    plot_re_projection_error(re_projection_errs, start)
