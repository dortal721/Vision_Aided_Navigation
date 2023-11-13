import sys
sys.path.insert(0, 'ALIKE-main')

from alike import ALike, configs
from demo import SimpleTracker
import torch
import numpy as np
import cv2
import pickle
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MEMORIZED_DETECTION_FILE = 'detection_res.pkl'
torch.manual_seed(0)


class CnnDetector:
    """
    This class implements the ALIKE detector by wrapping the original implementation.
    """
    def __init__(self):
        self.model = ALike(**configs['alike-l'], device=device, top_k=-1, scores_th=0.2,
                           n_limit=5000)
        self.tracker = SimpleTracker()
        self.read_memory = not torch.cuda.is_available()
        self.detection_res = None

        if self.read_memory:
            with open(MEMORIZED_DETECTION_FILE, 'rb') as f:
                self.detection_res = pickle.load(f)

    def extract_keypoints_descriptors(self, img):
        """
        This function extracts key points and descriptors.
        :param img: input image.
        :return: prediction.
        """
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred = self.model(x, sub_pixel=True)

        return pred

    def match_descriptors(self, desc1, desc2):
        """
        This function matches descriptors.
        :param desc1:
        :param desc2:
        :return: matches.
        """
        return self.tracker.mnn_mather(desc1, desc2)

    def read_frame_detection(self, frame_idx):
        """
        This function reads pre computed detection for single frame.
        :param frame_idx: frame index.
        :return: left prediction, right prediction.
        """
        if not self.read_memory:
            raise ValueError('Trying to read detection when not defined')

        frame_res_dict = self.detection_res.get(frame_idx)
        pause_time = frame_res_dict.get('time')
        left_pred = frame_res_dict.get('left_pred')
        right_pred = frame_res_dict.get('right_pred')

        # pausing runtime to simulate actual calculation
        time.sleep(min(pause_time, 0.09))

        return left_pred, right_pred
