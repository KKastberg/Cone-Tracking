# ColorMasker.py

import numpy as np
import cv2


class ColorMasker:
    """
    This class helps with image color masking and counting colored pixels
    """
    def __init__(self, gaussian_kernel=None):
        self.gaussian_kernel = gaussian_kernel if gaussian_kernel else (5,5)

    # Applies a color filter to a frame and returns the mask
    # Note! Need HSV frame
    def create_color_mask(self, frame, color_range_key):
        # lower_bound = np.array(self.color_range_dict.get(color_range_key)[0])
        lower_bound = np.array(color_range_key[0])
        # upper_bound = np.array(self.color_range_dict.get(color_range_key)[1])
        upper_bound = np.array(color_range_key[1])
        return cv2.inRange(frame, lower_bound, upper_bound)

    # Applies a gaussian filter to the image
    # Note! Need HSV frame
    def apply_gaussian_filter(self, frame):
        return cv2.GaussianBlur(frame, self.gaussian_kernel, 0)

    @staticmethod
    # Applies the a mask to the frame
    def combine_frame_with_mask(frame, mask):
        frame_clone = np.copy(frame)
        return cv2.bitwise_and(frame_clone, frame_clone, mask=mask)

    @staticmethod
    # Combines a list of masks passed in with
    def combine_masks(masks):
        final_mask = masks[0]
        for mask in masks[1:]:
            final_mask = final_mask & mask

        return final_mask

    @staticmethod
    # Invert mask
    def invert_mask(mask):
        mask = mask * -1
        mask = mask + 255
        return mask.astype("uint8")

    def convert_to_HSV(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return frame

    def count_masked_pixels_in_subframe(self, mask, subframe_coords):
        binary_mask = mask / (mask.max() if mask.max() else 1) # Ensure no division with 0
        x = subframe_coords[0]
        y = subframe_coords[1]
        w = subframe_coords[2]
        h = subframe_coords[3]
        subframe = binary_mask[y:y+h, x:x+w]

        return np.sum(subframe)
