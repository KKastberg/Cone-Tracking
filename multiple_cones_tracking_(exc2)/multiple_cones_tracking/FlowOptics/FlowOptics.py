# FlowOptics.py

# External imports
import cv2
import numpy as np

class FlowOptics:
    """
    Traces coordinates with flow optics in between frames
    """

    def __init__(self, first_frame=None, lk_params=None, fb_params=None):

        # If no first frame make sure the functions do
        # not crash without a previous frame
        if first_frame:
            gray = cv2.cvtColor(np.copy(first_frame),
                                cv2.COLOR_BGR2GRAY)
            self.last_frame = np.copy(gray)
            self.no_frame = False
        else:
            self.last_frame = None
            self.no_frame = True
        self.lk_params = lk_params
        self.fb_params = fb_params

    # Store last frame for future tracing
    def set_last_frame(self, gray):
        self.last_frame = gray
        self.no_frame = False

    # Takes in a gray frame and the coordinates to be traced
    # All points in the frame are traced with opencv's implementation of
    # Farneback optical flow and the relevant coords are then returned
    def get_farneback_vectors(self, gray, coords):
        if coords and not self.no_frame:
            flow_vectors = cv2.calcOpticalFlowFarneback(self.last_frame,
                                                        next=gray, **self.fb_params)
            requested_vectors = []
            for coord in coords:
                requested_vectors.append(flow_vectors[coord[1], coord[0]])
            return requested_vectors
        else:
            return []

    # Takes in a gray frame and the coordinates to be traced
    # Utilizing opencv's implementation of Lucas-Kanade Optical flow
    # It returns the new coordinates and status which is a boolean
    # list with equal length as the coordinate list which tells if a
    # coordinate was traced with flow or not.
    def get_lk_flow_coords(self, gray_frame, coords):
        # If no coords -> return empty list
        if coords and not self.no_frame:
            np_coords = np.array(coords).reshape((len(coords), 1, 2)).astype("float32")

            new_coords, status, err = cv2.calcOpticalFlowPyrLK(self.last_frame,
                                                               gray_frame,
                                                               np_coords,
                                                               None,
                                                               **self.lk_params)

            # Return all coords where flow have been found
            return new_coords, status
        else :
            return [], []
