# CascadeClassifier.py

# External imports
import cv2


class CascadeClassifier:
    """
    This class loads an pretrained XML cascade classifier
    and is then used to classify cones from frames.
    """

    def __init__(self, classifier_xml, cascade_params):
        # Load cascade classifier from XML
        self.cascade = cv2.CascadeClassifier(classifier_xml)
        self.cascade_params = cascade_params

    # Detect cones from on image and return their location
    # Note! Only accepts grayscale frames
    # Returns location in the following format:
    # (x, y, w, h)
    def detect_cones(self, gray_frame):
        # Detect cones using the loaded cascade classifier
        cones = self.cascade.detectMultiScale(gray_frame, **self.cascade_params)
        return cones
