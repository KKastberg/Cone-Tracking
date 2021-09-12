# ConeClassifier.py

# Internal imports
from ColorMasker.ColorMasker import ColorMasker
from ConeManager.ConeManager import Cone


class ConeClassifier:
    """
    This class classifies cones and creates new cone objects after classification
    """
    def __init__(self, color_table, display_unknown=False, color_threasholds=None):
        # == Init colors ==
        # Lookup table for all the colors where list index determines the color_idx
        # [(color_name, (rgb), (HSV_color_range), ...]
        # E.g. [("yellow", (0, 255, 255), ((0, 0, 0), (120, 255, 255))), ...]
        self.color_table = color_table

        # Threashold for cones to be classified as a specific color.
        # List of threasholds are organized by color_idx
        self.color_threashold = \
            color_threasholds if color_threasholds else [0.01] * len(color_table)

        # True/False - Display unknown colored cones in the stream
        self.display_unknown = display_unknown


        # == Helper classes ==
        self.color_masker = ColorMasker()


    # Classify cones based on their color
    def classify_cones(self, gaussian_hsv_frame, potential_cones, return_masks=False):
        gh_frame = gaussian_hsv_frame

        # Create color masks for all specified colors
        color_masks = []
        for color in self.color_table:
            hsv_range = color[2]
            mask = self.color_masker.create_color_mask(gh_frame, hsv_range)
            color_masks.append(mask)

        # Calculate the pixel count for every color mask for each cone window
        new_cones = []
        for cone in potential_cones:
            new_cone = Cone(x=cone[0], y=cone[1], width=cone[2], height=cone[3])

            # Calculate the score for each color for one cone
            # The score is calculated by:
            # colored pixels / cone window size
            top_color = None
            top_color_score = 0
            for color_idx, mask in enumerate(color_masks):
                pixs_count = self.color_masker.count_masked_pixels_in_subframe(mask=mask,
                                                                  subframe_coords=cone)

                # If the color score is larger than the threashold and
                # larger than the previous top color score then update the top color
                color_score = pixs_count / new_cone.size
                if color_score > self.color_threashold[color_idx] and \
                    color_score > top_color_score:
                    top_color = color_idx
                    top_color_score = color_score

            # Set the color and the color probability of the cone
            new_cone.color = top_color
            new_cone.color_pb = top_color_score

            if isinstance(top_color, int) or self.display_unknown: new_cones.append(new_cone)

        if return_masks:
            return new_cones, color_masks
        else:
            return new_cones




