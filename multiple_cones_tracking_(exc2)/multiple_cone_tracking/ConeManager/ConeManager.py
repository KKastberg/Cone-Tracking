# ConeManager.py

# External imports
import numpy as np
from dataclasses import dataclass

# Internal imports
from ColorMasker.ColorMasker import ColorMasker
from FrameManager.FrameManager import FrameManager

@dataclass
class Cone:
    """
    This dataclass defines one cone and its attributes
    """
    x: float                 # X-coord
    y: float                 # Y-coord
    width: int = 40          # Width
    height: int = 40         # Height
    color: int = None        # Color as an int value. None means unidentified
    color_pb: float = 0      # Probabilty of the color - calculated by: colored pixels / cone window size
    lifetime: int = 1
    flow_cone: bool = False

    @property
    # Returns the window size of the cone
    def size(self):
        return self.width * self.height

    @property
    # Returns the cones coordinates as integers
    def int_coord(self):
        return int(self.x), int(self.y)

    @property
    # Return the cone coordinates as floats
    def coord(self):
        return self.x, self.y

    @coord.setter
    # Set the cone coordinates
    def coord(self, coord):
        self.x = float(coord[0])
        self.y = float(coord[1])

    @property
    # Returns the center coordinates of the cone
    def center_coord(self):
        return self.x + self.width / 2,\
               self.y + self.height / 2

    @center_coord.setter
    # Sets the x and y value based on the center coordinate
    def center_coord(self, coord):
        self.x = coord[0] - self.width / 2
        self.y = coord[1] - self.height / 2

    @property
    def cone_window(self):
        return int(self.x), int(self.y), \
               self.width, self.height


class ConeManager:
    """
    Keeps track of all the confirmed detected cones and adds their boundary boxes into the frame
    """
    def __init__(self, color_table, unknown_color, color_threasholds=None, init_cones=None):
        # == Init colors ==
        # Lookup table for all the colors where list index determines the color_idx
        # [(color_name, (rgb), (HSV_color_range), ...]
        # E.g. [("yellow", (0, 255, 255), ((0, 0, 0), (120, 255, 255))), ...]
        self.color_table = color_table

        # Color to set to cones with unknown color
        self.unknown_color = unknown_color

        # == Init cones ==
        self.last_cones = []
        self.current_cones = init_cones if init_cones else []

        # == Helper classes ==
        self.color_masker = ColorMasker()
        self.frame_manager = FrameManager(color_table=color_table,
                                          unknown_color=unknown_color)

    # A function to update all the current cones
    # the last cones will be added to the last_cones variable
    def update_cones(self, frame, cones):
        self.last_cones = self.current_cones.copy()
        self.current_cones = cones
        return self.frame_manager.label_cones(frame=frame,
                                              cones=cones)


    @staticmethod
    # Create cone object from the coords expected by the flow algorithm. Also:
    #  - Flag all cones as flow cones
    #  - If the flow_cone's origin was a HAAR/COLOR predicted cone
    #     -> give lifetime <FLOW_CONE_LIFETIME>
    #  - If the flow_cone's origin was a flow_cone
    #     -> reduce lifetime with 1
    #  - If the flow_cone's origin's lifetime == 0
    #     -> remove flow_cone
    def create_flow_cones(cones, center_coords, lifetime):
        new_cones = []
        for (cone, coord) in zip(cones, center_coords):
            cone.center_coord = coord[0]

            # If cone previously flow cone reduce lifetime by 1
            if cone.flow_cone:
                if cone.lifetime > 0:
                    cone.lifetime -= 1
                    new_cones.append(cone)
            # If cone previously not flow cone, make it a flow cone
            # with full lifetime
            else:
                cone.lifetime = lifetime
                cone.flow_cone = True
                new_cones.append(cone)

        return new_cones


    @staticmethod
    # Combine the optical flow cones with the ones predicted by the
    # HAAR and color filtering. Combine algorithm:
    #  - Keep all HAAR/COLOR cones
    #  - If flow_cone_center_coord in a HAAR/COLOR cone window -> remove the flow_cone
    #  - Else if flow_cone_center_coord not in any HAAR/COLOR cone window -> Keep flow_cone
    def remove_duplicate_cones(cones_a, cones_b):
        combined_cones = cones_a

        # Check if B in all A:s
        def b_in_a_for_all_a(b_center):
            for A in cones_a:
                if A.x < b_center[0] < A.x + A.width and \
                   A.y < b_center[1] < A.y + A.height:
                    return True
            return False

        # For all B check if in A cone window
        #  - If not add B to cones
        for B in cones_b:
            if not b_in_a_for_all_a(B.center_coord):
                combined_cones.append(B)

        return combined_cones
