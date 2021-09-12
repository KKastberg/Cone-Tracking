#!/usr/bin/env python3
# main.py

"""
@author :       Kevin Kastberg
email:          kevinkas@kth.se
Date:           2021/09/07
Description:    This program
                - Reads a video from 'test_videos' folder and reads every frame
                  till end of video. It stores every frame in variable of same name.
                - A HAAR Cascade classifier then locates cones in the frame
                - The frame is then processed using HSV format and gaussian filtering,
                  which then is used to classify the color of the cone
                - In addition optical flow is applied to help track the cones
                  bewtween image

                Color Convention that we follow:
                ----------------
                    None - UNKNOWN
                    0    - YELLOW
                    1    - BLUE
                    2    - ORANGE

"""

# External imports
import cv2

# Internal imports
from CascadeClassifier.CascadeClassifier import CascadeClassifier
from FlowOptics.FlowOptics import FlowOptics
from ConeManager.ConeManager import ConeManager
from FrameManager.FrameManager import FrameManager
from ConeClassifier.ConeClassifier import ConeClassifier
from ColorMasker.ColorMasker import ColorMasker


# == Global params ==
# -- General params --
STREAM_PATH = '../../data/videos/track.mp4'  # Location of video stream
DRY_RUN = False  # Run without classification
SAVE_STREAM = True  # Save all frames from the stream
STREAM_SAVE_PATH = "./videos/output.mp4"  # Saving location of the frames
STREAM_IN_REDUCED_RESOLUTION = True  # Stream in reduced resolution
REDUCED_RESOLUTION = (630, 346)  # Reduced resolution
DISPLAY_UNKNOWN = True  # Display unclassified cones in stream

# -- Cascade Classifier params --
CASCADE_XML_PATH = "./CascadeClassifier/trained_cascades/3/cascade.xml"  # Path to the classifier XML
CASCADE_PARAMS = dict(  # Cascade Classifier params
    scaleFactor=1.1,
    minNeighbors=30,
    minSize=(10, 10),
    maxSize=(80, 80),
)

# -- Color params --
COLOR_TABLE = [  # Colorname, RGB color, HSV color range
    (
        "yellow",  # Color name
        (0, 255, 255),  # RGB color
        ([15, 80, 80], [50, 255, 255])  # HSV color range
    ),
    (
        "blue",
        (255, 0, 0),
        ([100, 45, 130], [160, 190, 200])
    ),
    (
        "orange",
        (0, 165, 255),
        ([0, 80, 80], [30, 255, 255])
    ),
]
UNKNOWN_COLOR = ("unknown", (0, 0, 255))  # The color specification for a unknown color
COLOR_THREASHOLDS = [0.01, 0.05, 0.05]  # Threashold for a cone to be classified as a specific color
GAUSSIAN_KERNEL = (5, 5)  # The kernel size of the gaussian filter

# Optical Flow params
ENABLE_OPTICAL_FLOW = False  # Run tracking with or without optical flow
FLOW_CONE_LIFETIME = 4  # How many frames a only flow tracked cone should remain
VISUALIZE_OPTICAL_FLOW = False  # Will draw optical flow lines on the stream when active
LK_PARAMS = dict(winSize=(15, 15),  # Parameters for Lucas-Kanade optical flow
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS |
                           cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                 )
FB_PARAMS = dict(flow=None,  # Parameters for Farneback optical flow
                 pyr_scale=0.5,
                 levels=3,
                 winsize=15,
                 iterations=3,
                 poly_n=5,
                 poly_sigma=1.2,
                 flags=0
                 )



# == Main Loop ==
def main():
    # Read video from disk and count frames
    cap = cv2.VideoCapture(STREAM_PATH)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0

    # Init the cone manager which keeps track of all the cones
    cone_mgr = ConeManager(color_table=COLOR_TABLE,
                           unknown_color=UNKNOWN_COLOR,
                           color_threasholds=COLOR_THREASHOLDS)

    # Init the cascade classifier class
    # which classifies cones with HAAR wavelets
    cascade = CascadeClassifier(classifier_xml=CASCADE_XML_PATH,
                                cascade_params=CASCADE_PARAMS)

    # Init the optical flow class which is used to
    # track cone movements between frames
    optical_flow = FlowOptics(lk_params=LK_PARAMS,
                              fb_params=FB_PARAMS)

    # Init the cone classifier which classifies cones based on HSV color
    cone_classifier = ConeClassifier(color_table=COLOR_TABLE,
                                     display_unknown=DISPLAY_UNKNOWN,
                                     color_threasholds=COLOR_THREASHOLDS)

    # Init helper classes
    frame_mgr = FrameManager()
    color_masker = ColorMasker()

    # Init stream writer which saves the stream as a video
    if SAVE_STREAM:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(STREAM_SAVE_PATH, fourcc, 20.0, REDUCED_RESOLUTION)

    # Loop through every frame until the end of the video
    while count < frame_count:
        ret, frame = cap.read()
        if ret == True:
            count = count + 1
        else:
            break

        # === Frame preprocessing ===
        # Resize frame
        if STREAM_IN_REDUCED_RESOLUTION:
            frame = frame_mgr.resize_frame(frame, REDUCED_RESOLUTION)

        gray_frame = frame_mgr.create_gray_frame(frame)
        hsv_frame = frame_mgr.create_hsv_frame(frame)
        g_h_frame = frame_mgr.create_gaussian_frame(hsv_frame,
                                                    kernel=GAUSSIAN_KERNEL)

        # === Locate cones with HAAR cascades classifier ===
        if DRY_RUN:
            potential_cones = []
        else:
            potential_cones = cascade.detect_cones(gray_frame)

        # === Classify Cones with HSV filtering ===
        # Returns a list of cone objects (see "Cone" dataclass for more info)
        cones = cone_classifier.classify_cones(g_h_frame,
                                               potential_cones=potential_cones,
                                               return_masks=False)

        # === Apply optical flow ===
        if ENABLE_OPTICAL_FLOW:
            # Retrive expected cone locations based on LK optical flow algorithm
            last_center_coords = [cone.center_coord for cone in cone_mgr.last_cones]
            flow_center_coords, status = optical_flow.get_lk_flow_coords(gray_frame,
                                                                         last_center_coords)

            # Create cone object from the coords expected by the flow algorithm. Also:
            #  - Flag all cones as flow cones
            #  - If the flow_cone's origin was a HAAR/COLOR predicted cone -> give lifetime <FLOW_CONE_LIFETIME>
            #  - If the flow_cone's origin was a flow_cone -> reduce lifetime with 1
            #  - If the flow_cone's origin's lifetime == 0 -> remove flow_cone
            flow_cones = cone_mgr.create_flow_cones(cone_mgr.last_cones[:],
                                                    flow_center_coords,
                                                    FLOW_CONE_LIFETIME)

            # Combine the optical flow cones with the ones predicted by the
            # HAAR and color filtering. Combine algorithm:
            #  - Keep all HAAR/COLOR cones
            #  - If flow_cone_center_coord in a HAAR/COLOR cone window -> remove the flow_cone
            #  - Else if flow_cone_center_coord not in any HAAR/COLOR cone window -> Keep flow_cone
            cones = cone_mgr.remove_duplicate_cones(cones, flow_cones)

            # Draw out the optical flow stream lines when active
            if VISUALIZE_OPTICAL_FLOW:
                coords = [(int(x*10), int(y*10)) for x in range(int(frame.shape[1]/10)) for y in range(int(frame.shape[0]/10))]
                vectors = optical_flow.get_farneback_vectors(gray_frame, coords)

                for (p1, vector) in zip(coords, vectors):
                    frame = cv2.line(frame, p1, (int(p1[0] + vector[0]), int(p1[1] + vector[1])), (0,255,255), 1)



            # Save current frame to the optical flow class for tracking in next frame
            optical_flow.set_last_frame(gray_frame)

        # === Update the official list of detected cones and add the cones to the frame ===
        frame = cone_mgr.update_cones(frame=frame, cones=cones)

        # === Play streams ===
        streams = [frame]
        for idx, stream in enumerate(streams):
            cv2.imshow(str(idx), stream)
            cv2.waitKey(10)

        # === Save stream ===
        if SAVE_STREAM:
            out.write(frame)


    # === end script ===
    cap.release()
    cv2.destroyAllWindows()
    if SAVE_STREAM:
        out.release()


if __name__ == '__main__':
    main()
