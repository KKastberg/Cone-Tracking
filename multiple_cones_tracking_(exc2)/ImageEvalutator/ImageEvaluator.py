# env.py

import numpy as np
from PIL import Image
import copy
import cv2


class ImageEvaluator:
    '''
    This class is used to evaluate the performance of the DQN agent.
    '''
    def __init__(self, cone_image, penalty_function=None, pixel_diff_discount=0.1, hsv_channels=None):

        # The image which is supposed to be hidden by the DQN agent in the frame.
        # Note! In RGB format
        self.cone_image = cone_image
        self.cone_image_width = cone_image.shape[0]
        self.cone_image_height = cone_image.shape[1]

        # Create the cone_image in HSV format for placement in the frame since frame is in HSV format
        self.hsv_cone_image = cv2.cvtColor(self.cone_image, cv2.COLOR_BGR2HSV)

        # The alpha mask is used to mask the image since it comes in PNG format
        # originally which cannot be represented in HSV. Therefore it is represented in RGB here.
        # TODO: Convert to binary mask to reduce memory usage
        self.cone_alpha_mask = self.cone_image

        # Define a loss function that will be applied on the pixel value diff
        self.penalty_function = penalty_function if penalty_function else lambda x: x ** 2
        self.v_penalty_function = np.vectorize(self.penalty_function)

        # The absolute value of the difference between two images is
        # for each pixel is multiplied with this value to account for very high values
        self.pixel_diff_discount = pixel_diff_discount

        # HSV channels means which HSV channels are of interests when looking at the
        # diff of two images in the calculate_penalty function.
        # Normaly only the hue is of interest
        # They are in PIL specified as [hue, sat, val]
        self.hsv_channels = hsv_channels if hsv_channels else [True, False, False]



    # Must receive images in HSV format for resonable performace. (RGB will give very random penalty values)
    def calculate_penalty(self, original_image, generated_image, return_diff_image=False):

        # Abort if the images are not of equal size
        assert original_image.size == generated_image.size

        # Convert images to np arrays and only keep the specified
        # hsv_channels by masking the third dimension with self.hsv_channels
        ori_data = np.asarray(original_image)[:, :, self.hsv_channels]
        gen_data = np.asarray(generated_image)[:, :, self.hsv_channels]

        # Get difference of the images and scale with the discount factor
        image_diff = np.absolute(ori_data-gen_data) * self.pixel_diff_discount

        # Feed every single pixel diff value through the penalty function
        image_diff_with_penalty = self.v_penalty_function(image_diff)

        # Sum the penalty diff for all values to get the total penalty
        penalty = np.sum(image_diff_with_penalty)

        if return_diff_image:
            return penalty, image_diff, image_diff_with_penalty
        else:
            return penalty

    # Paste all the cones according to the specified coordinates in the specified original image
    # original image must be in HSV!
    # original_image: PIL image in HSV format
    # cone_coords: [(x1,y1), (x2,y2), ...]
    # scales: [1.0, 0.8, ...]
    def create_generated_image(self, original_image, cone_coords:[tuple], scales:[float]):

        # If coords and scales dont match up there is a problem
        assert len(cone_coords) == len(scales)

        # For every cone coordinate and scale place the cone in the original image
        gen_image = copy.deepcopy(original_image)
        for (coord, scale) in zip(cone_coords, scales):
            gen_image = self.paste_cone_in_image(gen_image=gen_image, center_coord=coord, scale=scale)

        return gen_image


    # Paste one cone in the original image
    def paste_cone_in_image(self, gen_image, center_coord, scale):

        # Scale the cone_image and the cone_mask to specified size
        # Rounding errors are ignored
        resize_width = int(self.cone_image_width * scale)
        resize_height = int(self.cone_image_height * scale)
        cone = self.resize_image(self.hsv_cone_image, resize_width, resize_height)
        mask = self.resize_image(self.cone_alpha_mask, resize_width, resize_height)

        # Adjust the coord anchor point to the center of the cone
        # Note some minor rounding errors are ignored. They will not have significant impact.
        x_coord = (center_coord[0] - int(cone.shape[0])/2)
        y_coord = (center_coord[1] - int(cone.shape[1])/2)
        sub_frame = (x_coord, y_coord, cone.shape[0], cone.shape[1])

        # Place cone in the image with the alpha mask to ensure trasparancy around the cone
        gen_image_pasted = self.past_template(gen_image, template=cone, sub_frame=sub_frame)

        return gen_image_pasted

    # Paste the template in the numpy array and ignore alpha values
    def past_template(self, image, template, sub_frame):
        for x in range(sub_frame[2]):
            for y in range(sub_frame[3]):
                if template[x,y].any():
                    image[y + sub_frame[1], x + sub_frame[0]] = template[x, y]

        return gen_image

    @staticmethod
    # resize a image
    def resize_image(image, width, height):
        image = cv2.resize(image, (width, height))
        return image


# Testing
if __name__ == '__main__':
    # Load frame image
    frame_path = "../../data/videos/frames/track/frame0.jpg"
    frame_img = cv2.imread(frame_path)
    hsv_frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2HSV)

    # Load cone template
    cone_path = "./cone_template1.png"
    cone_img = cv2.imread(cone_path)

    # Create instance of ImageEvaluator
    ie = ImageEvaluator(cone_image=cone_img, hsv_channels=[True, False, False])

    # generate new image with cones
    coords = [(120, 80), (50,220)]
    scales = [0.3, 0.3]
    gen_image = ie.create_generated_image(original_image=hsv_frame_img,
                                          cone_coords=coords,
                                          scales=scales,)

    # Evaluate the generated image and receive penalty
    penalty = ie.calculate_penalty(original_image=frame_img,
                                   generated_image=gen_image)
    print(f"Penalty score: {penalty}")

    cv2.imshow(gen_image)
    hsv_frame_img.show()

    # Show diff images
    penalty, diff_image, penalty_diff_image = ie.calculate_penalty(original_image=frame_img,
                                                                   generated_image=gen_image,
                                                                   return_diff_image=True)

    cv2.imshow("1", diff_image)
    cv2.imshow("2", penalty_diff_image)
    cv2.waitKey(100000)

    # # Show generated image
    gen_image.show()

    # # Show original image
    hsv_frame_img.show()

