import numpy as np
import pandas as pd
import cv2
import glob
import matplotlib.pyplot as plt


DATASET_PATH = "../data/single_cone_images/"  # Path to the images to be classified
YELLOW_HSV_RANGE_LOWER = [15, 80, 80]    # HSV bottom threashold for yellow cone classification
YELLOW_HSV_RANGE_UPPER = [30, 255, 240]  # HSV top threashold for yellow cone classification
BLUE_HSV_RANGE_LOWER = [30, 100, 120]    # HSV bottom threashold for blue cone classification
BLUE_HSV_RANGE_UPPER = [179, 255, 255]   # HSV top threashold for blue cone classification
VISUALIZE_PERFORMANCE = True             # If true the script will generate graphs to visualize the performance

# Load the data into a pandas dataframe
# (overkill for this project, but why not :))
def create_df(dataset_path):
    img_paths = []
    colors = []

    # Iterate over all the images in the dataset folder
    for path in glob.glob(dataset_path + "*"):

        # If blue in the path then set the label to blue,
        # if yellow in the path set the label yellow or otherwise skip datapoint
        if "yellow" in path:
            color = "yellow"
        elif "blue" in path:
            color = "blue"
        else:
            print("Invalid data sample was skipped")
            continue

        # Append to list of datapoints
        img_paths.append(path)
        colors.append(color)

    # Create pandas df from the datapoint lists
    df = pd.DataFrame(list(zip(img_paths, colors)), columns=["img_path", "color"])

    return df


# Count the number of yellow and blue pixels in the image based on the color ranges
def count_colored_pixels(img, yellow_lower, yellow_upper, blue_lower, blue_upper):
    # Convert image to HSV format
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Calculate yellow and blue pixels in the image:
    # - Create a yellow and blue pixel mask mapping all pixels within
    #   the yellow or blue range to 255 and other pixels to 0
    # - the mask is then divided with 255 to make it a binary mask
    #   mapping pixels within the color range to 1 and other pixels to 0
    yellow_pixels_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper) / 255
    blue_pixels_mask = cv2.inRange(hsv_image, blue_lower, blue_upper) / 255

    # - Sum all pixels to get the total count of yellow respectively blue pixels
    yellow_pixel_count = np.sum(yellow_pixels_mask)
    blue_pixel_count = np.sum(blue_pixels_mask)

    return yellow_pixel_count, blue_pixel_count, yellow_pixels_mask, blue_pixels_mask

# Display all images and each of their blue and yellow pixel masks
def visualize_performance(df, pixel_data):
    fig = plt.figure(figsize=(30, 5))

    # Display original images with filename labels
    for idx in range(len(df)):
        fig.add_subplot(1, len(df), idx+1)
        image = cv2.imread(df["img_path"][idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        np_image = np.asarray(image)
        plt.axis('off')
        name = df["img_path"][idx].split("/")[-1]
        plt.title(name, fontsize=4)
        plt.imshow(np_image)

    # Display yellow filter masks
    for idx in range(len(df)):
        fig.add_subplot(2, len(df), idx+1 + len(df))
        plt.axis('off')
        plt.imshow(pixel_data[idx][2])

    # Display blue filter masks
    for idx in range(len(df)):
        fig.add_subplot(4, len(df), idx+1 + len(df) * 2)
        plt.axis('off')
        plt.imshow(pixel_data[idx][3])

    plt.show()


# Starting point of the script
if __name__ == '__main__':
    # Load dataset
    df = create_df(DATASET_PATH)

    # Add a column which describes every color as a binary number
    # 0: represents yellow, 1: represents blue
    df['binary_color'] = np.where(df['color'] == 'yellow', 0, 1)

    # Make np arrays out of the color ranges
    yellow_lower = np.array(YELLOW_HSV_RANGE_LOWER)
    yellow_upper = np.array(YELLOW_HSV_RANGE_UPPER)
    blue_lower = np.array(BLUE_HSV_RANGE_LOWER)
    blue_upper = np.array(BLUE_HSV_RANGE_UPPER)

    # Loop over each datapoint in the dataset and make a prediction
    # If data is supposed to be visualized save all pixel data for every image
    predictions = []
    pixel_data_list = []
    for img_path in df["img_path"]:
        # Open the image from the path
        image = cv2.imread(img_path)

        # Count the total amount of pixels in the blue and yellow ranges
        pixel_data = count_colored_pixels(img=image,
                                          yellow_lower=yellow_lower,
                                          yellow_upper=yellow_upper,
                                          blue_lower=blue_lower,
                                          blue_upper=blue_upper)

        # Save pixel data if visualizing active
        if VISUALIZE_PERFORMANCE:
            pixel_data_list.append(pixel_data)

        # Only use the count of blue and yellow colored pixels to predict color
        yellow_count, blue_count = pixel_data[:2]

        # If there are more yellow pixels it is most likely a yellow cone,
        # hence append 0 to the predictions
        if yellow_count > blue_count:
            predictions.append(0)

        # Otherwise it is most likely a blue cone
        # hence append 1 to the predictions
        else:
            predictions.append(1)

    # Add predictions to df
    df["prediction"] = predictions

    # Count occurences where binary_color == prediction (correct prediction)
    correct_predictions = len(df[df['binary_color'] == df["prediction"]])

    # Calculate accuracy
    total_predictions = len(df["prediction"])
    accuracy = correct_predictions / total_predictions

    # Print accuracy
    print(f"Accuracy: {accuracy}  ({correct_predictions}/{total_predictions})")

    # Visualize if it is set to true
    if VISUALIZE_PERFORMANCE:
        visualize_performance(df=df, pixel_data=pixel_data_list)
