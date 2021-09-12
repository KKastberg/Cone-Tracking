import cv2
import os
import glob
from subprocess import run, PIPE


def bash(cmd: str) -> str:
    split_cmd: [] = cmd.split(" ")
    process = run(split_cmd, text=True, stdout=PIPE)
    return str(process.stdout)

def resize_image(img, size):
    return cv2.resize(img, size)

def resize_all_images(input_path, output_path, size, name):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    counter = 0
    for path in glob.glob(input_path + "*"):
        try:
            img = cv2.imread(path)
            resized = resize_image(img, size)
            cv2.imwrite(output_path + f"/{name}{counter}.jpg", resized)
            counter += 1
        except:
            pass

    print(f"Resized {counter - 1} images")

def create_neg_info_file(file_path, data_path):
    with open(file_path, "w") as f:
        for path in glob.glob(data_path + "*"):
            print(path)
            line = path[2:] + "\n"
            f.write(line)

def create_pos_samples(pos_path, num, max_angle, bg_path, info_path, output_path):
    counter = 0

    os.mkdir(output_path)

    # Create photos for every image in subfolders
    for path in glob.glob(pos_path + "*"):
        os.mkdir(output_path + str(counter))

        stdout = bash(f"opencv_createsamples -img {path} -bg {bg_path} " +
                      f"-info {output_path + str(counter) + '/' + info_path} " +
                      f"-pngoutput {output_path} -maxxangle {max_angle[0]}" +
                      f"-maxyangle {max_angle[1]} -maxzangle {max_angle[2]} -num {num}")
        print(stdout)
        print(f"Dir{counter} created")
        counter += 1

    # Combine all subfolders and concat the info.lst file
    for (idx, dir) in enumerate(glob.glob(output_path + "*")):
        dir_name = dir.split("/")[-1]
        for file in glob.glob(dir + "/*"):
            file_name = file.split("/")[-1]
            if file_name == "info.lst":
                info_file = open(file, "r").read().strip()
                new_info_file = dir_name + "_" + info_file.replace("\n", "\n" + dir_name + "_") + "\n"
                with open(output_path + "info.lst", "a") as f:
                    f.write(new_info_file)
            else:
                bash(f"mv {file} {output_path + dir_name + '_' + file_name}")
        bash(f"rm -r {dir}")
        print(f"Dir{dir_name} finished")
        # opencv_createsamples -img "./pos/pos12.jpg" -bg "bg.txt" -info "info.lst" -pngoutput "info" -maxxangle 0.5 -maxyangle -0.5 -maxzangle 0.5 -num 50


def create_vec_file(info_path, info_file, size, output_vec):
    num = len(glob.glob(info_path + "*.jpg"))
    stdout = bash(f"opencv_createsamples -info {info_file} -num {num} -w {size[0]} -h {size[1]} -vec {output_vec}")
    print(stdout)


if __name__ == '__main__':
    # Resize negative images
    # resize_all_images(input_path="../Negative Images/",
    #                   output_path="./neg",
    #                   size=(100, 100),
    #                   name="neg")

    # Create negative info file
    # create_neg_info_file(file_path="./bg.txt",
    #                      data_path="./neg/")

    # Resize positive images
    # resize_all_images(input_path="../../exc1/single_cone_images/",
    #                   output_path="./pos/",
    #                   size=(50, 50),
    #                   name="pos")

    # Generate positive samples with opencv
    create_pos_samples(pos_path="./pos/",
                       num=150,
                       max_angle=(0.5, -0.5, 0.5),
                       bg_path="./bg.txt",
                       info_path="/info.lst",
                       output_path="./info/")

    create_vec_file(info_path="./info/",
                    info_file="./info/info.lst",
                    size=(32,32),
                    output_vec="./positives.vec")


    # Active nodes:
    # scp -r /Users/kevin/Docs/Studier/KTH/Formula Student/Rercuitment2/haar/training_deployment ubuntu@192.168.1.58:
    # scp -r /Users/kevin/Docs/Studier/KTH/Formula Student/Rercuitment2/haar/training_deployment ubuntu@192.168.1.80:


