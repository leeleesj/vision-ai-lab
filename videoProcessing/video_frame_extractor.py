import os
import cv2
from glob import glob

def extract_frames(video_path, output_folder):
    video = cv2.VideoCapture(video_path)

    # initialize video frame count
    count = 0
    while True:
        success, image = video.read()
        if not success:
            break

        cv2.imwrite(os.path.join(output_folder, f"frame_{count:04d}.jpg"), image)
        count += 1

    video.release()

def process_videos(input_folder):
    video_files = glob(os.path.join(input_folder, "*.*"))

    for video_file in video_files:
        video_name = os.path.splitext(os.path.basename(video_file))[0]

        output_folder = os.path.join(input_folder, video_name)
        os.makedirs(output_folder, exist_ok=True)

        extract_frames(video_file, output_folder)

        print(f"Processed {video_name}")


if __name__ == "__main__":
    # input_folder = input("Enter the path to the folder containing video files: ")
    input_folder = "your/input/video/path"
    process_videos(input_folder)