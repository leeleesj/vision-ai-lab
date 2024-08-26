import os
import cv2
from glob import glob
from tqdm import tqdm

def extract_frames(video_path, output_folder):
    try:
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            print(f"경고: {video_path}에서 프레임을 읽을 수 없습니다.")
            return

        for count in tqdm(range(total_frames), desc="프레임 추출 중"):
            success, image = video.read()
            if not success:
                print(f"경고: 프레임 {count}를 읽는 데 실패했습니다.")
                break

            cv2.imwrite(os.path.join(output_folder, f"frame_{count:04d}.jpg"), image)

    except Exception as e:
        print(f"오류: {video_path} 처리 중 예외 발생 - {str(e)}")
    finally:
        video.release()

def process_videos(input_folder):
    video_files = glob(os.path.join(input_folder, "*.*"))

    for video_file in video_files:
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        output_folder = os.path.join(input_folder, video_name)
        try:
            os.makedirs(output_folder, exist_ok=True)
            extract_frames(video_file, output_folder)
            print(f"Processed {video_name}")
        except Exception as e:
            print(f"error creating output folder: {e}")

        extract_frames(video_file, output_folder)


if __name__ == "__main__":
    # input_folder = input("Enter the path to the folder containing video files: ")
    input_folder = "your/input/video/path"
    process_videos(input_folder)