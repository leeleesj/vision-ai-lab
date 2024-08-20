import os
import argparse
import cv2
import numpy as np
import coremltools as ct
import tensorflow as tf
from PIL import Image
from torch.optim.optimizer import required
from ultralytics import YOLO

def load_model(model_path):
    if model_path.endswith('.pt'):
        model = YOLO(model_path)
        return model, 'torch'
    elif model_path.endswith('.mlmodel'):
        model = ct.models.MLModel(model_path)
        return model, 'mlmodel'
    elif model_path.endswith('.tflite'):
        model = tf.lite.Interpreter(model_path=model_path)
        return model, 'tflite'
    else:
        raise ValueError("Unsupported model format. Use .pt, .mlmodel, .tflite")

#
# def get_model_info(model, model_type):
#     info = f"Model Type: {model_type}\n"
#
#     if model_type == 'torch':
#         info += f"Model Name: {model.name}\n"
#         info += f"Model Task: {model.task}\n"
#         info += f"Model Stride: {model.stride}\n"
#     elif model_type == 'mlmodel':
#         info += f"Model Description: {model.get_spec().description}\n"
#         info += f"Model Version: {model.get_spec().version}\n"
#     elif model_type == 'tflite':
#         model.allocate_tensors()
#         input_details = model.get_input_details()
#         output_details = model.get_output_details()
#         info += f"Input Shape: {input_details[0]['shape']}\n"
#         info += f"Input Type: {input_details[0]['dtype']}\n"
#         info += f"Output Shape: {output_details[0]['shape']}\n"
#         info += f"Output Type: {output_details[0]['dtype']}\n"
#
#     return info


def process_image(image_path, model, model_type):
    try:
        img = Image.open(image_path)
        img_array = np.array(img)

        if model_type == 'torch':
            results = model(img_array)
            boxes = results[0].boxes
            if len(boxes) == 0:
                raise ValueError("No object detected")
            best_box = boxes[boxes.conf.argmax()]
            x1, y1, x2, y2 = best_box.xyxy[0]
            return (int(x1), int(y1), int(x2), int(y2))

        elif model_type == 'mlmodel':
            resized_img = img.resize((320, 320))
            prediction = model.predict({'image': resized_img})
            coordinates = prediction['coordinates'][0]
        elif model_type == 'tflite':
            model.allocate_tensors()
            input_details = model.get_input_details()
            output_details = model.get_output_details()

            input_shape = input_details[0]['shape']
            input_dtype = input_details[0]['dtype']

            resized_img = cv2.resize(img_array, (input_shape[1], input_shape[2]))

            if input_dtype == np.float32:
                input_data = resized_img.astype(np.float32) / 255.0
            elif input_dtype == np.uint8:
                input_data = resized_img.astype(np.uint8)
            else:
                raise ValueError(f"Unsupported input data type: {input_dtype}")

            input_data = np.expand_dims(input_data, axis=0)

            model.set_tensor(input_details[0]['index'], input_data)
            model.invoke()

            output_data = model.get_tensor(output_details[0]['index'])[0]

            best_detection = output_data[np.argmax(output_data[:, 4])]
            y1, x1, y2, x2 = best_detection[:4]

            height, width = img_array.shape[:2]
            x1, y1, x2, y2 = x1 * width, y1 * height, x2 * width, y2 * height

            return (int(x1), int(y1), int(x2), int(y2))

        if model_type in ['mlmodel', 'tflite']:
            if len(coordinates) == 0:
                raise ValueError("No object detected")

            x_center, y_center, width, height = coordinates
            x1 = int((x_center - width / 2) * img.width)
            y1 = int((y_center - height / 2) * img.height)
            x2 = int((x_center + width / 2) * img.width)
            y2 = int((y_center + height / 2) * img.height)

            return (max(0, x1), max(0, y1), min(img.width, x2), min(img.height, y2))

    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")


def crop_image(image_path, bbox):
    img = Image.open(image_path)
    return img.crop(bbox)


def draw_bbox(image_path, bbox):
    img = cv2.imread(image_path)
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    return img


def process_directory(input_dir, output_dir, model, model_type, mode):
    # model_info = get_model_info(model, model_type)
    # print("Model Information:")
    # print(model_info)
    # print("Processing images...")

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                try:
                    bbox = process_image(input_path, model, model_type)

                    if mode == 'crop':
                        cropped_img = crop_image(input_path, bbox)
                        cropped_img.save(output_path)
                    elif mode == 'draw':
                        img_with_bbox = draw_bbox(input_path, bbox)
                        cv2.imwrite(output_path, img_with_bbox)
                    print(f"Processed: {input_path}")
                except Exception as e:
                    print(f"Failed to process {input_path}: {str(e)}")
                    continue


def main():
    parser = argparse.ArgumentParser(description='Object Detection')
    parser.add_argument('--input_dir', required=True, help='Input directory containing images')
    parser.add_argument('--output_dir', required=True, help='Output directory for processed images')
    parser.add_argument('--model', required=True, help='Path to the model file (.pt, .mlmodel, or .tflite)')
    parser.add_argument('--mode', choices=['crop', 'draw'], required=True, help='Processing mode: crop or draw')
    args = parser.parse_args()

    model, model_type = load_model(args.model)
    process_directory(args.input_dir, args.output_dir, model, model_type, args.mode)


if __name__ == '__main__':
    main()

    '''
        python detection/detect.py path/to/your/input/datatset path/to/your/output/datatset --model path/to/your/model.mlmodel --mode crop
        python detection/detect.py path/to/your/input/datatset path/to/your/output/datatset --model path/to/your/model.tflite --mode draw
    '''