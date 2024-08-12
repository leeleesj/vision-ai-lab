import os
import argparse
import cv2
import numpy as np
import coremltools as ct
import tensorflow as tf
from PIL import Image
from ultralytics import YOLO

def load_model(model_path):
    if model_path.endswith('.pt'):
        return YOLO(model_path), 'torch'
    elif model_path.endswith('.mlmodel'):
        return ct.models.MLModel(model_path), 'mlmodel'
    elif model_path.endswith('.tflite'):
        return tf.lite.Interpreter(model_path=model_path), 'tflite'
    else:
        raise ValueError("Unsupported model format. Use .pt, .mlmodel, .tflite")

def process_image(image_path, model, model_type):
    try:
        img = Image.open(image_path)
        img_array = np.array(img)

        if model_type == 'torch':
            results = model(img_array)
            boxes = results[0].boxes
            if len(boxes) == 0:
                raise ValueError("No object detected")

            # Get the box with the highest confidence
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
            resized_img = cv2.resize(img_array, (320, 320))
            input_data = np.expand_dims(resized_img, axis=0).astype(np.float32)
            model.set_tensor(input_details[0]['index'], input_data)
            model.invoke()
            coordinates = model.get_tensor(output_details[0]['index'])[0]

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

def draw_bbox(image_path, bbox):
    img = cv2.imread(image_path)
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    return img

def process_directory(input_dir, output_dir, model, model_type):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                try:
                    bbox = process_image(input_path, model, model_type)

                    img_with_bbox = draw_bbox(input_path, bbox)
                    cv2.imwrite(output_path, img_with_bbox)

                    print(f"Processed: {input_path}")
                except Exception as e:
                    print(f"Failed to process {input_path}: {str(e)}")
                    continue

def main():
    parser = argparse.ArgumentParser(description='Object Detection')
    parser.add_argument('input_dir', help='Input directory containing images')
    parser.add_argument('output_dir', help='Output directory for processed images')
    parser.add_argument('--model', required=True, help='Path to the model file (.pt, .mlmodel, or .tflite)')
    args = parser.parse_args()

    model, model_type = load_model(args.model)
    process_directory(args.input_dir, args.output_dir, model, model_type)

if __name__ == '__main__':
    main()

'''
    python object_detection.py path/to/your/input_dataset path/to/your/output_dataset --model path/to/your/model.pt
    python object_detection.py path/to/your/input_dataset path/to/your/output_dataset --model path/to/your/model.mlmodel
    python object_detection.py path/to/your/input_dataset path/to/your/output_dataset --model path/to/your/model.tflite
'''