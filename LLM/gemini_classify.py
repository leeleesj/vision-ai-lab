import os
import base64
import google.generativeai as genai
from PIL import Image

# Configure the library
genai.configure(api_key='Enter your api key')

# Set up the model
model = genai.GenerativeModel('gemini-1.5-flash')

PROMPT = """
Prompt to enter characteristics for the things you want to classify

For example, if you want to classify dogs and cats, please enter the characteristics of dogs and cats
"""


def analyze_image(image_path):
    image = Image.open(image_path)

    response = model.generate_content([PROMPT, image])

    return response.text


def process_folder(folder_path):
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            print(f"Processing {filename}...")
            analysis = analyze_image(image_path)
            results.append((filename, analysis))
    return results


# Process images
folder_path = 'input/dataset/path'
results = process_folder(folder_path)

# Print results
for filename, analysis in results:
    print(f"\nFile: {filename}")
    print(analysis)
    print("-" * 50)