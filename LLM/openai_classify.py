import os
import base64
from openai import OpenAI
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import io

client = OpenAI(api_key='Enter your api key')

# 이미지를 base64로 인코딩하는 함수
def encode_image(image_path):
    try:
        with Image.open(image_path) as img:
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None


PROMPT = """
Prompt to enter characteristics for the things you want to classify

For example, if you want to classify dogs and cats, please enter the characteristics of dogs and cats
"""


# 이미지 분석 함수
def analyze_image(image_path):
    base64_image = encode_image(image_path)
    if base64_image is None:
        return None

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during API call for {image_path}: {str(e)}")
        return None


# 폴더 처리 함수
def process_folder(folder_path):
    results = []
    for category in ['class_name01', 'class_name02']:
        category_path = os.path.join(folder_path, category)
        for filename in os.listdir(category_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                image_path = os.path.join(category_path, filename)
                print(f"Processing {category}/{filename}...")
                analysis = analyze_image(image_path)
                if analysis is not None:
                    results.append((category, filename, analysis))
                else:
                    print(f"Skipping {category}/{filename} due to processing error")
    return results


# 분류 결과 추출 함수 수정
def extract_classification(analysis):
    classification = 'UNKNOWN'
    confidence = 0.0
    for line in analysis.split('\n'):
        if line.startswith('Classification:'):
            classification = line.split(':')[1].strip().upper()
        elif line.startswith('Confidence:'):
            conf_str = line.split(':')[1].strip().upper()
            confidence = {'LOW': 0.33, 'MEDIUM': 0.67, 'HIGH': 1.0}.get(conf_str, 0.0)
    return classification, confidence


# 성능 평가 함수 수정
def evaluate_performance(results):
    y_true = []
    y_pred = []
    y_scores = []
    for true_label, _, analysis in results:
        y_true.append(1 if true_label.upper() == 'class_name01' else 0)
        pred, conf = extract_classification(analysis)
        y_pred.append(1 if pred == 'class_name01' else 0)
        y_scores.append(conf if pred == 'class_name01' else 1 - conf)

    cm = confusion_matrix(y_true, y_pred)
    return y_true, y_pred, y_scores, cm


# 혼동 행렬 시각화 함수
def plot_confusion_matrix(cm):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks([0.5, 1.5], ['class_name02', 'class_name01'])
    plt.yticks([0.5, 1.5], ['class_name02', 'class_name01'])
    plt.show()


# 정밀도-재현율 곡선 시각화 함수
def plot_precision_recall_curve(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, marker='.')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()


# ROC 곡선 시각화 함수
def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


# 클래스별 성능 막대 그래프 시각화 함수
def plot_class_performance(cm):
    tn, fp, fn, tp = cm.ravel()
    real_accuracy = tp / (tp + fn)
    fake_accuracy = tn / (tn + fp)

    plt.figure(figsize=(10, 7))
    plt.bar(['class_name01', 'class_name02'], [real_accuracy, fake_accuracy])
    plt.title('Class-wise Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate([real_accuracy, fake_accuracy]):
        plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    plt.show()


# 메인 실행 코드
if __name__ == "__main__":
    folder_path = "/input/your/dataset/folder/path"

    results = process_folder(folder_path)

    if results:
        y_true, y_pred, y_scores, cm = evaluate_performance(results)

        # 시각화
        plot_confusion_matrix(cm)
        plot_precision_recall_curve(y_true, y_scores)
        plot_roc_curve(y_true, y_scores)
        plot_class_performance(cm)

        # 요약 통계 출력
        tn, fp, fn, tp = cm.ravel()
        print(f"\nclass_name01 accuracy: {tp / (tp + fn):.2f}")
        print(f"class_name02 accuracy: {tn / (tn + fp):.2f}")
        print(f"Overall accuracy: {(tp + tn) / (tp + tn + fp + fn):.2f}")
    else:
        print("No valid results to analyze. Please check your image files and API connection.")