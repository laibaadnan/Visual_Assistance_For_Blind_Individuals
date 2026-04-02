import os
import pandas as pd
from tqdm import tqdm
import requests
import time

# Function to check connection to a specific URL
def check_url_connection(url, timeout=5):
    try:
        response = requests.head(url, timeout=timeout)
        return response.status_code == 200
    except requests.ConnectionError:
        return False

# Function to download file with progress feedback
def download_file(url, local_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(local_path, 'wb') as f, tqdm(
                desc=local_path,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
    else:
        print(f"Failed to download {url}")

# Define the output directory
output_dir = 'openimages_v7'

# List of classes you want to download
classes = [
    'Tree', 'Car', 'Laptop', 'Mobile phone', 'Stairs', 'Plant', 'Bottle', 'Chair', 'Table', 'Person', 'Flowerpot', 'Clock', 'Light switch', 'Cat', 'Dog', 'Cupboard'
]

# Mapping of class names to their corresponding Open Images labels
class_descriptions_url = "https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv"

# Check URL connection before starting
if not check_url_connection(class_descriptions_url):
    print("Cannot reach the class descriptions URL. Please check the connection or URL.")
    exit(1)

print("Connection to class descriptions URL verified. Starting download...")

print("Downloading class descriptions...")
class_descriptions_df = pd.read_csv(class_descriptions_url)
class_name_to_label = {name: label for label, name in zip(class_descriptions_df['LabelName'], class_descriptions_df['DisplayName'])}
class_labels = [class_name_to_label[class_name] for class_name in classes]

# Ensure the output directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)

# Download images and annotations in smaller batches
annotations_bbox_url = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"

print("Downloading bounding box annotations...")
if not check_url_connection(annotations_bbox_url):
    print("Cannot reach the bounding box annotations URL. Please check the connection or URL.")
    exit(1)

# Use the download_file function to download the bounding box annotations
annotations_bbox_path = os.path.join(output_dir, 'annotations_bbox.csv')
download_file(annotations_bbox_url, annotations_bbox_path)

annotations_bbox_df = pd.read_csv(annotations_bbox_path)
annotations_bbox_df = annotations_bbox_df[annotations_bbox_df['LabelName'].isin(class_labels)]

# Limit the number of images per class
max_images_per_class = 200

for class_label, class_name in zip(class_labels, classes):
    print(f"Downloading images and annotations for class: {class_name}")
    class_images = annotations_bbox_df[annotations_bbox_df['LabelName'] == class_label]['ImageID'].unique()[:max_images_per_class]

    for image_id in tqdm(class_images, desc=f"Downloading {class_name} images"):
        image_url = f"https://storage.googleapis.com/openimages/2018_04/train/{image_id}.jpg"
        image_path = os.path.join(output_dir, 'images', f"{image_id}.jpg")
        
        # Check URL connection before each download
        if not check_url_connection(image_url):
            print(f"Cannot reach the image URL: {image_url}. Skipping this image.")
            continue
        
        if not os.path.exists(image_path):
            response = requests.get(image_url, timeout=10)
            if response.status_code == 200:
                with open(image_path, 'wb') as f:
                    f.write(response.content)
            else:
                print(f"Failed to download {image_url}")

        image_annotations = annotations_bbox_df[annotations_bbox_df['ImageID'] == image_id]
        yolo_annotations = []
        for _, row in image_annotations.iterrows():
            x_min, x_max = row['XMin'], row['XMax']
            y_min, y_max = row['YMin'], row['YMax']
            class_index = class_labels.index(row['LabelName'])
            yolo_annotations.append(f"{class_index} {x_min} {y_min} {x_max - x_min} {y_max - y_min}")
        
        annotation_path = os.path.join(output_dir, 'annotations', f"{image_id}.txt")
        with open(annotation_path, 'w') as f:
            f.write("\n".join(yolo_annotations))

print("Downloading completed.")