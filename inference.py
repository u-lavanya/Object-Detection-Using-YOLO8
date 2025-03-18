import os
import cv2
from ultralytics import YOLO

# Define paths
WEIGHTS_DIR = r"/Users/lava/Downloads/Person and PPE detection/weights"
PERSON_MODEL_PATH = os.path.join(WEIGHTS_DIR, "best_person.pt")
PPE_MODEL_PATH = os.path.join(WEIGHTS_DIR, "best_ppe.pt")
INPUT_DIR = r"/Users/lava/Downloads/Person and PPE detection/input_directory"
OUTPUT_DIR = r"/Users/lava/Downloads/Person and PPE detection/output_directory"

# Ensure output directories exist
def create_output_dirs(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "cropped"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotated"), exist_ok=True)

# Detect persons in images
def detect_persons(input_dir, output_dir, person_model, ppe_model):
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No images found in input directory.")
        return

    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        output_image_path = os.path.join(output_dir, "annotated", "output_" + image_file)
        image = cv2.imread(image_path)

        try:
            results = person_model.predict(source=image_path, verbose=False)
            boxes = results[0].boxes
            if not boxes:
                print(f"No persons found in {image_file}")
                continue

            person_boxes = []
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = person_model.names[int(box.cls)]
                conf = box.conf.item() * 100
                color = (0, 255, 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name}: {conf:.2f}%"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if class_name.lower() == "person":
                    cropped_person = image[y1:y2, x1:x2]
                    cropped_filename = f"{os.path.splitext(image_file)[0]}_person{i}.jpg"
                    cropped_path = os.path.join(output_dir, "cropped", cropped_filename)
                    cv2.imwrite(cropped_path, cropped_person)
                    person_boxes.append((x1, y1, cropped_path))
                    detect_ppe(ppe_model, cropped_path, image, x1, y1)

            cv2.imwrite(output_image_path, image)
            print(f"Annotated image saved at: {output_image_path}\n")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

# Detect PPE on cropped persons
def detect_ppe(model, image_path, full_image, x_offset, y_offset):
    results = model.predict(source=image_path, verbose=False)
    boxes = results[0].boxes
    if not boxes:
        print(f"No PPE detected in {os.path.basename(image_path)}")
        return

    cropped_image = cv2.imread(image_path)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_name = model.names[int(box.cls)]
        conf = box.conf.item() * 100
        color = (0, 0, 255)
        cv2.rectangle(cropped_image, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name}: {conf:.2f}%"
        cv2.putText(cropped_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        full_x1, full_y1, full_x2, full_y2 = x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset
        cv2.rectangle(full_image, (full_x1, full_y1), (full_x2, full_y2), color, 2)
        cv2.putText(full_image, label, (full_x1, full_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    ppe_output_path = os.path.join(os.path.dirname(image_path), "ppe_" + os.path.basename(image_path))
    cv2.imwrite(ppe_output_path, cropped_image)
    print(f"PPE annotated image saved at: {ppe_output_path}")

# Main function
if __name__ == "__main__":

    create_output_dirs(OUTPUT_DIR)

    # Load YOLO models
    person_model = YOLO(PERSON_MODEL_PATH)
    ppe_model = YOLO(PPE_MODEL_PATH)

    # Run detection
    detect_persons(INPUT_DIR, OUTPUT_DIR, person_model, ppe_model)
