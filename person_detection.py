import sys
import os
import cv2
import torch
from ultralytics import YOLO

def main():
    if len(sys.argv) != 2:
        sys.stdout.write("Usage: python person_detection.py \"<modelPath>##im$$<imagePath>\"\n")
        sys.exit(1)

    # Parse input
    config_string = sys.argv[1].strip()
    if "##im$$" not in config_string:
        sys.stdout.write("Error: Incorrect format. Use '##im$$' to separate model and image path.\n")
        sys.exit(1)

    model_path, image_path = map(str.strip, config_string.split("##im$$"))

    if not os.path.exists(model_path):
        sys.stdout.write(f"Error: Model file '{model_path}' does not exist.\n")
        sys.exit(1)

    if not os.path.exists(image_path):
        sys.stdout.write(f"Error: Image file '{image_path}' does not exist.\n")
        sys.exit(1)

    # Load YOLO model
    try:
        model = YOLO(model_path)
        sys.stdout.write("✅ Model loaded successfully!\n")
    except Exception as e:
        sys.stdout.write(f"Error loading model: {e}\n")
        sys.exit(1)

    # Run inference
    try:
        results = model.predict(source=image_path, verbose=False)

        # Load the image
        image = cv2.imread(image_path)

        # Get YOLO's class labels
        names = model.names

        # Process results
        boxes = results[0].boxes

        if not boxes:
            sys.stdout.write("$$ Class Not Found $$\n")
        else:
            sys.stdout.write(f"✅ Inference successful!\n")
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                cls = int(box.cls)  # Class index
                conf = box.conf.item() * 100  # Convert confidence to percentage
                class_name = names[cls] if cls < len(names) else f"Unknown Class {cls}"

                # Draw bounding box
                color = (0, 255, 0)  # Green
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name}: {conf:.2f}%"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Print results
                sys.stdout.write(f"$$ {class_name}: {conf:.2f}% $$\n")

        # Save image in the same directory as the input image
        output_path = os.path.join(os.path.dirname(image_path), "output_" + os.path.basename(image_path))
        cv2.imwrite(output_path, image)
        sys.stdout.write(f"✅ Image saved at: {output_path}\n")

    except Exception as e:
        sys.stdout.write(f"Error during inference: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
