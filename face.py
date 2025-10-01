import cv2
import os
import numpy as np
import time

# -------------------------------
# Simple Face Recognizer using Template Matching
# -------------------------------
class SimpleFaceRecognizer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.templates = {}  # Store template images for each person
        self.label_dict = {}
        
    def load_dataset(self):
        """Load dataset and create templates"""
        print("[INFO] Loading images from dataset...")
        
        current_id = 0
        for person in os.listdir(self.dataset_path):
            folder = os.path.join(self.dataset_path, person)
            
            if not os.path.isdir(folder):
                continue

            if not os.listdir(folder):
                print(f"[WARN] Empty folder: {person}")
                continue

            print(f"[INFO] Processing {person}...")
            self.label_dict[current_id] = person
            self.templates[current_id] = []

            image_count = 0
            for img_name in os.listdir(folder):
                img_path = os.path.join(folder, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    print(f"[WARN] Could not read image: {img_path}")
                    continue
                
                # Resize and normalize template
                template = cv2.resize(img, (100, 100))
                template = cv2.equalizeHist(template)  # Improve contrast
                self.templates[current_id].append(template)
                image_count += 1

            print(f"  -> Loaded {image_count} images for {person}")
            current_id += 1
        
        print(f"[INFO] Total: {sum(len(templates) for templates in self.templates.values())} images for {len(self.label_dict)} people")
        return len(self.templates) > 0
    
    def recognize_face(self, face_image):
        """Recognize face using template matching"""
        try:
            # Preprocess the input face
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
            query_face = cv2.resize(gray, (100, 100))
            query_face = cv2.equalizeHist(query_face)
            
            best_match_id = None
            best_confidence = 0
            
            # Compare with all templates
            for person_id, templates in self.templates.items():
                for template in templates:
                    # Use template matching
                    result = cv2.matchTemplate(query_face, template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    
                    # Update best match if this is better
                    if max_val > best_confidence:
                        best_confidence = max_val
                        best_match_id = person_id
            
            # Convert similarity to confidence (0-100 scale where higher is better)
            confidence = best_confidence * 100
            
            return best_match_id, confidence
            
        except Exception as e:
            print(f"[ERROR] Recognition error: {e}")
            return None, 0

# -------------------------------
# Main Application
# -------------------------------
def main():
    # 1. Dataset folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "dataset")

    if not os.path.exists(dataset_path):
        print("❌ Error: 'dataset' folder not found.")
        print(f"Looked for: {dataset_path}")
        return

    # 2. Initialize and load recognizer
    recognizer = SimpleFaceRecognizer(dataset_path)
    if not recognizer.load_dataset():
        print("❌ Error: Could not load any training data.")
        return

    # 3. Setup camera
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    if face_cascade.empty():
        print("❌ Error: Could not load face detector.")
        return

    # Try different camera backends
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, -1]
    cap = None
    for backend in backends:
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            print(f"✅ Camera opened with backend: {backend}")
            break

    if not cap or not cap.isOpened():
        print("❌ Error: Could not open camera.")
        return

    # 4. Real-time recognition loop
    print("\n🎥 Starting real-time face recognition...")
    print("Press 'q' to quit")
    print("Press 'r' to reset recognition")
    print("Press 't' to test current frame")
    
    # Set camera resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    recognition_history = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame.")
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            # Recognize face
            label_id, confidence = recognizer.recognize_face(face_roi)
            
            # Determine result
            if label_id is not None and confidence > 50:  # Confidence threshold
                name = recognizer.label_dict[label_id]
                color = (0, 255, 0)  # Green for recognized
                status = "RECOGNIZED"
            else:
                name = "Unknown"
                color = (0, 0, 255)  # Red for unknown
                status = "UNKNOWN"
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y - 35), (x + w, y), color, -1)
            cv2.putText(frame, f"{name} ({confidence:.1f}%)", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Store recognition result
            recognition_history.append({
                'name': name,
                'confidence': confidence,
                'timestamp': time.time()
            })
            
            # Keep only recent history
            if len(recognition_history) > 10:
                recognition_history.pop(0)

        # Display recognition statistics
        if recognition_history:
            recent_names = [entry['name'] for entry in recognition_history[-5:]]
            recognized_count = sum(1 for name in recent_names if name != "Unknown")
            cv2.putText(frame, f"Recent: {recognized_count}/5 recognized", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show frame
        cv2.imshow('Face Recognition - Press Q to quit', frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            recognition_history.clear()
            print("🔄 Recognition history cleared")
        elif key == ord('t'):
            if len(faces) > 0:
                print(f"\n🔍 Testing {len(faces)} detected faces:")
                for i, (x, y, w, h) in enumerate(faces):
                    face_roi = frame[y:y+h, x:x+w]
                    label_id, confidence = recognizer.recognize_face(face_roi)
                    name = recognizer.label_dict[label_id] if label_id is not None else "Unknown"
                    print(f"  Face {i+1}: {name} (confidence: {confidence:.1f}%)")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Application closed successfully")

# -------------------------------
# Run the application
# -------------------------------
if __name__ == "__main__":
    main()
