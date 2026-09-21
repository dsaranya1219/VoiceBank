import cv2
import time

def capture_face(filename="user_face.jpg"):
    print("Opening camera... Look directly at the webcam.")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False

    # Allow camera sensor to warm up/adjust lighting
    time.sleep(1)
    
    ret, frame = cap.read()
    cap.release()

    if ret and frame is not None:
        cv2.imwrite(filename, frame)
        print(f"Face image captured and saved successfully as {filename}.")
        return True
    else:
        print("Failed to capture image from camera.")
        return False

if __name__ == "__main__":
    capture_face()