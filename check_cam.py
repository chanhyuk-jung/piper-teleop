import cv2

# 1. Initialize the video capture object (0 is usually the built-in webcam)
cap = cv2.VideoCapture(2)

# Check if the camera opened correctly
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# 2. Grab a single frame
# ret is a boolean (True if successful), frame is the actual image array
ret, frame = cap.read()

if ret:
    # 3. Save the frame to your computer
    cv2.imwrite("captured_frame.jpg", frame)
    print("Frame saved successfully as 'captured_frame.jpg'!")
else:
    print("Error: Could not read a frame from the camera.")

# 4. Always release the camera resource when finished
cap.release()
