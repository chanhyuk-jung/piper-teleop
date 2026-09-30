import cv2

for i in range(8):
    cap = cv2.VideoCapture(i)

    ret, frame = cap.read()

    if ret:
        cv2.imwrite(f"{i}_img.jpg", frame)
        print(f"Frame saved successfully as {i}_img.jpg")
    else:
        print("Error: Could not read a frame from the camera.")

    cap.release()
