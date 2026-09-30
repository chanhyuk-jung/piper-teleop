import cv2

for i in range(6):
    try:
        cap = cv2.VideoCapture(i)
    except:
        continue

    ret, frame = cap.read()

    if ret:
        cv2.imwrite(f"{i}_img.jpg", frame)
        print(f"Frame saved successfully as {i}_img.jpg")
    else:
        print("Error: Could not read a frame from the camera.")

    cap.release()
