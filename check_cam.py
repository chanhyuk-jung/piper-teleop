from pathlib import Path

import cv2

paths = list(Path("/dev").glob("video*"))[3:]
print(list(paths))

for path in paths:
    try:
        cap = cv2.VideoCapture(str(path))
    except:
        continue

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        exit()

    ret, frame = cap.read()

    if ret:
        cv2.imwrite(f"{path.name}_img.jpg", frame)
        print(f"Frame saved successfully as {path.name}_img.jpg")
    else:
        print("Error: Could not read a frame from the camera.")

    cap.release()
