from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

cap = cv2.VideoCapture('test_img_1.jpg')
model = YOLO('best.pt')
# names = model.name
# print(names)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # cv2.imshow('Video Stream', frame)
    result = model.predict(source=frame, stream=True, imgsz=640, show=False)
    for r in result:
        # Get a BGR numpy array of predictions
        im_array = r.plot()
        # Convert to RGB PIL image
        im = Image.fromarray(im_array[..., ::-1])
        im.show()
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()




# for r in model.predict(source=frame, stream=True, imgsz=512, show=False):
#     for c in r.boxes.cls:
#         if names[int(c)] == '0':
#             print("Yes it is Monkey\n")
#         else:
#             print("Human")