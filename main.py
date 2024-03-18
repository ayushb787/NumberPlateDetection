from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import easyocr
# import tensorflow as tf

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

video_path = 'a.mp4'
cap = cv2.VideoCapture(video_path)
model = YOLO('best.pt')
reader = easyocr.Reader(['en'], gpu=True)  # English OCR

while cap.isOpened():
    success, frame = cap.read()

    if success:
        result = model.predict(source=frame, stream_buffer=False, conf=0.6, vid_stride=5, show_labels=True, show_conf=True)
        txt = ''
        for r in result:
            arr = r.boxes.xyxy
            im_bgr = r.plot()  # BGR-order numpy array
            im_rgb = Image.fromarray(im_bgr[..., ::-1])
            # im_rgb.show()

            if len(arr) > 0:  # Check if there are any detected regions
                for i in range(len(arr)):
                    xmin, ymin, xmax, ymax = map(int, arr[i])
                    img_predicted = im_bgr[ymin:ymax, xmin:xmax]
                    img_prediced_rgb = Image.fromarray(img_predicted[..., ::-1])
                    img_gray = img_prediced_rgb.convert('L')
                    img_gray_array = np.array(img_gray)
                    ocr_result = reader.readtext(img_gray_array)
                    for (bbox, text, prob) in ocr_result:
                        (top_left, top_right, bottom_right, bottom_left) = bbox
                        print(f'Text: {text}, Probability: {prob}')
                        txt = text
        annotated_frame = result[0].plot()
        annotated_frame = cv2.resize(annotated_frame, (640, 640))
        for (text, (x, y)) in zip([txt], [(10, 30)]):
            cv2.putText(annotated_frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()