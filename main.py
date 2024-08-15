from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import easyocr
# import tensorflow as tf
import re
import threading
from firebaseutil import findNoPlate


# video_path = 'sample_1.mp4'
# video_path=0
video_path = 'sample_2.mp4'
cap = cv2.VideoCapture(video_path)
model = YOLO('100EpochBest.pt')
# model = YOLO('no_plate_detection_8_epoch_best.pt')
reader = easyocr.Reader(['en'], gpu=True)  # English OCR
indian_number_plate_pattern = r'^[A-Za-z]{2}\d{2}[A-Za-z]{0,3}\d{0,4}$'
while cap.isOpened():
    success, frame = cap.read()

    if success:
        result = model.predict(source=frame, stream_buffer=False, conf=0.6, vid_stride=0, show_labels=True,
                               show_conf=True)
        txt = ''
        for r in result:
            arr = r.boxes.xyxy
            im_bgr = r.plot()
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
                        print("Raw Text - ", text)
                        text = text.replace(" ", "")
                        pattern = r'[^a-zA-Z0-9\s]'
                        text = re.sub(pattern, '', text)
                        print("Preprocessed Text - ", text)

                        if not re.match(indian_number_plate_pattern, text):
                            (top_left, top_right, bottom_right, bottom_left) = bbox
                            print(f'Text: {text}, Probability: {prob}')
                            txt = text
                            if len(txt) == 6:
                                threading.Thread(target=findNoPlate, args=(txt,)).start()
        annotated_frame = result[0].plot()
        annotated_frame = cv2.resize(annotated_frame, (640, 640))

        (text_width, text_height), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

        padding_x = 10
        padding_y = 10
        rect_x = 10 - padding_x
        rect_y = 30 - text_height - padding_y
        rect_width = text_width + 2 * padding_x
        rect_height = text_height + 2 * padding_y

        cv2.rectangle(annotated_frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (192, 192, 192),
                      -1)

        for (text, (x, y)) in zip([txt], [(10, 30)]):
            cv2.putText(annotated_frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.imshow("YOLOv8 Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()