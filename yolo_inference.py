import cv2
import numpy as np
from ultralytics import YOLO

def process_video(input_path, output_path):

    model = YOLO('yolov8x.pt')

    cap = cv2.VideoCapture(input_path)
    w, h, fps = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    lower_court, upper_court = int(h * 0.60), int(h * 0.40)
    prev_p1, prev_p2 = None, None


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, classes=0, verbose=False)[0].boxes.xywh.cpu().numpy()
        lower, upper = [], []

        for box in results:
            x, y, w, h = box
            if y > lower_court:
                lower.append(box)
            elif y < upper_court:
                upper.append(box)

        p1 = max(lower, key=lambda b: b[1]) if lower else prev_p1
        p2 = min(upper, key=lambda b: b[1]) if upper else prev_p2
        prev_p1, prev_p2 = p1, p2  # Save last detected positions

        for label, player in zip(['Player 1', 'Player 2'], [p1, p2]):
            if player is not None:
                x, y, w, h = player
                cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x - w/2), int(y - h/2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        out.write(frame)
        cv2.imshow('Tennis Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(), out.release(), cv2.destroyAllWindows()


process_video(r'C:\Users\YY\Desktop\Tennis System\Input_videos\Tennis Video.mp4', r'C:\Users\YY\Desktop\Tennis System\runs\detect\track\output_video.mp4')
