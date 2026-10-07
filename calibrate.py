"""Click road rectangle corners in TL, TR, BR, BL order."""
import cv2
import numpy as np

import config
from speed import video_source


points = []


def on_click(event, x, y, _flags, _parameter):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        print("Point {}: [{}, {}]".format(len(points), x, y))


capture = cv2.VideoCapture(video_source(config.VIDEO_PATH))
ok, frame = capture.read()
capture.release()
if not ok:
    raise SystemExit("Could not read the first frame from VIDEO_PATH: {}".format(config.VIDEO_PATH))

window = "Road calibration - click TL, TR, BR, BL; Q to finish"
cv2.namedWindow(window)
cv2.setMouseCallback(window, on_click)
while True:
    image = frame.copy()
    for index, point in enumerate(points):
        cv2.circle(image, tuple(point), 7, (0, 0, 255), -1)
        cv2.putText(image, str(index + 1), (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if len(points) > 1:
        cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, (0, 255, 0), 2)
    cv2.imshow(window, image)
    if cv2.waitKey(20) & 0xFF in (ord("q"), 27):
        break
cv2.destroyAllWindows()
if len(points) != 4:
    raise SystemExit("Select exactly four points before quitting; selected {}.".format(len(points)))
print("\nPaste into config.py:\nSOURCE_POINTS = {}".format(points))
