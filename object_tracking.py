import cv2
import numpy as np

# Function to select the region of interest (ROI)
def select_roi(event, x, y, flags, param):
    global roi_selected, roi_pts, frame, roi_box
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_pts = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        roi_pts.append((x, y))
        roi_selected = True
        roi_box = (roi_pts[0][0], roi_pts[0][1], roi_pts[1][0] - roi_pts[0][0], roi_pts[1][1] - roi_pts[0][1])
        if roi_box[2] > 0 and roi_box[3] > 0:  # Ensure ROI has positive size
            cv2.rectangle(frame, roi_pts[0], roi_pts[1], (0, 255, 0), 2)
        cv2.imshow('Frame', frame)

# Initialize variables
roi_selected = False
roi_pts = []
roi_box = None

# Video capture
cap = cv2.VideoCapture(0)

# Create a window and set mouse callback function
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', select_roi)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if roi_selected and roi_box[2] > 0 and roi_box[3] > 0:
        # Convert frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask for the ROI
        mask = cv2.inRange(hsv, np.array([0, 60, 32]), np.array([180, 255, 255]))

        # Compute the histogram backprojection
        roi_hist = cv2.calcHist([hsv], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # Apply the mean shift algorithm
        ret, roi_box = cv2.meanShift(back_proj, roi_box, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))

        # Draw the tracking result
        x, y, w, h = roi_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        roi_selected = False
        roi_pts = []

cap.release()
cv2.destroyAllWindows()
