# Real_Time_Object_Tracking
This project demonstrates a real-time object tracking system using histogram backprojection and the mean shift algorithm. The system allows a user to select a region of interest (ROI) in a video stream, and then tracks that object as it moves within the frame.

## Results
![GIFMaker_me track_5mb](https://github.com/user-attachments/assets/dd79e74f-5d6b-4252-9a35-46904ca5d765)


## Features
- **Object Selection:** Users can select an object by clicking and dragging to define an initial ROI.

- **Histogram Backprojection:** Utilizes histogram backprojection to create a probability map indicating where the object is likely to be in subsequent frames based on its color distribution.

- **Mean Shift Tracking:** Implements the mean shift algorithm to iteratively maximize the similarity between the histogram backprojection and the current image patch.

- **Dynamic Threshold Adjustment:** Includes a trackbar interface to adjust the motion sensitivity threshold dynamically during runtime for improved tracking accuracy.

## Requirements
- Python 3.x
- OpenCV 4.x

## Installation
Install dependencies:
``` pip install -r requirements.txt```
## Usage
- Run the main script
- The application window will open displaying the video stream from the default camera.
- **Select Object:** Click and drag to select the object of interest in the video stream.
- **Track Object:** The system will track the selected object using the mean shift algorithm based on histogram backprojection.
- **Adjust Sensitivity:** Use the "Motion Sensitivity" trackbar to adjust the sensitivity threshold dynamically.
- Press **'q'** to exit the application.

## Troubleshooting
If the object tracking is not effective, try adjusting the initial ROI selection or modifying the parameters of histogram backprojection and mean shift algorithm.
