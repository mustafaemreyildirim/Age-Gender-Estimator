'''
usage:
    videoprocess.py  --video_path=<str>    
'''

import cv2
from docopt import docopt
from face_align import Align
from PIL import Image
import pdb
import numpy as np
import pdb
args = docopt(__doc__)
path = args['--video_path']

try:
    path = int(path)
except:
    pass

a = Align()
video_capture = cv2.VideoCapture(path)


# Initialize variables
face_locations = []

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    try:
        fr = Image.fromarray(frame)
    # Find all the faces in the current frame of video
        frame = a(fr)
    except:
        pass
    # Display the resulting image
    frame = np.array(frame)
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
