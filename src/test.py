import cv2
import numpy as np
import pandas as pd

import sys
import time
import ctypes
from pynput.mouse import Listener

import pytesseract

from threading import Thread

user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)


trial_cap = cv2.VideoCapture("input/videos/input_1.mp4")
trial_cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

distract_cap = cv2.VideoCapture("input/videos/AttentionAssessment.mp4")
distract_cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

# change to your pytesseract file path here
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\laura\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

start_time = time.time()

# length of experiment phases in seconds
ini_adjust_time = 10
trial_time = 1000
distraction_time = 30
brightness = 1
contrast = 1.25

min_fps = 30
max_fps = 72
fps = 50
fps_delta_time = 0

black_frames = np.zeros((screen_height, screen_width, 3), np.uint8)

phase = 0
prev_time = 0

# which frame the distraction video is on
pos_frame = distract_cap.get(cv2.CAP_PROP_POS_FRAMES)

participant_clicked = False
frame_changed = False

participant_input = False
letter_is_present = False

fields = ['time', 'brightness', 'fps', 'participant click', 'correct click', 'participant input', 'correct input']
values = [] # default value for values, if the program is successful (see the try block below), then it should give the test values, otherwise it will print an empty string: " "
data_collection = {}
data_output = "output/demo_00.xlsx" if len(sys.argv) < 2 or sys.argv[1] == None else sys.argv[1]

cv2.namedWindow("test")

# def on_click(x, y, button, pressed):
#     if pressed:
#         participant_clicked = True
#     return False

# with Listener(on_click=on_click) as listener:
#     listener.join()

def letter_detection(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (5,5), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations = 1)
    invert = 255 - opening
    data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')
    print(data[0])
    return data

def find_time():
    current_time = time.localtime(time.time())
    time_string = ":".join("0%s" % (current_time[i]) if current_time[i] < 10 else "%s" % (current_time[i]) for i in range (3, 6))
    return time_string

def countdown(t_seconds):
    time.sleep(1)
    t_seconds -= 1
    return t_seconds

def write_text(text, frame):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (10, 10), (w + 12, h + 12), (0, 255, 0), cv2.FILLED)
    cv2.putText(frame, text, (12, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

while True:
    delta_time = time.time() - start_time
    fps_delta_time += 1

    data_collection.update({'time' : find_time()})
    data_collection.update({'brightness' : brightness})
    data_collection.update({'fps' : fps})
    data_collection.update({'participant clicked' : participant_clicked})
    data_collection.update({'correct click': participant_clicked == (fps_delta_time % fps == 0)})

    if cv2.waitKey(2) == ord('A'):
        phase = (phase + 1) % 4

    print(f'phase: {phase}')
    # Initial Adjustment Phase
    if phase == 0 and delta_time <= 200:
        # Capture frame-by-frame
        ret, frame = trial_cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        brightness = 0.75
        frame[:,:,2] = np.clip(frame[:,:,2] + brightness, 0, 255) 
        write_text(f'brightness: {brightness}', frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        frame = cv2.resize(frame, (screen_width, screen_height))
        cv2.imshow("test", frame)
    
    # Trial
    if phase > 0 and phase <= 3:
        # Capture frame-by-frame
        ret, frame = trial_cap.read()

        # Adjusts brightness and framerate of black frames
        if cv2.waitKey(5) == ord('b'):
            # decrease brightness
            brightness -= 1 if brightness - 1  > 0 else 0
        if cv2.waitKey(5) == ord('B'):
            # increase brightness   
            brightness += 1 if brightness + 1 < 100 else 100
        if cv2.waitKey(5) == ord('f'):
            # decrease fps of black frames   
            fps -= 2 if fps - 2 > min_fps else min_fps
        if cv2.waitKey(5) == ord('F'):
            # increase fps of black frames  
            fps += 2 if fps + 2 < max_fps else max_fps

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame[:,:,2] = np.clip(frame[:,:,2] + brightness, 0, 255) 

        write_text(f'brightness: {brightness} | fps: {fps}', frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        frame = cv2.resize(frame, (screen_width, screen_height))
        cv2.imshow("test", frame)

        # Start distraction task
        if phase == 2:
            d_ret, slide = distract_cap.read()
            if d_ret:
                cv2.namedWindow("task")
                cv2.moveWindow("task", screen_width, 0)
                cv2.imshow("task", slide)
                pos_frame = distract_cap.get(cv2.CAP_PROP_POS_FRAMES)
                detected_letter = letter_detection(slide)
            if cv2.waitKey(10) == ord(' '):
                ~participant_input
                data_collection.update({'participant input' : True})
                data_collection.update({'correct input' : ('A' == detected_letter[0]) and participant_input})
                ~participant_input
            else: 
            # The next frame is not ready, so try to read it again
                distract_cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
                print(f'frame is not ready: position {pos_frame}')
                cv2.waitKey(50)
        if fps_delta_time % fps == 0:
            cv2.imshow("test", black_frames)
            fps_delta_time = 0
        elif phase > 2:
            something = 2
            # cv2.destroyWindow("task")

        values.append(data_collection.copy())

    if cv2.waitKey(10) & 0xFF == ord('q') or cv2.waitKey(10) == ord('q') or cv2.waitKey(10) == ord('C'):
        df = pd.DataFrame(values)
        df.to_excel(data_output)
        trial_cap.release()
        distract_cap.release()
        cv2.destroyAllWindows()
        break

# When everything is done, release the capture
trial_cap.release()
distract_cap.release()
cv2.destroyAllWindows()
