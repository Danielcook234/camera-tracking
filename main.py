import cv2
import mediapipe as mp
from collections import defaultdict
import requests
import numpy as np
from io import BytesIO

def is_finger_on_key(x, y, key_x, key_y, key_w, key_h):
    return key_x < x < key_x + key_w and key_y < y < key_y + key_h

def fetch_image(query):
    print("fetching image")
    url = f"https://source.unsplash.com/640x480/?{query}"
    response = requests.get(url)
    if response.status_code == 200:
        image_bytes = np.asarray(bytearray(response.content),dtype=np.uint8)
        img = cv2.imdecode(image_bytes,cv2.IMREAD_COLOR)
        return img
    print("no image lol")
    return None

if __name__ == "__main__":
    #open camera
    vid = cv2.VideoCapture(0)
    vid.set(3,960)
    vid.set(4,720)
    
    #get width and height
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))

    #video writer
    fourcc = cv2.VideoWriter_fourcc(*'mpv4')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width,frame_height))

    #mediapipe hands
    mphands = mp.solutions.hands
    Hands = mphands.Hands(max_num_hands = 2, min_detection_confidence=0.7, min_tracking_confidence=0.6)
    mpdraw = mp.solutions.drawing_utils

    #virtual keyboard layout
    keys = [
        ['Q','W','E','R','T','Y','U','I','O','P'],
        ['A','S','D','F','G','H','J','K','L'],
        ['Z','X','C','V','B','N','M'],
        ['Space', 'Backspace', 'Enter']
    ]
    key_width = 60
    key_height = 60
    start_x = 50
    start_y = 50

    typed_text = ""
    hover_counts = defaultdict(int)
    hover_threshold = 10

    while True:
        success, frame = vid.read()
        if not success:
            break

        #mirror the image
        frame = cv2.flip(frame,1)

        #convert from bgr to rgb
        RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = Hands.process(RGBframe)

        finger_x = None
        finger_y = None
        key_hovered = None

        #draw hand landmarks
        if result.multi_hand_landmarks:
            for handLm in result.multi_hand_landmarks:
                mpdraw.draw_landmarks(frame,handLm,mphands.HAND_CONNECTIONS)
                #get index fingertip (landmark 8)
                index_fingertip = handLm.landmark[8]
                h,w,_ = frame.shape
                finger_x, finger_y = int(index_fingertip.x * w), int(index_fingertip.y * h)

        #draw keyboard
        for row_idx, row in enumerate(keys):
            for col_idx, key in enumerate(row):
                x = start_x + col_idx * key_width
                y = start_y + row_idx * key_height

                 # Adjust width for long keys
                if key == 'Space':
                    w = key_width * 3
                elif key == 'Backspace' or key == 'Enter':
                    w = key_width * 2
                else:
                    w = key_width
                
                #default colour
                colour = (255,255,255)

                #check for press
                if finger_x and finger_y and is_finger_on_key(finger_x,finger_y, x, y, key_width, key_height):
                    hover_counts[key] += 1
                    if hover_counts[key] >= hover_threshold:
                        if key == 'Space':
                            typed_text += ' '
                        elif key == 'Backspace':
                            typed_text = typed_text[:-1]
                        elif key == 'Enter':
                            query = typed_text.strip()
                            fetched_image = fetch_image(query)
                            if fetched_image is not None:
                                cv2.imshow("Search result", fetched_image)
                            typed_text = ""
                        else:
                            typed_text += key
                        hover_counts.clear()
                    colour = (0,255,0)
                else:
                    colour = (200,200,200)
                    hover_counts[key] = 0

                # Draw key background
                cv2.rectangle(frame, (x, y), (x + key_width, y + key_height), colour, -1)
                # Draw key border
                cv2.rectangle(frame, (x, y), (x + key_width, y + key_height), (0, 0, 0), 2)
                # Draw key label
                label_x = x + 10 if w == key_width else x + 5
                cv2.putText(frame, key, (label_x, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


        # Draw typed text above keyboard
        cv2.rectangle(frame, (40, 400), (900, 460), (50, 50, 50), -1)
        cv2.putText(frame, f"Typed: {typed_text}", (50, 445), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        #write frame to output file
        out.write(frame)

        #display captured frame
        cv2.imshow('Camera',frame)

        #press q to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    vid.release()
    out.release()
    cv2.destroyAllWindows()