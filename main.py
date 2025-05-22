import cv2
import mediapipe as mp

def is_finger_on_key(x, y, key_x, key_y, key_w, key_h):
    return key_x < x < key_x + key_w and key_y < y < key_y + key_h

if __name__ == "__main__":
    #open camera
    vid = cv2.VideoCapture(0)
    vid.set(3,960)
    
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
        ['Z','X','C','V','B','N','M']
    ]
    key_width = 60
    key_height = 60
    start_x = 50
    start_y = 50

    typed_text = ""
    last_pressed_key = None

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
        pressed_this_frame = False

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
                
                #default colour
                colour = (255,255,255)

                #check for press
                if finger_x and finger_y and is_finger_on_key(finger_x,finger_y, x, y, key_width, key_height):
                    if last_pressed_key != key:
                        typed_text += key
                        last_pressed_key = key
                    colour = (0,255,0)
                    pressed_this_frame = True

                cv2.rectangle(frame, (x, y), (x + key_width, y + key_height), colour, 2)
                cv2.putText(frame, key, (x + 20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

        #reset if no key pressed
        if not pressed_this_frame:
            last_pressed_key = None

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