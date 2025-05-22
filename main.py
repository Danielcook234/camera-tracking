import cv2
import mediapipe as mp

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
    start_y = 300

    while True:
        success, frame = vid.read()
        if not success:
            break

        #convert from bgr to rgb
        RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = Hands.process(RGBframe)

        if result.multi_hand_landmarks:
            for handLm in result.multi_hand_landmarks:
                for id, lm in enumerate(handLm.landmark):
                    h,w,_ = frame.shape
                    cx,cy = int(lm.x*w), int(lm.y*h)
                    mpdraw.draw_landmarks(frame,handLm,mphands.HAND_CONNECTIONS)

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