import cv2

if __name__ == "__main__":
    #open camera
    cam = cv2.VideoCapture(0)
    
    #get width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))

    fourcc = cv2.VideoWriter_fourcc(*'mpv4')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width,frame_height))

    while True:
        ret, frame = cam.read()

        #write frame to output file
        out.write(frame)

        #display captured frame
        cv2.imshow('Camera',frame)

        #press q to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    out.release()
    cv2.destroyAllWindows()