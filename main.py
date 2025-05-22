import cv2

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    
    frame_width = int(cam.get(cv2.CAP_PROP_FRAM_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAM_WIDTH))