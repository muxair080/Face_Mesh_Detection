from faceMeshModule import FaceMeshDetector
import cv2
import time
def main():
    cap = cv2.VideoCapture('videos/10.mp4')

    pTime = 0
    cTime=0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set the size of the display window to match the size of the video frame
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", (width, height))

    detector = FaceMeshDetector()
     
    while True:
    
        success, img = cap.read()
        img = detector.fineFaceMesh(img)
    

        if not success:
            # print("Issue")
            break
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN,
        3, (0,255,0),3)
        cv2.imshow('Image', img)
        cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


     
if __name__ == '__main__':
    main();