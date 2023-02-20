import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture('videos/10.mp4')

pTime = 0
cTime=0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=2, circle_radius=5) 

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set the size of the display window to match the size of the video frame
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", (width, height))



while True:

    success, img = cap.read()
    if not success:
        print("Issue")
        break
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            print(faceLms)
            mpDraw.draw_landmarks(img, faceLms, mp.solutions.face_mesh.FACEMESH_TESSELATION, 
            drawSpec, drawSpec)

            for id, lm in enumerate(faceLms.landmark):

                ih, iw, ic = img.shape
                x,y = int(lm.x* iw), int(lm.y*ih)
                print(id, x,y)
    

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