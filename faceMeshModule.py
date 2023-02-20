import cv2
import time
import mediapipe as mp


class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, 
    minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
    
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)


    
    def fineFaceMesh(self, img, draw=True):

        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        if self.results.multi_face_landmarks:
            # print('inside if.')
            for faceLms in self.results.multi_face_landmarks:
                # print(faceLms)
                self.mpDraw.draw_landmarks(img, faceLms, mp.solutions.face_mesh.FACEMESH_TESSELATION, 
                self.drawSpec, self.drawSpec)

                for id, lm in enumerate(faceLms.landmark):

                    ih, iw, ic = img.shape
                    # find pixles of each landmark
                    x, y = int(lm.x* iw), int(lm.y*ih)
                    # print(id, x,y)

        
        return img



