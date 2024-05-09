import cv2
import mediapipe as mp
import numpy as np
import  os
from datetime import datetime
import random
from googletrans import Translator
from itertools import product
import json
from filenoti import filenoti as fn #라인 알림

cap = cv2.VideoCapture(0)

    # 동영상이 제대로 열렸는지 확인
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

###############################

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
pose = mp_pose.Pose()
hand_data = []
pose_data = []

frame_index = 1
while True:
    ret, img = cap.read()
    if not ret: # 영상끝나면 종료
        break
    else:  ### speed 조정
        frame_index+=1                
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hand_result = hands.process(img)
        pose_result = pose.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # #pose 처리
        if pose_result.pose_landmarks and hand_result.multi_hand_landmarks is not None:
            if len(hand_result.multi_hand_landmarks) == 2 or len(hand_result.multi_hand_landmarks) == 1:
                mp_drawing.draw_landmarks(img, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                # 랜드마크 정보를 가져오기
                joint = np.zeros((33, 3))  
                for j, lm in enumerate(pose_result.pose_landmarks.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]
                v1_indices = [12, 11]  # 부모 관절 어깨
                v2_indices = [16, 15]  # 자식 관절 손목
                
                v1 = joint[v1_indices, :3]
                v2 = joint[v2_indices, :3]
                v = v2 - v1

                # 정규화
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # 벡터 사이의 각도 계산
                angle = np.arccos(np.einsum('nt,nt->n', v, v))  # 내적 계산
                angle = np.degrees(angle)
                angle = np.array([angle], dtype=np.float32)
                raw_pose_data = np.concatenate([joint.flatten(), angle.flatten()])

                if not np.isnan(raw_pose_data).any(): # nan 값 확인
                    # 모든 관절과 각도를 1차원 배열로 합쳐 저장
                    pose_data.append(raw_pose_data)
                    if np.isnan(raw_pose_data).any():
                        print('err occur in pose!!!!!!')
                        break
        #           #손 처리
                    single_hand = []
                    for res in hand_result.multi_hand_landmarks:  # res 잡힌 만큼 (max 손 개수 이하)
                        mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
                        joint = np.zeros((21, 3))
                        for j, lm in enumerate(res.landmark):
                            joint[j] = [lm.x, lm.y, lm.z]

                        # Compute angles between joints
                        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                        v = v2 - v1 # [20, 3]
                        # Normalize v
                        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                        # Get angle using arcos of dot product
                        angle = np.arccos(np.einsum('nt,nt->n',
                            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]\

                        angle = np.degrees(angle) # Convert radian to degree
                        angle = np.array([angle], dtype=np.float32)
                        single_hand.append(np.concatenate([joint.flatten(),angle.flatten()]))
                        if len(hand_result.multi_hand_landmarks)==1:
                            single_hand.append(np.zeros_like(single_hand[0]))
                    hand_data.append(np.concatenate(single_hand))        


    if cv2.waitKey(1) & 0xFF == ord('q'): # 속도조절 (delay 는 int 여야함 0이면 오류가능)
        break

cap.release()
cv2.destroyAllWindows()