import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os
from collections import Counter
from sentence_api import make_sentence
import json
import requests
import pickle
import threading

def post_json_request(end_point,data):
    array_list = data.tolist()
    url = f'http://203.250.133.192:8000/{end_point}'
    requests.post(url, json={"array": array_list})
    
with open('G:/내 드라이브/LAB/Sign_Language_Remaster/logs/api_log.json',encoding='utf-8') as json_file:
    dic = json.load(json_file)
    dic = dic['Daily']
with open(r'G:\내 드라이브\LAB\Sign_Language_Remaster\logs\act_list.pkl', 'rb') as file:
    # 리스트 로드
    actions = pickle.load(file)
    print(len(actions),'개의 액션이 저장되어있습니다.')
seq_length = 30
action = '?'
# model = load_model(r"C:/PlayData/lstm_test100_9act_e50_C0_B0.h5")
# url = f'http://203.250.133.192:8000/certification'
# requests.post(url, 'wjdghdus')

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)


data = np.zeros((1, 156))
action_seq = []
this_action = '?'
class_select = []
word_list = []
is_array_threre = False
CANT_FIND_HAND_COUNT = 0
thread = False
thread_count = 0
while cap.isOpened():
    ret, img = cap.read()
    # img0 = img.copy()

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        CANT_FIND_HAND_COUNT = 0
        da = []
        if len(result.multi_hand_landmarks) == 2 or len(result.multi_hand_landmarks) == 1:
            d= []
            for res in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS) # 랜드마크 그려주기
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
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                angle = np.degrees(angle) # Convert radian to degree
                #######################################
                angle = np.array([angle], dtype= np.float32)
                d.append(np.concatenate([joint.flatten(),angle.flatten()]))
                if len(result.multi_hand_landmarks)==1:
                    d.append(np.zeros_like(d[0]))
            da.append([np.concatenate(d)])
            if data.size != 0:
                data = np.vstack([data,np.concatenate(da)])
            else:
                data = da
        else:
            pass

        if len(data) < seq_length:
            continue

        # input_data = np.expand_dims(np.array(data[-seq_length:], dtype=np.float32), axis=0)
        # # api 호출
        # array_list = input_data.tolist()
        # url = 'http://203.250.133.192:8000/receive'

        # # POST 요청으로 바이너리 데이터 전송
        # requests.post(url, json={"array": array_list})
        thread_count+=1
        print(thread_count)
        if thread_count %2 == 0:
            input_data = np.expand_dims(np.array(data[-seq_length:], dtype=np.float16), axis=0)
            thread = threading.Thread(target=post_json_request, args=('receive',input_data,))
            thread.start()
        
        is_array_threre = True

    else:
        CANT_FIND_HAND_COUNT+=1
        if CANT_FIND_HAND_COUNT>10 and thread:
            if not thread.is_alive() and is_array_threre: # 손이 안보인지 10 프레임, 이전 스레드 완료, 서버에 배열 있음 
            #confirm 요청
                url = 'http://203.250.133.192:8000/confirm'
                response = requests.get(url).json() ##TODO 과부화 처리
                if response['CODE']:
                    is_array_threre = response['is_array_here']
                    action = actions[response['most_common_pred']]
                    print(action)
                    word_list.append(action)
                else:
                    action = 'NO DATA'
                CANT_FIND_HAND_COUNT = 0
        cv2.putText(img, f'{action.upper()}',org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 255), thickness=2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'): # 종료
        cv2.destroyAllWindows()
        break
    if cv2.waitKey(1) == ord('z'): # gpt api request
        # cv2.putText(img, f'Request sucess',org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 255), thickness=2)
        print(word_list)
        ans = make_sentence(word_list[1:])
        word_list = []
        print(ans)