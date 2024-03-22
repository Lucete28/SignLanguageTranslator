#  비디오 학습
import cv2
import mediapipe as mp
import numpy as np
import time, os
import random
from googletrans import Translator
from itertools import product


def cut_first(text):
    t = text.split(',')[0]
    return t

def trans_to_english(text):
    text = cut_first(text)
    translator = Translator()
    result = translator.translate(text, src='ko', dest='en')
    return result.text
##############################
def make_data(act_ko, v_path):
    def rotate_image(image, angle, size = 1):
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, size)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated_image

    #변수
    VIDEO_PATH = v_path

    ACTION = trans_to_english(act_ko) # cv2 출력 문제로 영어로 변경
    seq_length = 30
    created_time = int(time.time())
    print('created at :',created_time)
    os.makedirs(f'dataset/{ACTION}', exist_ok=True)

    # 동영상 파일 열기
    cap = cv2.VideoCapture(VIDEO_PATH)

    # 동영상이 제대로 열렸는지 확인
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    # # 동영상의 프레임 수와 크기 
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rotate_li = [0, 5, -5, 10, -10]
    speed_li = [1, 3, 5]
    size_li = [1, 1.25, 1.5]

    gen_param = list(product(rotate_li, speed_li, size_li))
    random.shuffle(gen_param)
    ###############################

    # MediaPipe hands model
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    data = []
    # repeat = 1

    for i, g_param in enumerate(gen_param):
        rotate, speed, size = g_param[0], g_param[1],g_param[2]
        print(f'{ACTION}, ({act_ko}), {i + 1}/{len(gen_param)}', f'speed : {speed}, rotated : {rotate}, size : {size}')
        # repeat +=1 
        frame_index = 1
        while True:
            ret, img = cap.read()
            if not ret: # 영상끝나면 종료
                break

            if frame_index % speed != 0:
                frame_index+=1
            else:  ### speed 조정
                frame_index+=1
                
                img = rotate_image(img, rotate, size)
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if result.multi_hand_landmarks is not None:
                    da = [] 
                    if len(result.multi_hand_landmarks) == 2 or len(result.multi_hand_landmarks) == 1:
                        d = []
                        for res in result.multi_hand_landmarks:  # res 잡힌 만큼 (max 손 개수 이하)
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
                            d.append(np.concatenate([joint.flatten(),angle.flatten()]))
                            if len(result.multi_hand_landmarks)==1:
                                d.append(np.zeros_like(d[0]))
                        da.append([np.concatenate(d)])
                        data.append(np.concatenate(da))        


                cv2.imshow('img', img)
            # if cv2.waitKey(int(1 * speed)) & 0xFF == ord('q'): # 속도조절 (delay 는 int 여야함 0이면 오류가능)
            if cv2.waitKey(1) & 0xFF == ord('q'): # 속도조절 (delay 는 int 여야함 0이면 오류가능)
                break
                # pass
            # # 동영상 속도에 따라 프레임 위치 설정
            # frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) + speed
            # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        # 동영상 다시재생
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        
        
    # 배열화
    data = np.array(data)
    data = data.reshape(data.shape[0],data.shape[-1])

    # 시쿼스 분리
    full_seq_data = []
    for seq in range(len(data) - seq_length):
        full_seq_data.append(data[seq:seq + seq_length])

    full_seq_data = np.array(full_seq_data)
    # print(ACTION, full_seq_data.shape, data.shape) # 데이터 모양 확인

    # 파일 저장
    if len(full_seq_data.shape) ==3 :
        np.save(os.path.join(f'dataset/{ACTION}', f'raw_{created_time}'), data)
        np.save(os.path.join(f'dataset/{ACTION}', f'seq_{created_time}_{full_seq_data.shape[0]}'), full_seq_data)
        print(ACTION,'saved', full_seq_data.shape, data.shape)
    cap.release()
    cv2.destroyAllWindows()

    # 비정상 폴더 삭제 (빈폴더)
    if os.path.exists(f'dataset/{ACTION}') and not os.listdir(f'dataset/{ACTION}'):
        os.rmdir(f'dataset/{ACTION}')  
        print(f"{f'dataset/{ACTION}'} 빈폴더 삭제")

print("This file is Data_generator.py \nThis file only have Function.\nPlease run ### run_generator.py ### file")