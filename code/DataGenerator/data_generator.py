#  비디오 학습
import cv2
import mediapipe as mp
import numpy as np
import  os
from datetime import datetime
import random
from googletrans import Translator
from itertools import product


def make_data(act_ko, v_path): #단어와 영상주소 
    def apply_settings(image, angle, size = 1):
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, size)
        new_img = cv2.warpAffine(image, rotation_matrix, (width, height))
        return new_img

    #변수
    VIDEO_PATH = v_path
    ACTION = act_ko # cv2 출력 문제로 영어로 변경
    seq_length = 30
    created_time = datetime.now().strftime('%y_%m_%d')
    print('created at :',created_time)
    os.makedirs(f'dataset/{ACTION}', exist_ok=True)
    # 파라미터 값
    rotate_li = [0, 5, -5]  # 각 범위 축소(+- 10 삭제)
    speed_li = [1, 3, 5]
    size_li = [1, 1.25, 1.5]
    
    # 동영상 파일 열기
    cap = cv2.VideoCapture(VIDEO_PATH)

    # 동영상이 제대로 열렸는지 확인
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()



    gen_param = list(product(rotate_li, speed_li, size_li))
    random.shuffle(gen_param)
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
    for i, g_param in enumerate(gen_param):
        rotate, speed, size = g_param[0], g_param[1],g_param[2]
        print(f'{ACTION}, ({act_ko}), {i + 1}/{len(gen_param)}', f'speed : {speed}, rotated : {rotate}, size : {size}')

        frame_index = 1
        while True:
            ret, img = cap.read()
            if not ret: # 영상끝나면 종료
                break
            if frame_index % speed != 0:
                frame_index+=1
            else:  ### speed 조정
                frame_index+=1                
                img = apply_settings(img, rotate, size)
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

                        # 모든 관절과 각도를 1차원 배열로 합쳐 저장
                        pose_data.append(np.concatenate([joint.flatten(), angle.flatten()]))

                # #손 처리
                    
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
                        # da.append([np.concatenate(d)])
                        hand_data.append(np.concatenate(single_hand))        


                cv2.imshow('img', img)
            # if cv2.waitKey(int(1 * speed)) & 0xFF == ord('q'): # 속도조절 (delay 는 int 여야함 0이면 오류가능)
            if cv2.waitKey(1) & 0xFF == ord('q'): # 속도조절 (delay 는 int 여야함 0이면 오류가능)
                break
                # pass        
        # 동영상 다시재생
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        
    # 데이터 저장   
    # 배열화
    hand_array = np.array(hand_data)
    pose_array = np.array(pose_data)
    # 두 배열을 두 번째 축(axis=1)에서 결합
    data = np.concatenate((hand_array, pose_array), axis=1)

    # 시쿼스 분리
    full_seq_data = []
    for seq in range(len(data) - seq_length):
        full_seq_data.append(data[seq:seq + seq_length])

    full_seq_data = np.array(full_seq_data)
    # print(ACTION, full_seq_data.shape, data.shape) # 데이터 모양 확인

    # 파일 저장
    if len(full_seq_data.shape) ==3 :
        np.save(os.path.join(f'dataset/{ACTION}', f'raw_{created_time}'), data)
        # np.save(os.path.join(f'dataset/{ACTION}', f'seq_{created_time}_{full_seq_data.shape[0]}'), full_seq_data)
        print(ACTION,'데이터가 저장되었습니다. shape: ', data.shape)
    
    #종료
    cap.release()
    cv2.destroyAllWindows()

    # 비정상 폴더 삭제 (빈폴더)
    if os.path.exists(f'dataset/{ACTION}') and not os.listdir(f'dataset/{ACTION}'):
        os.rmdir(f'dataset/{ACTION}')  
        print(f"{f'dataset/{ACTION}'} 비정상 삭제")

make_data('tmp','https://sldict.korean.go.kr/multimedia/multimedia_files/convert/20200825/735712/MOV000240883_700X466.mp4')