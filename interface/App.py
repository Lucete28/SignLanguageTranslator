import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model


seq_length = 30
pred_idx = 0
CANT_FIND_HAND_COUNT = 0
# set model
model = load_model(f"C:/PlayData/lstm_test43_1439act_e20_C8_B512.h5")
# set sessions
if 'camera' not in st.session_state:
    st.session_state.camera = False
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0  # 세션 상태에 FRAME_COUNT를 저장
if 'data_poket' not in st.session_state:
    st.session_state.data_poket = np.zeros((1, 156))
if 'sentence' not in st.session_state:
    st.session_state.sentence = [0]
# MediaPipe 손 인식 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

st.title("수어 인식 및 번역")

c1, c2 = st.columns(2)
if c1.button('Start'):
    st.session_state.camera = True

frame_placeholder = st.empty()
word_placeholder = st.empty()
state_placeholder = st.empty()
if c2.button('End'):
    st.session_state.camera = False
    # st.session_state.frame_count = 0  # 카메라가 꺼질 때 FRAME_COUNT 초기화

if st.session_state.camera:
    cap = cv2.VideoCapture(0)

    while cap.isOpened() and st.session_state.camera:
        ret, frame = cap.read()
        if not ret:
            break
        st.session_state.frame_count += 1  # 세션 상태를 사용하여 FRAME_COUNT 업데이트
        frame = cv2.flip(frame, 1)
        
        # 프레임 RGB 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        # 손 랜드마크 그리기                    
        if results.multi_hand_landmarks is not None:
            CANT_FIND_HAND_COUNT = 0
            da = []
            if len(results.multi_hand_landmarks) == 2 or len(results.multi_hand_landmarks) == 1:
                d= []
                for res in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS) # 랜드마크 그려주기
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
                    if len(results.multi_hand_landmarks)==1:
                        d.append(np.zeros_like(d[0]))
                da.append([np.concatenate(d)])
                if st.session_state.data_poket.size != 0:
                    st.session_state.data_poket = np.vstack([st.session_state.data_poket,np.concatenate(da)])
                    if st.session_state.data_poket.shape[0] >= 30:
                        input_data = np.expand_dims(np.array(st.session_state.data_poket[-seq_length:], dtype=np.float16), axis=0)
                        pred = model.predict(input_data, verbose=0).squeeze()
                        pred_idx= int(np.argmax(pred))
                else:
                    st.session_state.data_poket = da
        else: #손이 보이지 않을때
            CANT_FIND_HAND_COUNT +=1
            if CANT_FIND_HAND_COUNT >= 30 and st.session_state.sentence[-1]!=pred_idx:
                st.session_state.sentence.append(pred_idx)



        frame_placeholder.image(frame, channels="BGR", use_column_width=True)
        word_placeholder.write(f'WORD: {pred_idx}')
        state_placeholder.write(st.session_state.sentence)
    cap.release()
    hands.close()

# 카메라 작동 여부와 관계없이 FRAME_COUNT 표시
st.write(f'FRAME_COUNT: {st.session_state.frame_count}')
