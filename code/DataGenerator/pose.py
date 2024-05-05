import cv2
import mediapipe as mp

# Mediapipe 모듈 초기화
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Pose 및 Hands 모델 초기화
pose = mp_pose.Pose()
hands = mp_hands.Hands()

# 비디오 스트림 초기화
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # BGR -> RGB 변환
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Mediapipe를 사용한 포즈 및 손 랜드마크 인식
    pose_result = pose.process(img_rgb)
    hand_result = hands.process(img_rgb)

    # Pose 랜드마크 그리기
    if pose_result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Hand 랜드마크 그리기
    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 결과 이미지 출력
    cv2.imshow("Pose and Hand Landmarks", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
