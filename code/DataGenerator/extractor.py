import numpy as np

def extract_pose_vectors(pose_landmarks):
    joint = np.zeros((33, 3))
    for j, lm in enumerate(pose_landmarks.landmark):
        joint[j] = [lm.x, lm.y, lm.z]
    
    v1_indices = [12, 11]  # 부모 관절 어깨
    v2_indices = [16, 15]  # 자식 관절 손목
    v1 = joint[v1_indices, :3]
    v2 = joint[v2_indices, :3]
    v = v2 - v1
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
    angle = np.arccos(np.einsum('nt,nt->n', v, v))  # 내적 계산
    angle = np.degrees(angle)
    angle = np.array([angle], dtype=np.float32)
    return np.concatenate([joint.flatten(), angle.flatten()])

def extract_hand_vectors(hand_landmarks_list):
    hand_data = []
    for res in hand_landmarks_list:
        joint = np.zeros((21, 3))
        for j, lm in enumerate(res.landmark):
            joint[j] = [lm.x, lm.y, lm.z]
        
        v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
        v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
        v = v2 - v1
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
        angle = np.arccos(np.einsum('nt,nt->n',
                                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
        angle = np.degrees(angle)
        angle = np.array([angle], dtype=np.float32)
        hand_data.append(np.concatenate([joint.flatten(), angle.flatten()]))
    return np.concatenate(hand_data)
