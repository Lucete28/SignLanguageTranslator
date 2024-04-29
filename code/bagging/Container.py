""" READ ME
Bagging을 위해 컨테이너 분리 후 서버 별 API 작성
1. 할당된 모델을 준비
2. 배열을 전달 받음
3. 예측을 기록
4. Voting 결과를 메인 터미널 서버로 전송
- pip install -r requirements.txt
- uvicorn Container:app --reload --host 0.0.0.0 --port 8000
""" 
# cd C:/Users/oem/Desktop/jhy/signlanguage/SignLanguageTranslator/code/bagging; uvicorn Container1:app --reload --host 0.0.0.0 --port 800
from tensorflow.keras.models import load_model
import tensorflow as tf
from fastapi import FastAPI, Request,HTTPException
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from collections import Counter
import requests
import glob
CONTAINER_ID = 3  # 0~10
CONTAINER_SIZE = 3
GPU_NUM = 0
MODELS = []
MODEL_PATH = 'C:/Users/oem/Desktop/jhy/signlanguage/SignLanguageTranslator/model/2024-03-11_23-04-15G300D6'
PREDICT_LIST =[ [] for _ in range(CONTAINER_SIZE) ] #[[a],[b],[c]]

# GPU 설정
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:        
#         tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#     except RuntimeError as e:
#         print(e)

# 모델 준비
for i in range(CONTAINER_SIZE):
    print(f'{i+CONTAINER_ID*3}번 모델 load')
    model_pattern = f"./models/lstm_test103_G{i+CONTAINER_ID*3}_*.h5"
    model_file = glob.glob(model_pattern)[0]
    model = load_model(model_file)
    MODELS.append(model)
print('All models ready')


def model_predict(model, array):
    pred = model.predict(array, verbose=0).squeeze()
    return int(np.argmax(pred))


app = FastAPI()


@app.post("/receive") # 데이터 전송 받기
async def receive_array(request: Request):
    data = await request.json()
    # 배열 변환
    array_list = data['array']
    array = np.array(array_list, dtype=np.float16) 

    #스레딩
    with ThreadPoolExecutor(max_workers=25) as executor:
        future_to_model = {executor.submit(model_predict, model, array): i for i, model in enumerate(MODELS)}
        for future in as_completed(future_to_model):
            model_index = future_to_model[future]
            try:
                result = future.result()
            except Exception as exc:
                return {"CODE": False, "status": f'Model {model_index} generated an exception: {exc}'}
            else:
                PREDICT_LIST[model_index].append(result)
    # print(len(PREDICT_LIST[0]),PREDICT_LIST[0])
    return {"status": "array received", "shape": array.shape, "CODE": True, "tmp" : PREDICT_LIST[0]} #TODO 성공결과 반함이 속도에 미치는 영향확인 필요


@app.get("/confirm")  # 완료 결과를 터미널로 전송
def confirm():
    global PREDICT_LIST  # 전역 변수 사용 선언
    organize_li = []
    result = []
    act_len = 0
    for re in PREDICT_LIST:
        if re:
            most_common_num, most_common_count = Counter(re).most_common(1)[0]
            organize_li.append(most_common_num)
            re.clear()
            result.append([re, most_common_num, most_common_count])
            # act_len = len(re)
    PREDICT_LIST = [[] for _ in range(CONTAINER_SIZE)]  # 전역 변수 재할당
    if organize_li:
        data = {"CODE": True, "pred_list": organize_li}
        print(data, "전송")
        return data
    else:
        return {"status": "NO DATA", "CODE": False}






@app.get("/")
def test():
    return {f"Container ID : {CONTAINER_ID}"}
