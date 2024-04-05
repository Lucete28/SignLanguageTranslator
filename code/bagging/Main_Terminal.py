""" READ ME
Container들 에서 전달 받은 결과를 종합 및 로컬에 전송
#1. 로컬에서 모든 컨테이너에 배열 전송
2. 터미널에서 우선 수령후 컨테이너에 전송

1. 완료응답 Count 
2. Vote
3. 한글 변환
4. 전송
"""

from fastapi import FastAPI, Request,HTTPException
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from collections import Counter
import requests
import asyncio
import pickle

PATH_LIST = []
app = FastAPI()

with open(r'G:\내 드라이브\LAB\SignLanguageTranslator\logs\act_pkl\V2_A300.pkl', 'rb') as file:
        actions = pickle.load(file)
        # print(len(actions),'개의 액션이 저장되어있습니다.')

@app.post("/receive") # 로컬에서 데이터 전송 받아서 컨테이너에 뿌리기
def receive_array(request: Request):
    data = request.json()
    # 배열 변환
    array_list = data['array']
    array = np.array(array_list, dtype=np.float16) 
    for path in PATH_LIST:
        response = requests.post(f"{path}/recive",data=array)



WORD_LIST= []
@app.post("/Word_End")
async def Word_End(request: Request):
    data = await request.json()
    words = data["pred_list"]
    WORD_LIST.extend(words)
    return {"status": f"You completed {len(WORD_LIST)}st time."}

        
@app.get("/WhatIsThisWord")
async def WhatIsThisWord():
    while len(WORD_LIST) < 33:
        await asyncio.sleep(0.001)  # 대기
    word_idx = Counter(WORD_LIST).most_common[0][0]
    return {"CODE":True,"word": actions[word_idx],'is_array_here':False}  # 최빈단어 반환






@app.get("/")
def test():
    return {f"message": f"This is Main terminal"}