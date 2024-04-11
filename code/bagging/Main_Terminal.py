""" READ ME
Container들 에서 전달 받은 결과를 종합 및 로컬에 전송
#1. 로컬에서 모든 컨테이너에 배열 전송
2. 터미널에서 우선 수령후 컨테이너에 전송

1. 완료응답 Count 
2. Vote
3. 한글 변환
4. 전송
"""
# cd C:\Users\oem\Desktop\jhy\signlanguage\SignLanguageTranslator\code\bagging\; uvicorn Main_Terminal:app --reload --host 0.0.0.0 --port 8080
import httpx
from httpx import Timeout
from fastapi import FastAPI, Request,HTTPException
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from collections import Counter
import requests
import asyncio
import pickle
from datetime import datetime, timedelta, timezone
import time

KST = timezone(timedelta(hours=9))

current_time = datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')



# PATH_LIST = ['13.124.201.219:54205','52.79.75.52:53637','13.124.40.13:54075']
PATH_LIST = ['13.124.201.219:54205','52.79.75.52:53637']
app = FastAPI()

with open(r'C:\Users\oem\Desktop\jhy\signlanguage\SignLanguageTranslator\logs\1645_act_list.pkl', 'rb') as file:
        actions = pickle.load(file)
        print(len(actions),'개의 액션이 저장되어있습니다.')
"""
@app.post("/receive")
async def receive_data(request: Request):
    data = await request.body()  # 요청 본문을 바이트 스트림으로 수신

    async with httpx.AsyncClient(timeout=Timeout(60,connect=60)) as client:
        for path in PATH_LIST:
            # 수신된 요청 본문을 그대로 다른 엔드포인트로 비동기적으로 전달
            response = await client.post(f"{path}/receive", content=data)
            print(response.json())
    return {"message": "Data forwarded success"}
"""
@app.post("/receive")
async def receive_data(request: Request):
    startTime = time.time()
    data = await request.body()

    async def send_request(path):
        async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=60)) as client:
            response = await client.post(f"http://{path}/receive", content=data)
            return response.json()

    # 각 경로에 대한 요청을 병렬로 실행
    responses = await asyncio.gather(*(send_request(path) for path in PATH_LIST))

    # for response in responses:
    #     print(response)
    endTime=time.time()
    print(f"recieve and send take {endTime-startTime}")
    return {"message": "Data forwarded successfully"}


@app.get("/Word_End")
async def Word_End():
    WORD_LIST=[]

    async with httpx.AsyncClient() as client:
        tasks = (client.get(f'http://{path}/confirm') for path in PATH_LIST)  
        responses = await asyncio.gather(*tasks)  
        for response in responses:
            data = response.json()
            words = data["pred_list"]
            WORD_LIST.extend(words)
            word_idx = Counter(WORD_LIST).most_common()[0][0]

    return {"CODE":True,"word": actions[word_idx],'is_array_here':False}  # 최빈단어 반환


# @app.get("/WhatIsThisWord")
# async def WhatIsThisWord():
#     while len(WORD_LIST) < 3:
#         await asyncio.sleep(0.001)  # 대기
#     word_idx = Counter(WORD_LIST).most_common[0][0]
#     return {"CODE":True,"word": actions[word_idx],'is_array_here':False}  # 최빈단어 반환






@app.get("/")
def test():
    return {f"{current_time}": f"This is Main terminal"}