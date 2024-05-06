import requests
import xmltodict
import json
# from data_generator import trans_to_english
from datetime import datetime
import time
from tqdm import tqdm
from data_generator import *
# import datetime
#####################################################################################################
# C:/Users/oem/AppData/Local/Programs/Python/Python38/python.exe c:/Users/oem/Desktop/jhy/signlanguage/SignLanguageTranslator/code/DataGenerator/run_generator.py

def write_txt_log(T_path, text):
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')

    with open(T_path, 'a', encoding='utf-8') as file:
        file.write(f"{formatted_time} ::: {text}\n")

def write_json_log(J_PATH, data):
    with open(J_PATH, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

def update_json_log(JSON_LOG_PATH, page, job_todo):
    with open(JSON_LOG_PATH, "r", encoding='utf-8') as file:
        data = json.load(file)

    data["Daily"][page] = job_todo

    with open(JSON_LOG_PATH, "w", encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
#####################################################################################################
        



def get_source(start_page, repeat=10): #    해야하는 페이지 받아서 return item_li 반환
    print(repeat,'번 반복예정')
    TXT_LOG_PATH = r'C:\Users\oem\Desktop\jhy\signlanguage\signLanguageTranslator\logs\LOG.TXT'
    JSON_LOG_PATH = r'C:\Users\oem\Desktop\jhy\signlanguage\signLanguageTranslator\logs\new_log.json'
    job_todo = []
    for _ in range(repeat):
        start_page +=1
        subject = 'Daily'
        url = 'http://api.kcisa.kr/openapi/service/rest/meta13/getCTE01701'
        params = {
            'serviceKey': 'ecc7282e-731e-4aa0-91b1-017535926c8f',
            'numOfRows': 10,
            'pageNo': start_page -1,
        }
        response = requests.get(url, params=params)

        
        
        if response.status_code == 200:
            print('요청 성공')
            # write_txt_log(TXT_LOG_PATH, f'Page {page} api 요청 성공') #임시
            page_start_time = datetime.now()
            content_type = response.headers.get('Content-Type')
            if content_type and 'charset' in content_type:
                encoding = content_type.split('charset=')[-1].strip()
            else:
                encoding = 'utf-8'
            xml_dict = xmltodict.parse(response.text, encoding=encoding)
            json_response = json.dumps(xml_dict, ensure_ascii=False, indent=2)
            json_response = json.loads(json_response)
            # print('응답 내용:', json_response) # 확인용
            # print(type( json_response['response']['body']['items']['item']), json_response['response']['body']['items']['item']) # 확인용
            
            item_li = json_response['response']['body']['items']['item']
            # data_to_log = j_data
            # data_to_log[subject][page] = dict()
            for i, item in enumerate(item_li):
                item_start_time = datetime.now()
                # print(item['title'], f'No.{i +1} in Page {page}')
                title, url_path = item['title'],item['subDescription'] 
                job_todo.append([title,url_path])
                item_end_time = datetime.now()
            page_end_time = datetime.now()
            # write_txt_log(TXT_LOG_PATH, f'Page {todo_page} 요청 완료 (페이지 완료까지 걸린시간 {page_end_time - page_start_time})')
    
        else:
            print(f'요청이 실패했습니다. 응답 코드: {response.status_code}')
            # write_txt_log(TXT_LOG_PATH, f'Page {todo_page} api 요청 실패\n 응답 코드: {response.status_code}') $ 임시
            break
        
        
    retries = 0
    while retries < 5:
        try:
            update_json_log(JSON_LOG_PATH,f'{start_page- repeat}-{start_page-1}',job_todo)
            print("Successfully updated the JSON log.")
            call_generator(job_todo)
            break  # 성공 시 루프 탈출
        except Exception as e:
            retries += 1
            print(f"Error occurred: {e}. Retry {retries}/{5} in {5} seconds...")
            time.sleep(5)

    if retries == 5:
        print("Max retries reached. Could not update the JSON log.")
    
def call_generator(todo_list):
    for job in tqdm(todo_list):
        title, url_path = job
        # make_data(title, url_path)


import sys

if __name__ == "__main__":
    start_page = input("시작페이지를 입력하세요: ")
    get_source(int(start_page))


