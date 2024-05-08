import requests
import xmltodict
import json
# from data_generator import trans_to_english
from datetime import datetime
import time
from tqdm import tqdm
from data_generator import *
from filenoti import filenoti as fn #라인 알림
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
            # print('요청 성공')
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
                # item_start_time = datetime.now()
                # print(item['title'], f'No.{i +1} in Page {page}')
                title, url_path = item['title'],item['subDescription'] 
                job_todo.append([title,url_path])
            #     item_end_time = datetime.now()
            # page_end_time = datetime.now()
            # write_txt_log(TXT_LOG_PATH, f'Page {todo_page} 요청 완료 (페이지 완료까지 걸린시간 {page_end_time - page_start_time})')
    
        else:
            print(f'요청이 실패했습니다. 응답 코드: {response.status_code}')
            # write_txt_log(TXT_LOG_PATH, f'Page {todo_page} api 요청 실패\n 응답 코드: {response.status_code}') $ 임시
            break
        
        
    retries = 0
    while retries < 5:
        try:
            print('###############')
            # print(start_page)
            update_json_log(JSON_LOG_PATH,f'{start_page- repeat}-{start_page-1}',job_todo)
            print("Successfully updated the JSON log.")
            call_generator(job_todo,start_page)
            # fn.noti_print(f'{start_page}번 페이지 프레임: {total_frame}')
            break  # 성공 시 루프 탈출
        except Exception as e:
            retries += 1
            print(f"Error occurred: {e}. Retry {retries}/{5} in {5} seconds...")
            fn.noti_print(f'KEY: {start_page- repeat}-{start_page-1}\n에서 jlog 업데이트 문제 발생')
            time.sleep(5)
    
    if retries == 5:
        print("Max retries reached. Could not update the JSON log.")
    
def call_generator(todo_list,start_page):
    i = 0
    for idx, job in enumerate(tqdm(todo_list)):
        # fn.noti_print(f'{start_page-10+i}번 페이지 작업 시작')
        i+=1
        title, url_path = job
        result = make_data(title, url_path)
        frame = result['data.shape'][0]
        increment = len(todo_list) // 10
        if (idx + 1) % increment == 0:
            fn.noti_print(f"{start_page-10} 페이지 {(idx + 1) / len(todo_list) * 100:.0f}% 완료")
        # fn.noti_print(f'{start_page-10} 페이지 {idx+1}번({title}) 작업완료: {frame}')

def page_todo():
    with open('C:/Users/oem/Desktop/jhy/signlanguage/SignLanguageTranslator/logs/new_log.json', 'r',encoding='utf-8') as jfile:
        data = json.load(jfile)
    data=data['Daily']
    max_value = 0
    max_key = None

    for key in data.keys():
        last_page = int(key.split('-')[-1])
        # 현재까지의 최대 값과 비교
        if last_page > max_value:
            max_value = last_page
            max_key = key
    return max_value+1 #max_key
import sys

if __name__ == "__main__":
    with open("C:/Users/oem/Desktop/jhy/signlanguage/Sign_Language_Remaster/key.json", 'r',encoding='utf-8') as json_file:
        data = json.load(json_file)
    fn.api_key = data['Line_api']
    # start_page = input("시작페이지를 입력하세요: ")
    start_page= page_todo()
    # start_page=int(start_page)
    print(f"{start_page}-{start_page+9} 페이지 작업 시작")
    with fn.main():
        get_source(start_page)
    fn.noti_print(f"{start_page}-{start_page+9} 페이지 작업 완료")


