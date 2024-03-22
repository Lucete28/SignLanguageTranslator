import requests
import xmltodict
import json
from data_generator import trans_to_english
from datetime import datetime
from data_generator import *
# import datetime
#####################################################################################################

def page_todo(file_path, key):
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:

        data = {}

    if key not in data: # key 없을때 

        data[key] = dict()


        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
            print(f"{file_path}에 키 '{key}'가 없어서 추가하였습니다.")
            return 1
    else: # 키 있으면 해당키의 최근 페이지 반환
        if data[key]:
            return max(int(k) for k in data[key].keys()) +1 , data     # 있으면 다음페이지랑 data 반환
        else:
            return 1
        
####
print('test')
def write_txt_log(T_path, text):
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')

    with open(T_path, 'a', encoding='utf-8') as file:
        file.write(f"{formatted_time} ::: {text}\n")

def write_json_log(J_PATH, data):
    with open(J_PATH, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    
#####################################################################################################
        



def get_response(P=50): #    해야하는 페이지 받아서 return item_li 반환
    print(P,'번 반복예정')
    TXT_LOG_PATH = r'C:\Users\oem\Desktop\jhy\signlanguage\Sign_Language_Remaster\logs\LOG.TXT'
    JSON_LOG_PATH = r'C:\Users\oem\Desktop\jhy\signlanguage\Sign_Language_Remaster\logs\api_log.json'
    for _ in range(P):
        subject = 'Daily'
        todo_page, j_data = page_todo(JSON_LOG_PATH, subject)

        url = 'http://api.kcisa.kr/openapi/service/rest/meta13/getCTE01701'
        params = {
            'serviceKey': 'ecc7282e-731e-4aa0-91b1-017535926c8f',
            'numOfRows': 10,
            'pageNo': todo_page,
        }
        response = requests.get(url, params=params)

        if response.status_code == 200:
            print('요청 성공')
            write_txt_log(TXT_LOG_PATH, f'Page {todo_page} api 요청 성공')
            page_start_time = datetime.now()
            content_type = response.headers.get('Content-Type')
            if content_type and 'charset' in content_type:
                encoding = content_type.split('charset=')[-1].strip()
            else:
                encoding = 'utf-8'

            xml_dict = xmltodict.parse(response.text, encoding=encoding)
            json_response = json.dumps(xml_dict, ensure_ascii=False, indent=2)
            json_response = json.loads(json_response)
            # print('응답 내용:', json_response)
            # print(type( json_response['response']['body']['items']['item']), json_response['response']['body']['items']['item'])
            item_li = json_response['response']['body']['items']['item']
            data_to_log = j_data
            data_to_log[subject][todo_page] = dict()
            for i, item in enumerate(item_li):
                item_start_time = datetime.now()
                print(item['title'], f'No.{i +1} in Page {todo_page}')
                make_data(item['title'],item['subDescription'] )
                item_end_time = datetime.now()
                write_txt_log(TXT_LOG_PATH, f'\t{i+1}. {item["title"]} 작성완료 (걸린시간 {item_end_time - item_start_time})')
                data_to_log[subject][todo_page][trans_to_english(item['title'])] = [item['title'],item['subDescription']]
                write_json_log(JSON_LOG_PATH, data_to_log)
            page_end_time = datetime.now()
            write_txt_log(TXT_LOG_PATH, f'Page {todo_page} 작성 완료 (페이지 완료까지 걸린시간 {page_end_time - page_start_time})')
    
        else:
            print(f'요청이 실패했습니다. 응답 코드: {response.status_code}')
            write_txt_log(TXT_LOG_PATH, f'Page {todo_page} api 요청 실패\n 응답 코드: {response.status_code}')
            break





import sys

if __name__ == "__main__":
    file_start_time = datetime.now()
    
    if len(sys.argv)==2:
        get_response(int(sys.argv[1]))
    else:
        get_response()
    file_end_time = datetime.now()
    elapsed_time = file_end_time - file_start_time
    print(f"Generator 시작 시간: {file_start_time}")
    print(f"Generator 끝난 시간: {file_end_time}")
    print(f"Generator 걸린 시간: {elapsed_time}") 

