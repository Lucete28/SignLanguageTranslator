import streamlit as st
import json

import re

def is_english(word):
    return re.match(r'[a-zA-Z]+$', word) is not None

def find_url_by_kor(search_meaning, data_dict):
    for group_key, group_value in data_dict.items():
        for word_key, word_info in group_value.items():
            meaning = word_info[0]  # 단어의 한국어 뜻
            if search_meaning in meaning:  # 검색어가 한국어 뜻에 포함되어 있는지 확인
                return word_info[1]  # 해당하는 URL 반환
    return None

def find_url_by_en(word, data_dict):
    for group_key, group_value in data_dict.items():
        for word_key, word_value in group_value.items():
            if word_key.lower() == word.lower():  # 대소문자 구분 없이 비교
                return word_value[1]  # URL 반환
    return None

with open('api_log.json','r', encoding='utf-8') as f:
    j_data = json.load(f)
st.title('Sign Language')
st.text('Created to find videos managed in Project')
selected_url= None
GROUP = st.selectbox('Group',list(j_data.keys()))
c1, c2 = st.columns(2)
search_word = c1.text_input('Search Word')
if c2.button('Search'):
    if is_english(search_word):
        selected_url=find_url_by_en(search_word,j_data[GROUP])
    else:
        selected_url=find_url_by_kor(search_word,j_data[GROUP])

page = st.selectbox('Select a page:', list(j_data[GROUP].keys()))

c11, c12, c13 = st.columns(3)
# 테이블의 각 행을 위한 반복
for word, (meaning, url) in j_data[GROUP][page].items():
    
    c11.text(f"{word}")
    c12.text(f"{meaning}")
    if c13.button(f"{word} Video"):
        selected_url = url

if selected_url:
    st.video(selected_url)

