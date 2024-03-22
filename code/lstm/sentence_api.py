# gpt api 사용
import openai
import json
with open('G:/내 드라이브/LAB/Sign_Language_Remaster/key.json',encoding='utf-8') as json_file:
    KEY = json.load(json_file)
openai.api_key  = KEY['open_ai']

def make_sentence(list):
    messages = []
    content = f'단어 리스트를 줄깨 리스트의 단어들로 단어로 문장을 만들고싶어 순서를 최대한 바꾸지 말고 조사등을 이용해서 자연스럽게 해줘 [{list}]'

    messages.append({"role":"user", "content":content})

    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=messages
    )

    chat_response = completion.choices[0].message.content
#   print(f'ChatGPT: {chat_response}')
    messages.append({"role":"assistant", "content": chat_response})
    return chat_response