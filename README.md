# T Brand ChatBot

제안서 : [2024 SKT AI Challenge Idea 제안서_아삭바삭_김이삭.pdf](https://github.com/user-attachments/files/17844101/2024.SKT.AI.Challenge.Idea._._.pdf)

## Run 
* poetry로 dependancy lib 설치
* streamlit 
```
poetry update
poetry run streamlit run trand-chatbot/main.py
```

## api key 
* `.env`에 openai api key, sms api key 등을 입력해야함

## process
1. google sheet를 polling
2. 문자 메세지를 읽어서 langchain으로 RAG
3. 답변을 문자로 답변 

## SMS statistic dashboard
* AWS S3에 build된 static dashboard 웹 호스팅
* https://bit.ly/3CKvrdR
