import streamlit as st
import os
import queue
import time
# from PyPDF2 import PdfReader
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, CSVLoader
from dotenv import load_dotenv
from send_sms import send_many
from google_sheet import GoogleSheetsMonitor

ADMIN_PHONE = '01025299685'

load_dotenv()

# 임베딩 초기화
@st.cache_resource
def initialize_embeddings():
    return OpenAIEmbeddings(        
        base_url="https://api.platform.a15t.com/v1",
        model="openai/text-embedding-3-small"
    )

# 문서 로더 및 벡터 스토어 생성
@st.cache_resource
def load_documents(_embedding):
    # 문서 폴더 경로 설정 (실제 경로로 변경 필요)
    documents_folder = "./documents"
    
    # 텍스트 분할기
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # 문서들을 담을 리스트
    docs = []
    
    # 폴더 내 모든 PDF, TXT, CSV 파일 처리
    for filename in os.listdir(documents_folder):
        filepath = os.path.join(documents_folder, filename)
        print(f"----------------- Processing file: {filepath}")
        
        # #PDF 파일 처리
        # if filename.endswith('.pdf'):
        #     pdf_reader = PdfReader(filepath)
        #     text = ""
        #     for page in pdf_reader.pages:
        #         text += page.extract_text() + "\n"
        #     splits = text_splitter.split_text(text)
        #     docs.extend(splits)
        
        # TXT 파일 처리
        if filename.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                splits = text_splitter.split_text(text)
                docs.extend(splits)
        
        # CSV 파일 처리
        elif filename.endswith('.csv'):
            print(f"Processing CSV file: {filepath}")
            loader = CSVLoader(filepath)
            docs = loader.load()

            # # Split documents into chunks
            split_docs = text_splitter.split_documents(docs)         
            docs.extend(split_docs)
    
    # ChromaDB 벡터 스토어 생성
    DB_PATH = "data/chroma"
    vectorstore = Chroma(    
        embedding_function=_embedding, 
        collection_name="store_faqs", 
        persist_directory=DB_PATH
    )
    vectorstore.reset_collection()
    vectorstore.add_documents(docs)
    #vectorstore.persist()
    return vectorstore

# 사이드바 생성
with st.sidebar:
    # 문서 업로드 섹션
    st.header("FAQ 생성")
    #-- 
    select_event = st.sidebar.selectbox('자동 답변에 쓰일 FAQ를 만들 방법을 고르세요',
                                        ['FAQ 문서 업로드','핸드폰 음성/문자 기반 자동 생성'])

    if select_event == 'FAQ 문서 업로드':
        uploaded_files = st.file_uploader(
            "PDF, TXT, CSV 파일을 업로드하세요", 
            type=['pdf', 'txt', 'csv'], 
            accept_multiple_files=True
        )
        
        # 문서 저장 버튼
        if st.button("문서 저장", icon="📁"):
            if uploaded_files:
                # 문서 저장 경로
                os.makedirs("./documents", exist_ok=True)
                
                for uploaded_file in uploaded_files:
                    with open(os.path.join("./documents", uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                with st.spinner("문서 저장 중... 📁"):
                    time.sleep(3)
                # # 문서 업로드 완료 후 벡터 스토어 업데이트    
                # try:
                #     # 임베딩 및 벡터 스토어 다시 로드
                #     embeddings = initialize_embeddings()
                #     vectorstore = load_documents(embeddings)
                #     #st.success("DB 새로고침 완료 ✅")
                # except Exception as e:
                #     st.error(f"벡터 DB 업데이트 중 오류 발생 ⚠️")            
                st.success("문서 저장 완료 ✅")        

    else:    
        # 음성 및 문자 가져오기 버튼
        phone_number = st.text_input("핸드폰 번호를 입력하세요", value='01025299685')
        # checkbox
        st.checkbox("음성 녹음 데이터")
        st.checkbox("문자 메세지 데이터") 
        get_data_button = st.button("데이터 기반 FAQ 자동 생성 :robot_face:")
        if get_data_button:
            with st.spinner('Wait for it...'):
                time.sleep(3)
            st.write("데이터 가져오기 완료 ✅")
        
        
        # 대화 초기화 버튼
        # clear_button = st.button("대화 초기화", icon="🗑️")
        # if clear_button:
        #     st.write("초기화 완료 ✅")
        #     st.session_state["messages"] = []


# 메시지 추가 및 표시 함수
def append_message(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))
    
def display_messages():
    for message in st.session_state["messages"]:
        st.chat_message(message.role).write(message.content)

def create_sentiment_chain():
    # LLM 설정
    llm = ChatOpenAI(        
        base_url="https://api.platform.a15t.com/v1",
        model_name="openai/gpt-4o-2024-08-06",
        temperature=0
    )
    
    # 프롬프트 템플릿
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 텍스트의 감정을 분석하는 AI 어시스턴트입니다.
        주어진 텍스트의 감정을 분석하여 'Positive', 'Negative', 'Neutral' 중 하나로 답변해주세요."""),
        ("human", "다음 텍스트의 감정을 분석해주세요: {text}")
    ])
    
    # 감정 분석 결과 처리 함수
    def process_sentiment(response):
        sentiment = response.lower()
        if 'negative' in sentiment:
            return True
        else:
            return False
    
    # 감정 분석 체인 구성
    sentiment_chain = (
        {"text": lambda x: x}
        | prompt 
        | llm 
        | StrOutputParser()
        | process_sentiment
    )
    
    return sentiment_chain

# RAG 체인 생성 함수
def create_rag_chain():
    # 임베딩 및 벡터 스토어 초기화
    embeddings = initialize_embeddings()
    vectorstore = load_documents(embeddings)
    
    # 프롬프트 템플릿
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", """당신은 제공된 문서 정보를 기반으로 정확하고 도움이 되는 답변을 제공하는 AI 어시스턴트입니다. 
    #     주어진 문맥과 관련된 정보만을 사용하여 45자 이내로 답변하세요. 만약 문서에서 충분한 정보를 찾을 수 없다면, '담당자가 확인 후에 별도 안내 드리도록 하겠습니다'라고 답변해주세요."""),
    #     ("human", "질문: {question}\n\n문맥: {context}")
    # ])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 제공된 세탁소 FAQ 문서 정보를 기반으로 도움이 되는 답변을 제공하는 세탁소 AI 어시스턴트입니다. 
        주어진 문맥과 관련된 정보를 최대한 사용하여 30자 이내로 매우 친절하게 답변하세요."""),
        ("human", "질문: {question}\n\n문맥: {context}")
    ])
    
    # LLM 설정
    llm = ChatOpenAI(        
        base_url="https://api.platform.a15t.com/v1",
        model_name="openai/gpt-4o-2024-08-06",
        temperature=0
    )
    
    # 문서 검색 함수
    def get_retriever_context(question):
        # 유사도 검색으로 관련 문서 조각 찾기 (상위 3개)
        relevant_docs = vectorstore.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        return context
    
    # RAG 체인 구성
    rag_chain = (
        {"context": get_retriever_context, "question": lambda x: x}
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    return rag_chain

# 큐 생성
input_queue = queue.Queue()
extra_queue = queue.Queue()

st.image('./tbrandchatbot_banner.png', use_container_width=True)

# Streamlit 앱 설정 (기존 코드와 동일)
#st.title('T Brand Chatbot :speech_balloon:')

# 이전 코드의 나머지 부분 (initialize_embeddings, load_documents 등)은 그대로 유지

# vector store add
def add_documents(doc):
    embeddings = initialize_embeddings()
    DB_PATH = "data/chroma"
    vectorstore = Chroma(    
        embedding_function=embeddings, 
        collection_name="store_faqs", 
        persist_directory=DB_PATH
    )
    if isinstance(doc[0], str) and isinstance(doc[1], str):
        doc = [Document(page_content=doc[0] + "\n" + doc[1])]
        print(f'Add documents: {doc}')
    ids = vectorstore.add_documents(doc)
    #print(vectorstore.get())
    #vectorstore.persist()
    return ids[0] if ids else None  # 첫 번째 ID만 반환
    

def main():
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    
    # Google Sheets 모니터링 스레드 시작
    sheet_monitor = GoogleSheetsMonitor(
        sheet_key=os.environ['GOOGLE_SHEET_KEY'],
        #sheet_key='1PzERiLTGOCArG2tx8HNaid9z370yq8-hUBf2fELVRzI',  # 실제 Google Sheet 키로 대체
        input_queue=input_queue,
        check_interval=5  # 5초마다 체크
    )
    
    # 큐에서 입력 처리
    def process_queue():
        try:
            while True:
                handover = False
                if not input_queue.empty():
                    user_input, phone_number, row_index = input_queue.get_nowait()
                    print(f"Processing input from queue: {user_input}, from Phone Num: {phone_number}, row_index: {row_index}")
                                        
                    # RAG 체인 생성
                    rag_chain = create_rag_chain()
                    print("rag chain created")
                    try:
                        # 관리자가 답변 준 경우에
                        if phone_number == ADMIN_PHONE:
                            if "통계" in user_input or "대시보드" in user_input:
                                send_many(phone_number, "https://bit.ly/3CKvrdR")
                                continue
                            
                            answer = user_input
                            ex_question, ex_phone_number, ex_row_index = extra_queue.get_nowait()
                            print(f"Processing input from extra queue: {ex_question}, from Phone Num: {ex_phone_number}, row_index: {ex_row_index}")
                            
                            # 답변 전송
                            send_many(ex_phone_number, answer)
                            print(sheet_monitor.worksheet.update_cell(ex_row_index, 5, answer))
                            new_doc = [ex_question, answer]
                            id = add_documents(new_doc)
                            print("Add documents to vector store. ID: ", id)
                            continue
                        
                        print("rag chain invoked") 
                        
                        st.chat_message("user").write(user_input + " (고객 : " + phone_number + ")")
                        sentiment_chain = create_sentiment_chain()    
                        # 감정 분석 응답 생성
                        is_negative = sentiment_chain.invoke(user_input)
                        print(f"[감정분석] Q : {user_input}, A : {'Negative' if is_negative else 'Positive'}")
                        if is_negative:
                            msg = "고객감정: 부정\n"+ user_input + "\n" + "고객 : " + phone_number 
                            send_many(ADMIN_PHONE, msg)
                            sheet_monitor.worksheet.update_cell(row_index, 6, "Negative")
                            ai_response = "고객님 불편을 드려 죄송합니다. 담당자가 바로 연락드리겠습니다"
                            send_many(phone_number,ai_response)
                            continue
                            
                        
                        if '사장님 일요일 영업 하나요' in user_input:
                            handover = True
                            # handover to admin
                            send_many(ADMIN_PHONE, user_input + "\n" + "(고객 : " + phone_number + ")")
                            extra_queue.put((user_input, phone_number, row_index))
                            ai_response = "담당자 확인 후에 별도 안내 드리겠습니다."
                            # send message to customer
                            send_many(phone_number, ai_response)
                            
                        # 스트리밍 응답 생성
                        if not handover:
                            response = rag_chain.stream(user_input)
                            
                            with st.chat_message("ai"):
                                container = st.empty()
                                ai_response = ""
                                for token in response:
                                    ai_response += token
                                    container.markdown(ai_response)
                        else:
                            with st.chat_message("ai"):
                                container = st.empty()    
                                container.markdown(ai_response)
                        
                        # 메시지 추가
                        append_message("user", user_input)
                        print(sheet_monitor.worksheet.update_cell(row_index, 5, ai_response))
                        append_message("ai", ai_response)
                        send_many(phone_number, ai_response)
                    
                    except Exception as e:
                        st.error(f"오류 발생: {e}")
                        #st.error("문서를 먼저 업로드해주세요.")
                else:
                    time.sleep(1)
        except Exception as e:
            st.error(f"큐 처리 중 오류: {e}")


    sheet_monitor.start()

    # 이전 대화 기록 표시
    display_messages()    
    st.chat_input('궁금한 내용을 물어보세요:')

    # 큐에 있는 입력 처리
    process_queue()   
    
    
# Streamlit 앱 실행
if __name__ == "__main__":
    main()