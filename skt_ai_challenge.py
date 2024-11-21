import streamlit as st
import os
import queue
import time
from PyPDF2 import PdfReader
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, CSVLoader
from dotenv import load_dotenv
from send_sms import send_many
from google_sheet import GoogleSheetsMonitor


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
        
        #PDF 파일 처리
        if filename.endswith('.pdf'):
            pdf_reader = PdfReader(filepath)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            splits = text_splitter.split_text(text)
            docs.extend(splits)
        
        # TXT 파일 처리
        elif filename.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                splits = text_splitter.split_text(text)
                docs.extend(splits)
        
        # CSV 파일 처리
        elif filename.endswith('.csv'):
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
    
    vectorstore.add_documents(docs)
    
    return vectorstore

# 사이드바 생성
with st.sidebar:
    # 문서 업로드 섹션
    st.header("FAQ 문서 업로드")
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
            
            st.success("문서 저장 완료 ✅")
    
    # 벡터 DB 새로고침 버튼
    if st.button("DB 새로고침", icon="🔄"):
        try:
            # 임베딩 및 벡터 스토어 다시 로드
            embeddings = initialize_embeddings()
            vectorstore = load_documents(embeddings)
            st.success("DB 새로고침 완료 ✅")
        except Exception as e:
            st.error(f"벡터 DB 업데이트 중 오류 발생 ⚠️")
    
    # 대화 초기화 버튼
    clear_button = st.button("대화 초기화", icon="🗑️")
    if clear_button:
        st.write("초기화 완료 ✅")
        st.session_state["messages"] = []


# 메시지 추가 및 표시 함수
def append_message(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))
    
def display_messages():
    for message in st.session_state["messages"]:
        st.chat_message(message.role).write(message.content)

# RAG 체인 생성 함수
def create_rag_chain():
    # 임베딩 및 벡터 스토어 초기화
    embeddings = initialize_embeddings()
    vectorstore = load_documents(embeddings)
    
    # 프롬프트 템플릿
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 제공된 문서 정보를 기반으로 정확하고 도움이 되는 답변을 제공하는 AI 어시스턴트입니다. 
        주어진 문맥과 관련된 정보만을 사용하여 45자 이내로 답변하세요. 만약 문서에서 충분한 정보를 찾을 수 없다면, 모른다고 답변해주세요."""),
        ("human", "질문: {question}\n\n문맥: {context}")
    ])
    
    # LLM 설정
    llm = ChatOpenAI(        
        base_url="https://api.platform.a15t.com/v1",
        model_name="openai/gpt-4o-mini-2024-07-18",
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

# Streamlit 앱 설정 (기존 코드와 동일)
st.title('T Brand Chat Bot :speech_balloon:')

# 이전 코드의 나머지 부분 (initialize_embeddings, load_documents 등)은 그대로 유지

# 메인 실행 로직 수정

def main():
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    
    # Google Sheets 모니터링 스레드 시작
    sheet_monitor = GoogleSheetsMonitor(
        sheet_key='1PzERiLTGOCArG2tx8HNaid9z370yq8-hUBf2fELVRzI',  # 실제 Google Sheet 키로 대체
        input_queue=input_queue,
        check_interval=5  # 5초마다 체크
    )
    
    # 큐에서 입력 처리
    def process_queue():
        try:
            while True:
                if not input_queue.empty():
                    user_input, phone_number, row_index = input_queue.get_nowait()
                    
                    print(f"Processing input from queue: {user_input}, from Phone Num: {phone_number}, row_index: {row_index}")
                    st.chat_message("user").write(user_input)
                    
                    try:
                        # RAG 체인 생성
                        rag_chain = create_rag_chain()
                        
                        # 스트리밍 응답 생성
                        response = rag_chain.stream(user_input)
                        
                        with st.chat_message("ai"):
                            container = st.empty()
                            ai_response = ""
                            for token in response:
                                ai_response += token
                                container.markdown(ai_response)
                        
                        # 메시지 추가
                        append_message("user", user_input)
                        append_message("ai", ai_response)
                        print(sheet_monitor.worksheet.update_cell(row_index, 5, ai_response))
                        send_many(phone_number, ai_response)
                        
                
                    except Exception as e:
                        st.error(f"오류 발생: {e}")
                        st.error("문서를 먼저 업로드해주세요.")
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