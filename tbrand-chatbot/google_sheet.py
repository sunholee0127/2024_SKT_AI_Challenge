import threading
import gspread
import time


# class Singleton(type):
#     _instances = {}
#     def __call__(cls, *args, **kwargs):
#         if cls not in cls._instances:
#             cls._instances[cls] = super().__call__(*args, **kwargs)
#         return cls._instances[cls]

class GoogleSheetsMonitor(threading.Thread):
    def __init__(self, sheet_key, input_queue, worksheet_index=0, check_interval=5):
        
        # if not hasattr(self, 'initialized'):
        threading.Thread.__init__(self)
        self.sheet_key = sheet_key
        self.worksheet_index = worksheet_index
        self.check_interval = check_interval
        self.daemon = True
        self.input_queue = input_queue
        
        self.gc = gspread.service_account(filename='skt-ai-challenge-51b817bfbe25.json')
        self.sheet = self.gc.open_by_key(sheet_key)
        self.worksheet = self.sheet.get_worksheet(worksheet_index)
        self.initial_row_count = len(self.worksheet.get_all_values())
        self.stop_flag = False
        self.processed_rows = set()  # 처리된 행을 추적하기 위한 set
            
            # self.initialized = True

    def run(self):

        print("Google Sheets 모니터링 시작...")
        while not self.stop_flag:
            try:
                print(f"Checking for new input... Thread ID: {threading.get_ident()}")
                current_rows = self.worksheet.get_all_values()
                
                # 새로운 행 감지
                if len(current_rows) > self.initial_row_count:
                    new_rows = current_rows[self.initial_row_count:]
                    
                    # 각 새로운 행에 대해 처리
                    for row in new_rows:
                        # 첫 번째 열의 값을 user_input으로 사용 (필요에 따라 조정)
                        row_index = self.initial_row_count + new_rows.index(row) +1
                        
                        if row_index not in self.processed_rows and row and row[3] :
                            phone_number = row[1]
                            
                            if phone_number[0] == '1':
                                phone_number = '0' + phone_number
                            
                            print(f"New input detected: {row[3]}, from Phone Num: {phone_number}, row_index: {row_index}")
                            # 큐에 입력 추가
                            self.input_queue.put((row[3], phone_number, row_index))
                            # 처리된 행으로 표시
                            self.processed_rows.add(row_index)
                    
                    # 행 카운트 업데이트
                    self.initial_row_count = len(current_rows)
                
                # 지정된 간격만큼 대기
                print(f"Waiting for {self.check_interval} seconds...")  
                time.sleep(self.check_interval)
            
            except Exception as e:
                print(f"Google Sheets 모니터링 중 오류: {e}")
                time.sleep(self.check_interval)

    def stop(self):
        self.stop_flag = True