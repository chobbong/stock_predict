import warnings
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from datetime import date
import json
import streamlit as st

warnings.filterwarnings('ignore')

# JSON 파일 읽기
with open("krx_code.json", "r") as f:
    stock_dict = json.load(f)

yf.pdr_override()

class PatternFinder():

  def __init__(self, period=5):
      self.period = period

  def set_stock (self, code: str):
      self.code = code
      self.data = pdr.get_data_yahoo(code)
      self.close = self.data['Close']
      self.data['Change'] = self.close.pct_change()  # 'Change'를 계산하고 데이터프레임에 추가
      self.change = self.data['Change']
      return self.data

  def search(self, start_date, end_date, threshold=0.98):
      base = self.close[start_date:end_date]
      self.base_norm = (base - base.min()) / (base.max()-base.min())
      self.base = base

      st.dataframe(base)

      window_size = len(base)
      moving_cnt = len(self.data) - window_size - self.period - 1
      cos_sims = self.__cosine_sims(moving_cnt, window_size)

      self.window_size = window_size
      cos_sims = cos_sims [cos_sims > threshold]
      return cos_sims

  def __cosine_sims (self, moving_cnt, window_size):
      def cosine_similarity(x, y):
          return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

      sim_list = []
      for i in range(moving_cnt):
          target = self.close[i:i+window_size]
          target_norm = (target - target.min()) / (target.max() - target.min())
          cos_similarity = cosine_similarity(self.base_norm, target_norm)
          sim_list.append(cos_similarity)

      return pd.Series(sim_list).sort_values(ascending=False)

  def plot_pattern(self, idx, period=5):
    if period != self.period:
        self.period = period

    top = self.close[idx:idx+self.window_size+period]
    top_norm =  (top - top.min()) / (top.max() -top.min())

    fig, ax = plt.subplots()
    ax.plot(self.base_norm.values, label='base')
    ax.plot(top_norm.values, label='target')
    ax.axvline(x=len(self.base_norm)-1, c='r', linestyle='--')
    ax.axvspan(len(self.base_norm.values)-1, len(top_norm.values)-1, facecolor='yellow', alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    preds = self.change[idx+self.window_size: idx+self.window_size+period]
    st.dataframe(preds)
    st.write(f'pred: {preds.mean()*100} %')

  def stat_prediction(self, result, period=5):
    idx_list = list(result.keys())
    mean_list = []
    for idx in idx_list:
        pred = self.change[idx+self.window_size : idx+self.window_size+period]
        mean_list.append(pred.mean())
    return np.array(mean_list)
  

def AA(ticker_name):
    p = PatternFinder()
    p.set_stock(ticker_name)
    today = date.today().strftime('%Y-%m-%d')
    result = p.search('2023-05-01',today)
    
    if len(result) >= 1:
        first_row_index = result.index[0]  # 두 번째 행의 인덱스 값을 가져옵니다.
        p.plot_pattern(first_row_index)
    else:
        st.write('Not enough data for this ticker.')
        return
    
today_2 = date.today().strftime('%Y-%m-%d')
st.subheader("""
AI Stock Prediction App  
ver.0.0.1
""")
st.write("""
        하기 주가예측은 참고용으로만 사용하시기 바랍니다.   
        이 주가예측 시스템은 사용자의 투자의 이익이나 손실에 대해 어떠한 책임도 지지 않습니다.  
        이 시스템은 AI에 기반한 것으로, 정확도나 신뢰도를 보장하지 않습니다.     
        """)
st.write(f"이 예측은 {today_2}부터 5일후까지의 예측입니다")

with st.form(key='my_form'):
    ticker_num = st.text_input("종목명 또는 종목코드를 입력하세요: ")
   # 종목 코드 또는 이름에 따라 종목 찾기
    ticker_name = None
    if ticker_num in stock_dict.values():  # 입력 값이 종목 코드인 경우
        ticker_name = ticker_num + ".KS"
    elif ticker_num in stock_dict.keys():  # 입력 값이 종목 이름인 경우
        ticker_name = stock_dict[ticker_num] + ".KS"
    
    # When the user presses the button, the form will stop showing, and this function will return True.
    submit_button = st.form_submit_button(label='Submit')
    if submit_button:
        AA(ticker_name)
