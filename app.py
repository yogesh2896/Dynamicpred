import streamlit as st


def activate(url):
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  from pmdarima import auto_arima
  from statsmodels.tsa.statespace.sarimax import SARIMAX
  import warnings
  warnings.filterwarnings("ignore", category=DeprecationWarning)

# url=input("Enter the url of data")
  df=pd.read_csv(url)
  df=df[['date','Sales_Value']]
  df=df.set_index('date')

  df['date'] = pd.date_range(start='06/01/2011', end='12/01/2023', freq='M')

  df.set_index('date',inplace=True)




  autoari=auto_arima(df,seasonal=True,maxiter=300,suppress_warnongs=True)
  l=round(len(df)/3)
  train=df[:-l]
  test=df[-l:]

  p=autoari.order[0]+1
  d=autoari.order[1]+1
  q=autoari.order[2]+1
  model=SARIMAX(train,order=(p,d,q),seasonal_order=(p,d,q,12),trend=None)
  model=model.fit()

  trained_model=model.get_forecast(len(test)+24)

  predictions=trained_model.predicted_mean

  predictions=pd.DataFrame(predictions)

  predictions=predictions.reset_index()

  df=df.reset_index()
  predictions.columns=df.columns
  df=df.merge(predictions,how='outer',on='date')

  sns.lineplot(x=df['date'],y=df['Sales_Value_x'])
  sns.lineplot(x=df['date'],y=df['Sales_Value_y'])

  st.write('Forecasting Visual')
  df.set_index('date',inplace=True)
  df.columns=['Actual','Predicted']
  st.line_chart(df)

st.write('Welcome to the Predictive Model by Enoah')

url=st.text_input('Enter the URL')
if st.button('Predict'):
  activate(url)
