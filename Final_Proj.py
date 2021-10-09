# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go
import pandas as pd
from statsmodels.tsa.arima_model import ARMA, ARIMA, ARMAResults, ARIMAResults
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import requests
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

montreal = pd.read_excel('data/Seasonally Adjusted.xlsx',sheet_name='MONTREAL_CMA', index_col= 'Date')
montreal.index.freq = 'MS'
montreal = montreal[['Single_Family_Benchmark_SA','One_Storey_Benchmark_SA', 'Two_Storey_Benchmark_SA',
       'Townhouse_Benchmark_SA', 'Apartment_Benchmark_SA']]
montreal.name = 'Montreal'

toronto = pd.read_excel('data/Seasonally Adjusted.xlsx',sheet_name='GREATER_TORONTO', index_col= 'Date')
toronto.index.freq = 'MS'
toronto = toronto[['Single_Family_Benchmark_SA','One_Storey_Benchmark_SA', 'Two_Storey_Benchmark_SA',
       'Townhouse_Benchmark_SA', 'Apartment_Benchmark_SA']]
toronto.name = 'Toronto'

quebec_city = pd.read_excel('data/Seasonally Adjusted.xlsx',sheet_name='QUEBEC_CMA', index_col= 'Date')
quebec_city.index.freq = 'MS'
quebec_city = quebec_city[['Single_Family_Benchmark_SA','One_Storey_Benchmark_SA', 'Two_Storey_Benchmark_SA',
       'Townhouse_Benchmark_SA', 'Apartment_Benchmark_SA']]
quebec_city.name = 'Quebec City'

vancouver = pd.read_excel('data/Seasonally Adjusted.xlsx',sheet_name='GREATER_VANCOUVER', index_col= 'Date')
vancouver.index.freq = 'MS'
vancouver = vancouver[['Single_Family_Benchmark_SA','One_Storey_Benchmark_SA', 'Two_Storey_Benchmark_SA',
       'Townhouse_Benchmark_SA', 'Apartment_Benchmark_SA']]
vancouver.name = 'Vancouver'

lst = ['Apartment_Benchmark_SA', 'Single_Family_Benchmark_SA', 'One_Storey_Benchmark_SA','Two_Storey_Benchmark_SA','Townhouse_Benchmark_SA']

url = 'https://www.superbrokers.ca/tools/mortgage-rate-history'
url1 = 'https://www.rateinflation.com/inflation-rate/canada-historical-inflation-rate/'
cpi_file = pd.read_csv('data/1810025601_databaseLoadingData.csv')

def p2f(x):
    return float(x.strip('%'))/100

def getDataframe(url):
    html = requests.get(url).content
    df_list = pd.read_html(html,index_col = "Year")
    df = df_list[0]
    return df

def interestRate(url):
    df = getDataframe(url)
    df.index = pd.to_datetime(df.index,format='%Y')
    df['Variable Rate1'] = df['Variable Rate'].apply(p2f)
    df['5 Year Rate1'] = df['5 Year Rate'].apply(p2f)
    df = df.drop(['Variable Rate','5 Year Rate','Best Over Worst','Best Rate?'],axis=1)
    return df

def inflation(url):
    df = getDataframe(url)
    df.columns=["01","02","03","04","05","06","07","08","09","10","11","12","Annual"]
    df['Year'] = df.index
    df1 = pd.melt(df, id_vars = ["Year"],value_vars=['01', '02', '03', '04', '05', '06', '07', '08', '09', '10','11', '12'],value_name= 'Inflation Rate')
    df1["Date"] = df1['Year'].astype(str)+"-"+df1["variable"].astype(str)
    df1['date'] = pd.to_datetime(df1['Date'],format ='%Y-%m')
    df1 = df1.set_index('date')
    df1.drop(['Year','variable','Date'],axis=1,inplace=True)
    df1['Inflation Rate'] = df1['Inflation Rate'].str.replace(r'%', r'0').astype('float') / 100.0
    df1 = df1.sort_values(by='date')
    df1.dropna(inplace=True)
    return df1

def cpi(df):
    cpi1 = df.loc[df['UOM'] == 'Percent']
    cpi1['REF_DATE1']=pd.to_datetime(cpi1['REF_DATE'],format ='%Y-%m')
    cpi1 = cpi1[['REF_DATE1','Alternative measures','VALUE']]
    cpi1 = cpi1.set_index('REF_DATE1')
    cpi1['VALUE'] = cpi1['VALUE']/100
    cpi2 = cpi1.loc[cpi1['Alternative measures'] == 'Measure of core inflation based on a factor model, CPI-common (year-over-year percent change)']
    cpi3 = cpi1.loc[cpi1['Alternative measures'] == 'Measure of core inflation based on a weighted median approach, CPI-median (year-over-year percent change)']
    cpi4 = cpi1.loc[cpi1['Alternative measures'] == 'Measure of core inflation based on a trimmed mean approach, CPI-trim (year-over-year percent change)']
    cpi2 = cpi2.drop('Alternative measures',axis=1)
    cpi3 = cpi3.drop('Alternative measures',axis=1)
    cpi4 = cpi4.drop('Alternative measures',axis=1)
    cpi2.columns =['Based on a factor model, CPI-common (year-over-year percent change)']
    cpi3.columns =['Based on a weighted median approach, CPI-median (year-over-year percent change)']
    cpi4.columns =['Based on a trimmed mean approach, CPI-trim (year-over-year percent change)']    
    frames=[cpi2,cpi3,cpi4]
    output = pd.concat(frames,axis=1)   
    return output

def changeInPrice(df):
    new_df=pd.DataFrame()
    for index,data in df.items():
        lastDayOfYear = df.groupby(df.index.year)[index].last()
        firstDayOfYear = df.groupby(df.index.year)[index].first()
        percent = (lastDayOfYear-firstDayOfYear)/firstDayOfYear
        new_df[index] = percent
    new_df['date1'] = new_df.index
    new_df = new_df[['date1','Single_Family_Benchmark_SA','One_Storey_Benchmark_SA','Two_Storey_Benchmark_SA','Townhouse_Benchmark_SA','Apartment_Benchmark_SA']]
    return new_df

def arima_forecasting1 (df):   
    output = []
    for name1 in lst:
        model_fcast = ARIMA(df[name1],order=(1,1,0))
        results_fcast = model_fcast.fit()
        fcast= results_fcast.predict(start =len(df), end=len(df)+11,typ='levels').rename(name1)#+' Forecast')
        output.append(fcast)
    output = pd.concat(output,axis=1)
    frames = [df,output]
    resulttest = pd.concat(frames)
    return resulttest

def holt_winter_fcast(df):   
    output = []
    for name in lst:
        final_model= ExponentialSmoothing(df[name],trend='mul',seasonal='mul',seasonal_periods=12).fit()
        forecast_predictions = final_model.forecast(12).rename(name)
        output.append(forecast_predictions)
    output = pd.concat(output,axis=1)
    frames = [df,output]
    resulttest = pd.concat(frames)
    return resulttest

def merging_table(arima,holtwinters):
    arima = arima[-12:]
    holtwinters = holtwinters[-12:]
    arima.columns = ['Single_Family_ARIMA', 'One_Storey_Benchmark_ARIMA',
       'Two_Storey_Benchmark_ARIMA', 'Townhouse_Benchmark_ARIMA','Apartment_Benchmark_ARIMA']
    holtwinters.columns = ['Single_Family_HW', 'One_Storey_Benchmark_HW',
       'Two_Storey_Benchmark_HW', 'Townhouse_Benchmark_HW', 'Apartment_Benchmark_HW']
    frames= [arima,holtwinters]
    result = pd.concat(frames,axis=1)
    result['date'] = pd.DatetimeIndex(result.index).date 
    result = result[['date', 'Single_Family_ARIMA', 'Single_Family_HW','One_Storey_Benchmark_ARIMA','One_Storey_Benchmark_HW',
       'Two_Storey_Benchmark_ARIMA','Two_Storey_Benchmark_HW', 'Townhouse_Benchmark_ARIMA','Townhouse_Benchmark_HW','Apartment_Benchmark_ARIMA','Apartment_Benchmark_HW']] 
    return result
framesMerge = [inflation(url1),cpi(cpi_file)]
mergeFrames = pd.concat(framesMerge,axis=1)

fig = px.line(interestRate(url),title='Interest Rate in Canada')
fig1 = px.line(mergeFrames,title='Inflation Rate and CPI in Canada')
fig2 = px.line(montreal)
fig3 = px.line(toronto)
fig4 = px.line(quebec_city)
fig5 = px.line(vancouver)
fig6 = px.line(arima_forecasting1 (montreal))
fig7 = px.line(arima_forecasting1 (toronto))
fig8 = px.line(arima_forecasting1 (quebec_city))
fig9 = px.line(arima_forecasting1 (vancouver))
fig10 = px.line(holt_winter_fcast (montreal))
fig11 = px.line(holt_winter_fcast (toronto))
fig12 = px.line(holt_winter_fcast (quebec_city))
fig13 = px.line(holt_winter_fcast (vancouver))

app.layout = html.Div(children=[
    html.H1(children='Housing Analysis',
    style={
        'textAlign': 'center'
    }),
    dcc.Graph(
        id='interest',
        figure=fig
    ),
    dcc.Graph(
        id='InflationCPI',
        figure=fig1
    ),
    html.H3(children='Housing Price from Jan 2005 to March 2021',
    style={
        'textAlign': 'center'
    }),

    dcc.Graph(
        id='montreal',
        figure=fig2
    ),

     dcc.Graph(
        id='toronto',
        figure=fig3,
    ),

    dcc.Graph(
        id='quebeccity',
        figure=fig4
    ),

    dcc.Graph(
        id='vancouver',
        figure=fig5
    ),

    html.H3(children='Price Change in Montreal From Jan 2005 to Mar 2021',
    style={
        'textAlign': 'center'
    }), dash_table.DataTable(
    data=changeInPrice(montreal).to_dict('records'),
    columns=[{'id': c, 'name': c} for c in changeInPrice(montreal).columns],
    style_cell_conditional=[
        {
            'if': {'column_id': c},
            'textAlign': 'left'
        } for c in lst
    ],
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(248, 248, 248)'
        }
    ],
    style_header={
        'backgroundColor': 'rgb(230, 230, 230)',
        'fontWeight': 'bold'
    }
),
    
    html.H3(children='Forecasting using ARIMA model',
    style={
        'textAlign': 'center'
    }),

    dcc.Graph(
        id='montreal_fcast_arima',
        figure=fig6
    ),

     dcc.Graph(
        id='toronto_fcast_arima',
        figure=fig7,
    ),

    dcc.Graph(
        id='quebeccity_arima',
        figure=fig8
    ),

    dcc.Graph(
        id='vancouver_arima',
        figure=fig9
    ),

    html.H3(children='Forecasting using Holt-Winters Model',
    style={
        'textAlign': 'center'
    }),

    dcc.Graph(
        id='montreal_fcast_holtWinters',
        figure=fig10
    ),

     dcc.Graph(
        id='toronto_fcast_holtWinters',
        figure=fig11,
    ),

    dcc.Graph(
        id='quebeccity_fcast_holtWinters',
        figure=fig12
    ),

    dcc.Graph(
        id='vancouver_fcast_holtWinters',
        figure=fig13
    ),

    html.H3(children='Montreal ARIMA vs Holt-Winters',
    style={
        'textAlign': 'center'
    }), dash_table.DataTable(
    data=merging_table(arima_forecasting1(montreal),holt_winter_fcast(montreal)).to_dict('records'),
    columns=[{'id': c, 'name': c} for c in merging_table(arima_forecasting1(montreal),holt_winter_fcast(montreal)).columns],
    style_cell_conditional=[
        {
            'if': {'column_id': c},
            'textAlign': 'left'
        } for c in ['Single_Family_ARIMA', 'Single_Family_HW',
       'One_Storey_Benchmark_ARIMA', 'One_Storey_Benchmark_HW',
       'Two_Storey_Benchmark_ARIMA', 'Two_Storey_Benchmark_HW',
       'Townhouse_Benchmark_ARIMA', 'Townhouse_Benchmark_HW',
       'Apartment_Benchmark_ARIMA', 'Apartment_Benchmark_HW']
    ],
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(248, 248, 248)'
        }
    ],
    style_header={
        'backgroundColor': 'rgb(230, 230, 230)',
        'fontWeight': 'bold'
    }
),

])

fig.update_layout(
    title={
        'text': "Interest Rate in Canada",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Date",
    yaxis_title="Interest rate",
    )
fig1.update_layout(
    title={
        'text': "CPI and Inflation Rate in Canada",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Date",
    yaxis_title="Percentage",
    )
fig2.update_layout(
    title={
        'text': "Montreal",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Date",
    yaxis_title="Price"
    )
fig3.update_layout(
    title={
        'text': "Toronto",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Date",
    yaxis_title="Price"
    )
fig4.update_layout(
    title={
        'text': "Quebec City",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Date",
    yaxis_title="Price"
    )
fig5.update_layout(
    title={
        'text': "Vancouver",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Date",
    yaxis_title="Price"
    )
fig6.update_layout(
    title={
        'text': "Montreal (ARIMA Model)",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Date",
    yaxis_title="Price",
    shapes=[
    dict(
      type= 'line',
      yref= 'paper', y0= 0, y1= 1,
      xref= 'x', x0= '2021-03-01', x1= '2021-03-01'
      )
])
fig7.update_layout(
    title={
        'text': "Toronto (ARIMA Model)",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Date",
    yaxis_title="Price",
    shapes=[
    dict(
      type= 'line',
      yref= 'paper', y0= 0, y1= 1,
      xref= 'x', x0= '2021-03-01', x1= '2021-03-01'
    )
])
fig8.update_layout(
    title={
        'text': "Quebec City (ARIMA Model)",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Date",
    yaxis_title="Price",
    shapes=[
    dict(
      type= 'line',
      yref= 'paper', y0= 0, y1= 1,
      xref= 'x', x0= '2021-03-01', x1= '2021-03-01'
    )
])
fig9.update_layout(
    title={
        'text': "Vancouver (ARIMA Model)",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Date",
    yaxis_title="Price",
    shapes=[
    dict(
      type= 'line',
      yref= 'paper', y0= 0, y1= 1,
      xref= 'x', x0= '2021-03-01', x1= '2021-03-01'
    )
])
fig10.update_layout(
    title={
        'text': "Montreal (Holt-Winters Model)",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Date",
    yaxis_title="Price",
    shapes=[
    dict(
      type= 'line',
      yref= 'paper', y0= 0, y1= 1,
      xref= 'x', x0= '2021-03-01', x1= '2021-03-01'
    )
])
fig11.update_layout(
    title={
        'text': "Toronto (Holt-Winters Model)",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Date",
    yaxis_title="Price",
    shapes=[
    dict(
      type= 'line',
      yref= 'paper', y0= 0, y1= 1,
      xref= 'x', x0= '2021-03-01', x1= '2021-03-01'
    )
])
fig12.update_layout(
    title={
        'text': "Quebec City (Holt-Winters Model)",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Date",
    yaxis_title="Price",
    shapes=[
    dict(
      type= 'line',
      yref= 'paper', y0= 0, y1= 1,
      xref= 'x', x0= '2021-03-01', x1= '2021-03-01'
    )
])
fig13.update_layout(
    title={
        'text': "Vancouver (Holt-Winters Model)",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Date",
    yaxis_title="Price",
    shapes=[
    dict(
      type= 'line',
      yref= 'paper', y0= 0, y1= 1,
      xref= 'x', x0= '2021-03-01', x1= '2021-03-01'
    )
])

fig.show()

if __name__ == '__main__':
    app.run_server(debug=True)