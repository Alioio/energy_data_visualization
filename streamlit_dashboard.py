import numpy as np
import pandas as pd
#import polars as pl
import os
import re
import datetime
import timeit
from datetime import datetime as dt
from datetime import date, timedelta
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
from plotly.subplots import make_subplots
#import altair_catplot as altcat
alt.renderers.set_embed_options(tooltip={"theme": "dark"})
alt.data_transformers.disable_max_rows()
import streamlit as st
from streamlit_plotly_events import plotly_events
from st_aggrid import AgGrid, GridUpdateMode, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import threading
import concurrent.futures
import json


selectd_variable_dict = {
        "Arbeitspreis": "dataunit",
        "dataunit": "Arbeitspreis",
        "Grundpreis": "datafixed",
        "kWh Preis":"kwh_price",
        "Jahreskosten":"Jahreskosten"}

def my_theme():
  return {
    'config': {
      'view': {"stroke": "transparent",'continuousHeight': 300, 'continuousWidth': 400},  # from the default theme
      'range': {'category': ['#4650DF','#FC6E44', '#006E78', '#20B679', '#929898','#EBB5C5', '#54183E', '#CDE9FF', '#FAB347', '#E3D1FF']},
      "axisY": {
                "size":'1px',
                "color":'lightgray',
                "domain": False,
                "tickSize": 0,
                "gridDash": [2, 8]
            },

        "axisX": {
                "size":'1px',
                "color":'lightgray',
                "domain": False,
                "tickSize": 0,
                "gridDash": [2, 8]
            },
        
    }
  }
alt.themes.register('my_theme', my_theme)
alt.themes.enable('my_theme')

st.set_page_config(page_title="Energy Dashboard",
                    page_icon=":bar_chart:",
                    layout="wide")

st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)

def unit_to_month(currentunit, value_with_current_unit):
      #  ['month', 'once', 'year', 'week', 'nan', 'day', 'indefinitely'],
    if(currentunit == 'month'):
        return np.abs(value_with_current_unit)
    elif(currentunit == 'year'):
        return np.abs(value_with_current_unit *12)
    if(currentunit == 'week'):
        return np.abs(value_with_current_unit *0.25)
    if(currentunit == 'day'):
        return np.abs(value_with_current_unit / 30)
    if( (( currentunit == 'nan' or currentunit == 'once' or currentunit == 'indefinitely') & value_with_current_unit == 0)):
        return 0
    else:
        return int(-1)

def set_plz(ID):
  if((ID==3) | (ID==1)):
    return '10245'
  elif((ID==7) | (ID==5)):
    return '99425'
  elif((ID==11)| (ID==9)):
    return '33100'
  elif((ID==15) |(ID==13)):
    return '50670'
  elif((ID==19) |(ID==17)):
    return '71771'

@st.cache(ttl=7*24*60*60)
def read_energy_data_100(energy_type, verbrauch):
    ## Lese alle Dateien und füge sie zu einer Liste zusammen

    gas_path = 'data/{energy_type}'.format(energy_type=energy_type)
    files = os.listdir(gas_path)
    #print(files)

    dfs = list([])
    names = list([])

    for file in files:
        if(re.search("[15|9|13|3]+[0]{3}.csv$", str(file))):
            f = open(os.path.join(gas_path, file),'r')
            df = pd.read_csv(f)
            names.append(str(file))
            dfs.append(df)
            f.close()
        else:
            continue

    ## Fasse alle Abfragen in einer DF zusammen
    files = pd.DataFrame()
    all_dates = pd.DataFrame()
    for name, df in zip(names, dfs):
        date = name.split('_')[0].split()[:-1]
        date = ' '.join(date)
        date = datetime.datetime.strptime(date, '%b %d %y')  
        consumption = name.split('_')[-1:][0].split('.')[0]
        files = files.append(pd.DataFrame({'date':[date], 'consumption':[consumption]}))

        df['date'] = date
        df['consumption'] = consumption

        all_dates = all_dates.append(df)

    ## Filter nach Verbrauch und wandle dataunit in float um
    all_dates = all_dates.drop_duplicates(['date', 'providerName', 'tariffName', 'signupPartner', 'plz', 'dataeco', 'datatotal'])

    all_dates = all_dates[['date', 'providerName', 'tariffName', 'signupPartner', 'plz', 'dataunit',  'datafixed','datatotal','contractDurationNormalized', 'priceGuaranteeNormalized', 'dataeco']]
    
    all_dates['dataunit'] = all_dates['dataunit'].str.replace(',','.').astype(float)
    all_dates['datatotal'] = all_dates['datatotal'].str.replace(',','.').astype(float)
    all_dates['datafixed'] = all_dates['datafixed'].str.replace(',','.').astype(float)


    print('MIT DEM EINLESEN DER 100 PLZ DATEN FERTIG')

    
    vx_df = all_dates[all_dates['signupPartner'] == 'vx']
    c24_df = all_dates[all_dates['signupPartner'] == 'c24']
    joined_df = pd.merge(left=vx_df,
                        right=c24_df,
                        how='inner',
                        on=['date', 'plz','providerName', 'tariffName'],
                        suffixes=('_vx', '_c24'))

    vx_columns = [x for x in joined_df.columns if re.search('_vx', x)]
    c4_columns = [x for x in joined_df.columns if re.search('_c24', x)]
    for vx_column, c24_column in zip(vx_columns, c4_columns):
        if(  (vx_column.split('_vx')[0] == c24_column.split('_c24')[0]) & (joined_df[vx_column].dtype == int) | (joined_df[vx_column].dtype == float)   ):
            column_name = 'delta_'+vx_column.split('_vx')[0]
            joined_df[column_name] = np.abs(joined_df[c24_column] - joined_df[vx_column] )
    
    return joined_df, all_dates
    
@st.cache(ttl=7*24*60*60)
def read_energy_data(energy_type, verbrauch):
    ## Lese alle Dateien und füge sie zu einer Liste zusammen

    gas_path = 'data/{energy_type}'.format(energy_type=energy_type)
    files = os.listdir(gas_path)
    print('FILES:!!!! ',files)

    dfs = list([])
    names = list([])

    for file in files:
        if(re.search("[15|9|13|3]+[0]{3}.csv$", str(file))):
            f = open(os.path.join(gas_path, file),'r')
            df = pd.read_csv(f)
            names.append(str(file))
            dfs.append(df)
            f.close()
        else:
            continue

    ## Fasse alle Abfragen in einer DF zusammen
    files = pd.DataFrame()
    all_dates = pd.DataFrame()
    for name, df in zip(names, dfs):
        date = name.split('_')[0].split()[:-1]
        date = ' '.join(date)
        date = datetime.datetime.strptime(date, '%b %d %y')  
        consumption = name.split('_')[-1:][0].split('.')[0]
        files = files.append(pd.DataFrame({'date':[date], 'consumption':[consumption]}))

        df['date'] = date
        df['consumption'] = consumption

        all_dates = all_dates.append(df)

    ## Filter nach Verbrauch und wandle dataunit in float um
    all_dates = all_dates.drop_duplicates(['date', 'providerName', 'tariffName', 'signupPartner', 'plz', 'dataeco', 'datatotal'])
    print('COLUMNS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: ',all_dates.columns)
    if('commoncarrier' in all_dates.columns):
        all_dates = all_dates[['date', 'providerName', 'tariffName', 'signupPartner', 'plz', 'dataunit','datatotal',  'datafixed','contractDurationNormalized', 'priceGuaranteeNormalized', 'dataeco','commoncarrier','onlyExistingCustomer', 'recommended']]
    else:
        all_dates = all_dates[['date', 'providerName', 'tariffName', 'signupPartner', 'plz', 'dataunit','datatotal',  'datafixed','contractDurationNormalized', 'priceGuaranteeNormalized', 'dataeco','onlyExistingCustomer', 'recommended']]
    all_dates.rename(columns = {'datatotal':'Jahreskosten'}, inplace = True)
    
    all_dates['dataunit'] = all_dates['dataunit'].str.replace(',','.').astype(float)
    all_dates['Jahreskosten'] = all_dates['Jahreskosten'].str.replace(',','.').astype(float)
    all_dates['datafixed'] = all_dates['datafixed'].str.replace(',','.').astype(float)
    all_dates['dataeco'] = all_dates['dataeco'].replace({0:False, 1:True})
    all_dates['dataeco'] = all_dates['dataeco'].replace({'false':False, 'true':True})
    all_dates['dataeco'] = all_dates['dataeco'].replace({'False':False, 'True':True})
    all_dates['dataeco'] = all_dates['dataeco'].astype(bool)
    all_dates['kwh_price'] = ((all_dates['datafixed']*100)/3000) + all_dates['dataunit']

    
    print('MIT DEM EINLESEN DER 100 PLZ DATEN FERTIG')

    '''
    #### lese die Daten der wöchentlichen Abfrage zu den 5 Städten
    wa_df = pd.read_excel('data/wa_{energy_type}.xlsx'.format(energy_type=energy_type))

    if( ((energy_type == 'gas') & (verbrauch=="15000")) | ((energy_type == 'electricity') & (verbrauch=="3000")) ):
        wa_df = wa_df[(wa_df.ID == 3) | 
        (wa_df.ID == 7) |
        (wa_df.ID == 11) |
        (wa_df.ID == 15) |
        (wa_df.ID == 19) ]
    elif(((energy_type == 'gas') & (verbrauch=="9000")) | ((energy_type == 'electricity') & (verbrauch=="1300"))):
        wa_df = wa_df[(wa_df.ID == 1) | 
        (wa_df.ID == 5) |
        (wa_df.ID == 9) |
        (wa_df.ID == 13) |
        (wa_df.ID == 17) ]

    ###
    wa_df['Einheit Vertragslaufzeit'].fillna('nan')
    wa_df['Einheit Garantielaufzeit'].fillna('nan')

    wa_df['contractDurationNormalized'] = wa_df.apply(lambda row : unit_to_month(row['Einheit Vertragslaufzeit'], row['Vertragslaufzeit']), axis = 1)
    wa_df['priceGuaranteeNormalized'] = wa_df.apply(lambda row : unit_to_month(row['Einheit Garantielaufzeit'], row['Garantielaufzeit']), axis = 1)

    print('davor: ',wa_df.columns)

    if('Grunpreis' in wa_df.columns):
        wa_df.rename(columns = {'Datum':'date','Anbieter':'providerName','Tarifname':'tariffName', 'Partner':'signupPartner', 'Arbeitspreis':'dataunit', 'Öko':'dataeco', 'Kosten pro Jahr':'Jahreskosten', 'Grunpreis':'datafixed'}, inplace = True)
    else:
        wa_df.rename(columns = {'Datum':'date','Anbieter':'providerName','Tarifname':'tariffName', 'Partner':'signupPartner', 'Arbeitspreis':'dataunit', 'Öko':'dataeco', 'Kosten pro Jahr':'Jahreskosten', 'Grundpreis':'datafixed'}, inplace = True)

    print('danach: ',wa_df.columns)
    wa_df['plz'] = wa_df.apply(lambda row : set_plz(row['ID']), axis = 1)
    wa_df = wa_df[wa_df.date < all_dates.date.min()]
    wa_df = wa_df.drop_duplicates(['date', 'providerName', 'tariffName', 'signupPartner', 'plz'])
    
    all_dates = pd.concat([wa_df, all_dates])
    #all_dates = wa_df.copy()
    ###
    '''
    data_types_dict = {'date':'<M8[ns]', 'providerName':str, 'tariffName':str,'signupPartner':str, 'dataunit':float, 'plz':str, 'datafixed':float, 'Jahreskosten':float}
    all_dates = all_dates.astype(data_types_dict)

    print('MIT DEM EINLESEN DER 5 PLZ DATEN FERTIG ',energy_type)
    if('commoncarrier' in all_dates.columns):
        return all_dates[['date', 'providerName', 'tariffName', 'signupPartner', 'plz', 'dataunit',  'datafixed','kwh_price','Jahreskosten','contractDurationNormalized', 'priceGuaranteeNormalized', 'dataeco','commoncarrier','onlyExistingCustomer', 'recommended']]
    else:
        return all_dates[['date', 'providerName', 'tariffName', 'signupPartner', 'plz', 'dataunit',  'datafixed','kwh_price','Jahreskosten','contractDurationNormalized', 'priceGuaranteeNormalized', 'dataeco','onlyExistingCustomer', 'recommended']]

with concurrent.futures.ThreadPoolExecutor() as executor:
    electricity_reader_thread_3000 = executor.submit(read_energy_data, 'electricity', '3000')
    #electricity_reader_thread_1300 = executor.submit(read_energy_data, 'electricity', '1300')
    gas_reader_thread_15000 = executor.submit(read_energy_data, 'gas', '15000')
    #gas_reader_thread_9000 = executor.submit(read_energy_data, 'gas', '9000')

    #electricity_reader_thread_100_3000 = executor.submit(read_energy_data_100, 'electricity', '3000')
    #electricity_reader_thread_100_1300 = executor.submit(read_energy_data_100, 'electricity', '1300')
    #gas_reader_thread_100_15000 = executor.submit(read_energy_data_100, 'gas', '15000')
    #gas_reader_thread_100_9000 = executor.submit(read_energy_data_100, 'gas', '9000')

    
    electricity_results_3000 = electricity_reader_thread_3000.result()
    #electricity_results_1300 = electricity_reader_thread_1300.result()
    gas_results_15000 = gas_reader_thread_15000.result()
    #gas_results_9000 = gas_reader_thread_9000.result()

    #electricity_results_100_3000 = electricity_reader_thread_100_3000.result()
    #electricity_results_100_1300 = electricity_reader_thread_100_1300.result()
    #gas_results_100_15000 = gas_reader_thread_100_15000.result()
    #gas_results_100_9000 = gas_reader_thread_100_9000.result()

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def summarize(results, seperation_var='priceGuaranteeNormalized',selection_short=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], selection_long=[12, 13, 14, 15, 16, 24], consumption='unknown',selected_variable='dataunit', top_n = '10'):
    
    sep_var_readable = seperation_var
    if(seperation_var == 'Vertragslaufzeit'):
        seperation_var = 'contractDurationNormalized'
    elif(seperation_var == 'Preisgarantie'):
        seperation_var='priceGuaranteeNormalized'
    elif(seperation_var == 'Öko Tarif/ Konventioneller Tarif'):
        seperation_var = 'dataeco'
    elif(seperation_var == 'Partner'):
        seperation_var = 'signupPartner'
    elif(seperation_var== 'Kein Unterscheidungsmerkmal'):
        seperation_var = 'None'
    elif(seperation_var == "Grundversorgungstarif"):
        seperation_var = 'commoncarrier'

    variables_dict = {
        "Arbeitspreis": "dataunit",
        "kWh Preis": "kwh_price",
        "Jahreskosten": "Jahreskosten"
        }

    global_summary = results.groupby(['date']).count()
    global_summary['date'] = global_summary.index
    global_summary['count_global'] = global_summary.providerName
    global_summary = global_summary[['date', 'count_global']].reset_index(drop=True).copy()

    agg_functions = {
        variables_dict[selected_variable]:
        [ 'mean', 'median','std', 'min', 'max', 'count']
    }
    
    if( (seperation_var == 'contractDurationNormalized') | (seperation_var == 'priceGuaranteeNormalized') ):

        ohne_laufzeit  = results[results[seperation_var].isin(selection_short)]
        ohne_laufzeit_all = ohne_laufzeit.copy()

        summary_ohne_laufzeit_all = ohne_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
        summary_ohne_laufzeit_all.columns =  [ 'mean_all', 'median_all','std_all', 'min_all', 'max_all', 'count_all']
  
        ohne_laufzeit.sort_values(['date', 'plz', variables_dict[selected_variable]], ascending=[True, True, True], inplace=True)
        ohne_laufzeit['rank'] = 1
        ohne_laufzeit['rank'] = ohne_laufzeit.groupby(['date', 'plz'])['rank'].cumsum()
        if(top_n != 'Alle'):
            ohne_laufzeit = ohne_laufzeit[ohne_laufzeit['rank'] <= int(top_n)]

        summary_ohne_laufzeit = ohne_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
        summary_ohne_laufzeit.columns =  [ 'mean', 'median','std', 'min', 'max', 'count']
        summary_ohne_laufzeit['date'] = summary_ohne_laufzeit.index
        summary_ohne_laufzeit['beschreibung'] = 'Kurze '+sep_var_readable
        
        #mit_laufzeit  = high_consume[high_consume['contractDurationNormalized'] > 11]
        mit_laufzeit  = results[results[seperation_var].isin(selection_long)]
        mit_laufzeit_all = mit_laufzeit.copy()
        summary_mit_laufzeit_all = mit_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
        summary_mit_laufzeit_all.columns =  [ 'mean_all', 'median_all','std_all', 'min_all', 'max_all', 'count_all']

        
        mit_laufzeit.sort_values(['date', 'plz',variables_dict[selected_variable]], ascending=[True, True, True], inplace=True)
        mit_laufzeit['rank'] = 1
        mit_laufzeit['rank'] = mit_laufzeit.groupby(['date', 'plz'])['rank'].cumsum()
        if(top_n != 'Alle'):
            mit_laufzeit = mit_laufzeit[mit_laufzeit['rank'] <= int(top_n)]

        summary_mit_laufzeit = mit_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
        summary_mit_laufzeit.columns =  [ 'mean', 'median','std', 'min', 'max', 'count']
        summary_mit_laufzeit['date'] = summary_mit_laufzeit.index
        summary_mit_laufzeit['beschreibung'] = 'Lange '+sep_var_readable

        summary = pd.concat([summary_mit_laufzeit, summary_ohne_laufzeit])
        summary_all = pd.concat([summary_mit_laufzeit_all, summary_ohne_laufzeit_all])
        print('all: ',len(summary_all))
        print('summary: ',len(summary))
        summary = pd.concat([summary, summary_all], axis=1)

        print('summary before merge: ',len(summary),'   ',summary.columns)

        summary = summary.reset_index(drop=True).copy()

        summary = pd.merge(left=summary,
                 right=global_summary,
                 how='left',
                 on='date')

        print('summary after merge: ',len(summary),'   ',summary.columns)
    elif(seperation_var == 'dataeco'):
        ohne_laufzeit  = results[results[seperation_var] == False]
        ohne_laufzeit_all = ohne_laufzeit.copy()

        summary_ohne_laufzeit_all = ohne_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
        summary_ohne_laufzeit_all.columns =  [ 'mean_all', 'median_all','std_all', 'min_all', 'max_all', 'count_all']


        
        ohne_laufzeit.sort_values(['date', 'plz', variables_dict[selected_variable]], ascending=[True, True, True], inplace=True)
        ohne_laufzeit['rank'] = 1
        ohne_laufzeit['rank'] = ohne_laufzeit.groupby(['date', 'plz'])['rank'].cumsum()
        if(top_n != 'Alle'):
            ohne_laufzeit = ohne_laufzeit[ohne_laufzeit['rank'] <= int(top_n)]

        summary_ohne_laufzeit = ohne_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
        summary_ohne_laufzeit.columns =  [ 'mean', 'median','std', 'min', 'max', 'count']
        summary_ohne_laufzeit['date'] = summary_ohne_laufzeit.index
        summary_ohne_laufzeit['beschreibung'] = 'Öko'
        
        #mit_laufzeit  = high_consume[high_consume['contractDurationNormalized'] > 11]
        mit_laufzeit  = results[results[seperation_var] == True]
        mit_laufzeit_all = mit_laufzeit.copy()
        summary_mit_laufzeit_all = mit_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
        summary_mit_laufzeit_all.columns =  [ 'mean_all', 'median_all','std_all', 'min_all', 'max_all', 'count_all']

        
        mit_laufzeit.sort_values(['date', 'plz', variables_dict[selected_variable]], ascending=[True, True, True], inplace=True)
        mit_laufzeit['rank'] = 1
        mit_laufzeit['rank'] = mit_laufzeit.groupby(['date', 'plz'])['rank'].cumsum()
        if(top_n != 'Alle'):
            mit_laufzeit = mit_laufzeit[mit_laufzeit['rank'] <= int(top_n)]

        summary_mit_laufzeit = mit_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
        summary_mit_laufzeit.columns =  [ 'mean', 'median','std', 'min', 'max', 'count']
        summary_mit_laufzeit['date'] = summary_mit_laufzeit.index
        summary_mit_laufzeit['beschreibung'] = 'Nicht-Öko'

        summary = pd.concat([summary_mit_laufzeit, summary_ohne_laufzeit])
        summary_all = pd.concat([summary_mit_laufzeit_all, summary_ohne_laufzeit_all])
        print('all: ',len(summary_all))
        print('summary: ',len(summary))
        summary = pd.concat([summary, summary_all], axis=1)

        print('summary before merge: ',len(summary),'   ',summary.columns)

        summary = summary.reset_index(drop=True).copy()

        summary = pd.merge(left=summary,
                 right=global_summary,
                 how='left',
                 on='date')

        print('summary after merge: ',len(summary),'   ',summary.columns)
    elif(seperation_var == 'signupPartner'):
        ohne_laufzeit  = results[results[seperation_var] == 'vx']
        ohne_laufzeit_all = ohne_laufzeit.copy()

        summary_ohne_laufzeit_all = ohne_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
        summary_ohne_laufzeit_all.columns =  [ 'mean_all', 'median_all','std_all', 'min_all', 'max_all', 'count_all']

        ohne_laufzeit.sort_values(['date', 'plz', variables_dict[selected_variable]], ascending=[True, True, True], inplace=True)
        ohne_laufzeit['rank'] = 1
        ohne_laufzeit['rank'] = ohne_laufzeit.groupby(['date', 'plz'])['rank'].cumsum()
        if(top_n != 'Alle'):
            ohne_laufzeit = ohne_laufzeit[ohne_laufzeit['rank'] <= int(top_n)]

        summary_ohne_laufzeit = ohne_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
        summary_ohne_laufzeit.columns =  [ 'mean', 'median','std', 'min', 'max', 'count']
        summary_ohne_laufzeit['date'] = summary_ohne_laufzeit.index
        summary_ohne_laufzeit['beschreibung'] = 'Verivox'
        
        #mit_laufzeit  = high_consume[high_consume['contractDurationNormalized'] > 11]
        mit_laufzeit  = results[results[seperation_var] == 'c24']
        mit_laufzeit_all = mit_laufzeit.copy()
        summary_mit_laufzeit_all = mit_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
        summary_mit_laufzeit_all.columns =  [ 'mean_all', 'median_all','std_all', 'min_all', 'max_all', 'count_all']

        
        mit_laufzeit.sort_values(['date', 'plz', variables_dict[selected_variable]], ascending=[True, True, True], inplace=True)
        mit_laufzeit['rank'] = 1
        mit_laufzeit['rank'] = mit_laufzeit.groupby(['date', 'plz'])['rank'].cumsum()
        if(top_n != 'Alle'):
            mit_laufzeit = mit_laufzeit[mit_laufzeit['rank'] <= int(top_n)]

        summary_mit_laufzeit = mit_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
        summary_mit_laufzeit.columns =  [ 'mean', 'median','std', 'min', 'max', 'count']
        summary_mit_laufzeit['date'] = summary_mit_laufzeit.index
        summary_mit_laufzeit['beschreibung'] = 'Check24'

        summary = pd.concat([summary_mit_laufzeit, summary_ohne_laufzeit])
        summary_all = pd.concat([summary_mit_laufzeit_all, summary_ohne_laufzeit_all])
        print('all: ',len(summary_all))
        print('summary: ',len(summary))
        summary = pd.concat([summary, summary_all], axis=1)

        print('summary before merge: ',len(summary),'   ',summary.columns)

        summary = summary.reset_index(drop=True).copy()

        summary = pd.merge(left=summary,
                 right=global_summary,
                 how='left',
                 on='date')
    elif(seperation_var=='None'):

        ohne_laufzeit = None
        ohne_laufzeit_all = None
        mit_laufzeit  = results.copy()
        mit_laufzeit_all = results.copy()
        summary_mit_laufzeit_all = mit_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
        summary_mit_laufzeit_all.columns =  [ 'mean_all', 'median_all','std_all', 'min_all', 'max_all', 'count_all']
        
        mit_laufzeit.sort_values(['date', 'plz', variables_dict[selected_variable]], ascending=[True, True, True], inplace=True)
        mit_laufzeit['rank'] = 1
        mit_laufzeit['rank'] = mit_laufzeit.groupby(['date', 'plz'])['rank'].cumsum()
        if(top_n != 'Alle'):
            mit_laufzeit = mit_laufzeit[mit_laufzeit['rank'] <= int(top_n)]

        summary_mit_laufzeit = mit_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
        summary_mit_laufzeit.columns =  [ 'mean', 'median','std', 'min', 'max', 'count']
        summary_mit_laufzeit['date'] = summary_mit_laufzeit.index
        summary_mit_laufzeit['beschreibung'] = 'Verbrauch: '+consumption

        summary = summary_mit_laufzeit.copy()
        summary_all = summary_mit_laufzeit_all.copy()
        print('all: ',len(summary_all))
        print('summary: ',len(summary))
        summary = pd.concat([summary, summary_all], axis=1)

        print('summary before merge: ',len(summary),'   ',summary.columns)

        summary = summary.reset_index(drop=True).copy()

        summary = pd.merge(left=summary,
                 right=global_summary,
                 how='left',
                 on='date')
    elif(seperation_var == 'commoncarrier'):
        ohne_laufzeit  = results[results[seperation_var] == 'no']
        ohne_laufzeit_all = ohne_laufzeit.copy()

        summary_ohne_laufzeit_all = ohne_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
        summary_ohne_laufzeit_all.columns =  [ 'mean_all', 'median_all','std_all', 'min_all', 'max_all', 'count_all']

        ohne_laufzeit.sort_values(['date', 'plz', variables_dict[selected_variable]], ascending=[True, True, True], inplace=True)
        ohne_laufzeit['rank'] = 1
        ohne_laufzeit['rank'] = ohne_laufzeit.groupby(['date', 'plz'])['rank'].cumsum()
        if(top_n != 'Alle'):
            ohne_laufzeit = ohne_laufzeit[ohne_laufzeit['rank'] <= int(top_n)]

        summary_ohne_laufzeit = ohne_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
        summary_ohne_laufzeit.columns =  [ 'mean', 'median','std', 'min', 'max', 'count']
        summary_ohne_laufzeit['date'] = summary_ohne_laufzeit.index
        summary_ohne_laufzeit['beschreibung'] = 'Sondertrarife'
        
        #mit_laufzeit  = high_consume[high_consume['contractDurationNormalized'] > 11]
        mit_laufzeit  = results[results[seperation_var] == 'yes']
        mit_laufzeit_all = mit_laufzeit.copy()
        summary_mit_laufzeit_all = mit_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
        summary_mit_laufzeit_all.columns =  [ 'mean_all', 'median_all','std_all', 'min_all', 'max_all', 'count_all']

        mit_laufzeit.sort_values(['date', 'plz', variables_dict[selected_variable]], ascending=[True, True, True], inplace=True)
        mit_laufzeit['rank'] = 1
        mit_laufzeit['rank'] = mit_laufzeit.groupby(['date', 'plz'])['rank'].cumsum()
        if(top_n != 'Alle'):
            mit_laufzeit = mit_laufzeit[mit_laufzeit['rank'] <= int(top_n)]

        summary_mit_laufzeit = mit_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
        summary_mit_laufzeit.columns =  [ 'mean', 'median','std', 'min', 'max', 'count']
        summary_mit_laufzeit['date'] = summary_mit_laufzeit.index
        summary_mit_laufzeit['beschreibung'] = 'Grundversorgunstarife'

        summary = pd.concat([summary_mit_laufzeit, summary_ohne_laufzeit])
        summary_all = pd.concat([summary_mit_laufzeit_all, summary_ohne_laufzeit_all])
        print('all: ',len(summary_all))
        print('summary: ',len(summary))
        summary = pd.concat([summary, summary_all], axis=1)

        print('summary before merge: ',len(summary),'   ',summary.columns)

        summary = summary.reset_index(drop=True).copy()

        summary = pd.merge(left=summary,
                 right=global_summary,
                 how='left',
                 on='date')
        
    return ohne_laufzeit, mit_laufzeit, ohne_laufzeit_all, mit_laufzeit_all, summary

def create_plotly_chart(summary, aggregation='mean', seperation_value=12, widtht=880, height=300, selected_variable='dataunit', events_df=None, seperation_var='priceGuaranteeNormalized'):
    
    aggregation_dict = {
        "Durchschnitt": "mean",
        "Median": "median",
        "Standardabweichung": "std",
        "Minimum":"min",
        "Maximum":"max",
        "mean":"Durchschnitt",
        "median":"Median",
        "min":"Minimum",
        "max":"Maximum",
        "std":"Standardabweichung",
        "count_all":"count_all"
        }

    color_discrete_map = {'Kurze Preisgarantie': '#4650DF', 'Lange Preisgarantie': '#FC6E44', 
                          'Vertragslaufzeit >= 12': '#4650DF', 'Vertragslaufzeit < 12': '#FC6E44',
                          'Check24': '#4650DF', 'Verivox': '#FC6E44',
                          'Öko':'#4650DF', 'Nicht-Öko':'#FC6E44',
                          'Verbrauch: 3000':'#4650DF','Verbrauch: 15000':'#4650DF',
                          'Grundversorgunstarife':'#4650DF','Sondertrarife':'#FC6E44'}

    fig = px.line(summary, 
                x="date", 
                y=aggregation_dict[aggregation], 
                color='beschreibung', color_discrete_map=color_discrete_map
                )
    fig.update_traces(mode="markers+lines", hovertemplate=None)

    fig2 = px.bar(summary, 
        x="date", y="count_all", color="beschreibung", color_discrete_map=color_discrete_map
        )

    fig.update_yaxes(
        autorange = True,
        fixedrange = False
    )

    fig2.update_yaxes(
        autorange = True,
        fixedrange = False
    )

    fig2.update_traces(showlegend=False)


    figure1_traces = []
    figure2_traces = []

    for trace in range(len(fig["data"])):
        figure1_traces.append(fig["data"][trace])
    for trace in range(len(fig2["data"])):
        figure2_traces.append(fig2["data"][trace])

    this_figure = go.FigureWidget(sp.make_subplots(rows=2, cols=1, shared_xaxes=True, shared_yaxes=True, 
                                vertical_spacing=0.3,
                                row_heights=[0.7, 0.3]))

    # Get the Express fig broken down as traces and add the traces to the proper plot within in the subplot
    for traces in figure1_traces:
        this_figure.append_trace(traces, row=1, col=1)
    for traces in figure2_traces:
        this_figure.append_trace(traces, row=2, col=1)

    this_figure.update_layout(template='plotly_dark')

    #fig.update_yaxes(tickcolor="black", tickwidth=200, ticklen=10, ticklabelmode="period")

    fig.update_yaxes(title_text="<b>kWh Preis</b> Y - axis ", secondary_y=False)
    fig2.update_yaxes(title_text="<b>Suchanfragenergebnisse</b> Y - axis ", secondary_y=False)


    this_figure['layout']['yaxis1'].update(title='kWh Preis',
                                        autorange = True,
                                        ticks="outside",
                                        fixedrange = False,
                                        ticksuffix = ' Cent',
                                        tickformat = ',.')

    this_figure['layout']['xaxis1'].update(showgrid=False)

    this_figure['layout']['yaxis2'].update(title='Anfragenergebnisse',
                                        autorange = True,
                                        ticks="outside",
                                        fixedrange = False,
                                        showgrid=True)   
    

    this_figure.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                        label="1 Monat",
                        step="month",
                        stepmode="backward"),
                    dict(count=3,
                        label="3 Monate",
                        step="month",
                        stepmode="backward"),
                    dict(count=1,
                        label="1 Jahr",
                        step="year",
                        stepmode="backward"),
                    dict(label="Alles",
                        step="all")
                ]),
                bgcolor='#FC6E44'
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date",
            showspikes=True
        ),
        yaxis=dict(
            showspikes=True
        ),
        yaxis2=dict(
            showspikes=True
        ),
        hovermode="x",
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)")

    this_figure.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    
    this_figure.update_traces(hovertemplate='%{y}')
    
    return this_figure

def load_events_df():
    events_df = pd.DataFrame([
        {
            "start": "2022-07-01",
            "end":"2022-07-01",
            "ereignis": "Abschaffung EEG Umlage",
            "tooltip":"EEG Umlage wurde abgeschafft.",
            "intervall":False
        },
        {
            "start": "2022-02-24",
            "end":"2022-02-24",
            "ereignis": "Krieg in der Ukraine",
            "tooltip":"Invasion in der Ukraine",
            "intervall":False
        },
        {
            "start": "2022-10-01",
            "end":"2022-10-01",
            "ereignis": "Mehrwertsteuersenkung für Gas",
            "tooltip":"Mehrwertsteuer für Gas wurde von 19% auf 7% gesenkt.",
            "intervall":False
        },
        {
            "start": "2022-06-30",
            "end":"2022-06-30",
            "ereignis": "Alarmstufe des Notfallplans Gas",
            "tooltip":"Seit dem 23.06.2022 gilt die Alarmstufe des Notfallplans.",
            "intervall":False
        },
        {
            "start": "2022-06-05",
            "end":"2022-07-09",
            "ereignis": "Drosselung der Gasliefermenge auf 40%",
            "tooltip":"Drosselung der Gasliefermenge auf 40%",
            "intervall":True
        },
        {
            "start": "2022-07-10",
            "end":"2022-07-20",
            "ereignis": "Keine Gaslieferung durch Nord Stream 1",
            "tooltip":"Mehrwertsteuer für Gas wurde von 19% auf 7% gesenkt.",
            "intervall":True
        },
        {
            "start": "2022-07-21",
            "end":"2022-07-27",
            "ereignis": "Gasliefermenge nach Wartung weiterhin auf 40% gedrosselt",
            "tooltip":"Gasliefermenge nach Wartung weiterhin auf 40% gedrosselt",
            "intervall":True
        },
        {
            "start": "2022-09-01",
            "end":"2022-09-01",
            "ereignis": "Keine Gasimporte mehr aus Russland",
            "tooltip":"Keine Gasimporte aus Russland. https://www.bundesnetzagentur.de/DE/Gasversorgung/aktuelle_gasversorgung/_svg/Gasimporte/Gasimporte.html;jsessionid=BC4D6020F61B843F1C0FB52C4384DE6E",
            "intervall":False
        },
        {
            "start": "2022-07-28",
            "end":"2022-08-30",
            "ereignis": "Weitere Drosselung der Gasliefermenge auf 20% reduziert",
            "tooltip":"Weitere Drosselung der Gasliefermenge auf 20% reduziert",
            "intervall":True
        },
        {
            "start": "2022-09-26",
            "end":"2022-09-29",
            "ereignis": "Explosionen NordStream 1 & 2",
            "tooltip":"In der Nacht zum Montag, dem 26. September 2022, fiel der Druck in einer der beiden Röhren der Pipeline NordStream 2 stark ab. Montagabend meldete dann auch der Betreiber von NordStream 1 einen Druckabfall – in diesem Fall für beide Röhren der Pipeline. Am Dienstag teilte die dänische Energiebehörde mit, es gebe insgesamt drei Gaslecks nahe der Insel Bornholm – zwei Lecks an NordStream 1 nordöstlich der Ostsee-Insel sowie eines an NordStream 2 südöstlich der Insel. Zudem zeichneten Messstationen auf schwedischem und dänischem Hoheitsgebiet am Montag mächtige Unterwasser-Explosionen auf. Die Schwedische Küstenwache teilte am 29. September 20022 mit, dass ein viertes Leck in den NordStream-Pipelines entdeckt wurde. [Quelle: WWF]",
            "intervall":False
        }

    ])
    return events_df


def get_table(results, selected_date, rng, dom):

    top_n_strom_tarife = results.copy()
    top_n_strom_tarife = top_n_strom_tarife[top_n_strom_tarife.date == selected_date][['plz','rank', 'providerName', 'tariffName', 'signupPartner','dataunit','datafixed', 'Jahreskosten',  'dataeco', 'priceGuaranteeNormalized', 'contractDurationNormalized']]
        

    gd = GridOptionsBuilder.from_dataframe(top_n_strom_tarife)
    gd.configure_pagination(enabled=True)
    gd.configure_default_column(editable=False, groupable=True)
    gd.configure_selection(selection_mode='single', use_checkbox=False, pre_selected_rows=[0])
    #gd.configure_column("date", type=["customDateTimeFormat"], custom_format_string='yyyy-MM-dd')
    gd.configure_column("rank", header_name="Rang")
    gd.configure_column("plz", header_name="PLZ")
    gd.configure_column("signupPartner", header_name="Partner")
    gd.configure_column("providerName", header_name="Versorger")
    gd.configure_column("tariffName", header_name="Tarif Name")
    gd.configure_column('dataunit', header_name="Arbeitspreis")
    gd.configure_column('datafixed', header_name="Grundpreis")
    gd.configure_column('dataeco', header_name="Öko")
    gd.configure_column('priceGuaranteeNormalized', header_name="Preisgarantie")
    gd.configure_column('contractDurationNormalized', header_name="Vertragslaufzeit")
    #um date picker einzufügen: https://discuss.streamlit.io/t/ag-grid-component-with-input-support/8108/349?page=17
    gridoptions = gd.build()

    grid_table = AgGrid(top_n_strom_tarife, 
        gridOptions=gridoptions, 
        update_mode=GridUpdateMode.GRID_CHANGED, 
        enable_enterprise_modules= True,
        #fit_columns_on_grid_load=True,
        height = 350,
        width=805,
        allow_unsafe_jscode=True,
        theme='alpine'
        )

    return grid_table, top_n_strom_tarife

def get_tariff_table(results, selected_date):

    std_grundpreis = results.datafixed.std()
    mean_grundpreis = results.datafixed.mean()
    abnormal_high = 900

    cellstyle_jscode_numeric = JsCode("""
        function(params){{
            if(params.value > ({mean} + 3*{std})){{
                return {{
                        'color':'black',
                        'backgroundColor':'yellow'
                    }}
                }}
            else if((params.value > {abnormal_high}) | (params.value < 0) ){{
                return {{
                        'color':'black',
                        'backgroundColor':'red'
                    }}
                }}
            }};
    """.format(std=std_grundpreis, mean=mean_grundpreis, abnormal_high=abnormal_high))

    top_n_strom_tarife = results.copy()
    top_n_strom_tarife = top_n_strom_tarife[top_n_strom_tarife.date == selected_date][['date','plz', 'providerName', 'tariffName','signupPartner', 'dataunit', 'kwh_price','datafixed', 'contractDurationNormalized', 'priceGuaranteeNormalized', 'commoncarrier', 'recommended']]
    top_n_strom_tarife = top_n_strom_tarife.drop_duplicates(['providerName', 'tariffName', 'signupPartner'])

    gd = GridOptionsBuilder.from_dataframe(top_n_strom_tarife)
    gd.configure_pagination(enabled=True)
    gd.configure_default_column(editable=False, groupable=True)
    gd.configure_selection(selection_mode='single', use_checkbox=False)
    gd.configure_column("date", type=["customDateTimeFormat"], custom_format_string='yyyy-MM-dd')
    gd.configure_column("plz", header_name="PLZ")
    gd.configure_column("dataunit", header_name="Arbeitspreis")
    #gd.configure_column("dataunit_c24", header_name="Arbeitspreis C24")

    gd.configure_column("datafixed", header_name="Grundpreis", hide=True)
    gd.configure_column("kwh_price", header_name="kWh Preis", sort='asc')
    if('commoncarrier' in top_n_strom_tarife.columns):
        gd.configure_column('commoncarrier', header_name="Grundversorgungstarif")

    gd.configure_column('recommended', header_name="Finanztip Empfählung", hide=True)
    gd.configure_column("providerName", header_name="Versorger")
    gd.configure_column("tariffName", header_name="Tarifname")
    gd.configure_column('contractDurationNormalized', header_name="Vertragslaufzeit")
    gd.configure_column('priceGuaranteeNormalized', header_name="Preisgarantie")
    
    #um date picker einzufügen: https://discuss.streamlit.io/t/ag-grid-component-with-input-support/8108/349?page=17
    gridoptions = gd.build()

    grid_table = AgGrid(top_n_strom_tarife, 
        gridOptions=gridoptions, 
        update_mode=GridUpdateMode.GRID_CHANGED, 
        enable_enterprise_modules= True,
        #fit_columns_on_grid_load=True,
        height = 650,
        width=805,
        allow_unsafe_jscode=True,
        theme='alpine'
        )

    return grid_table

def get_provider_table(results, selected_date):

    std_grundpreis = results.datafixed.std()
    mean_grundpreis = results.datafixed.mean()
    abnormal_high = 900

    cellstyle_jscode_numeric = JsCode("""
        function(params){{
            if(params.value > ({mean} + 3*{std})){{
                return {{
                        'color':'black',
                        'backgroundColor':'yellow'
                    }}
                }}
            else if((params.value > {abnormal_high}) | (params.value < 0) ){{
                return {{
                        'color':'black',
                        'backgroundColor':'red'
                    }}
                }}
            }};
    """.format(std=std_grundpreis, mean=mean_grundpreis, abnormal_high=abnormal_high))

    days_results = results[results.date==selected_date]

    tariffs = days_results.groupby("providerName")["tariffName"].unique().agg(list)
    plzs = days_results.groupby("providerName")["plz"].unique().agg(list)

    days_results["Tariffs"] = days_results["providerName"].map(tariffs[tariffs.str.len().ge(1)])
    days_results["plzs"] = days_results["providerName"].map(plzs[plzs.str.len().ge(1)])

    days_results = days_results.drop_duplicates('providerName')
    days_results["TariffsCount"] = [len(tariffs) for tariffs in days_results["Tariffs"]]
    days_results["plzCount"] = [len(tariffs) for tariffs in days_results["plzs"]]
    days_results = days_results[['providerName','Tariffs', 'TariffsCount', 'plzs','plzCount']]

    gd = GridOptionsBuilder.from_dataframe(days_results)
    gd.configure_pagination(enabled=True)
    gd.configure_default_column(editable=False, groupable=True)
    gd.configure_selection(selection_mode='single', use_checkbox=False)
    gd.configure_column("plzs", header_name="Verfügbar in",wrapText= True)
    gd.configure_column("plzCount", header_name="Anzahl PLZ's")
    #gd.configure_column("dataunit_c24", header_name="Arbeitspreis C24")

    gd.configure_column("Tariffs", header_name="Tarife",wrapText= True)
    gd.configure_column("TariffsCount", header_name="Anzahl Tarife", sort='asc')
    #if('commoncarrier' in top_n_strom_tarife.columns):
    #    gd.configure_column('commoncarrier', header_name="Grundversorgungstarif"
    
    #um date picker einzufügen: https://discuss.streamlit.io/t/ag-grid-component-with-input-support/8108/349?page=17
    gridoptions = gd.build()

    grid_table = AgGrid(days_results, 
        gridOptions=gridoptions, 
        update_mode=GridUpdateMode.GRID_CHANGED, 
        enable_enterprise_modules= True,
        #fit_columns_on_grid_load=True,
        height = 650,
        width=805,
        allow_unsafe_jscode=True,
        theme='alpine'
        )

    return grid_table

def get_provider_table_for_plz(results):

    std_grundpreis = results.datafixed.std()
    mean_grundpreis = results.datafixed.mean()
    abnormal_high = 900

    cellstyle_jscode_numeric = JsCode("""
        function(params){{
            if(params.value > ({mean} + 3*{std})){{
                return {{
                        'color':'black',
                        'backgroundColor':'yellow'
                    }}
                }}
            else if((params.value > {abnormal_high}) | (params.value < 0) ){{
                return {{
                        'color':'black',
                        'backgroundColor':'red'
                    }}
                }}
            }};
    """.format(std=std_grundpreis, mean=mean_grundpreis, abnormal_high=abnormal_high))

    

    gd = GridOptionsBuilder.from_dataframe(results)
    gd.configure_pagination(enabled=True)
    gd.configure_default_column(editable=False, groupable=True)
    gd.configure_selection(selection_mode='single', use_checkbox=False)
    gd.configure_column("providerName", header_name="Anbieter",wrapText= True)
    gd.configure_column("tariffName", header_name="Tarifname")
    #gd.configure_column("dataunit_c24", header_name="Arbeitspreis C24")

    gd.configure_column("dataunit", header_name="Arbeitspreis",wrapText= True)
    gd.configure_column("kwh_price", header_name="kWh Preis", sort='asc')
    gd.configure_column("contactDurationNormalized", header_name="Vertragslaufzeit")
    gd.configure_column("priceGruaranteeNormalized", header_name="Preisgarantie")
    gd.configure_column("dataeco", header_name="Ökotarif")
    #if('commoncarrier' in top_n_strom_tarife.columns):
    #    gd.configure_column('commoncarrier', header_name="Grundversorgungstarif"
    
    #um date picker einzufügen: https://discuss.streamlit.io/t/ag-grid-component-with-input-support/8108/349?page=17
    gridoptions = gd.build()

    grid_table = AgGrid(results, 
        gridOptions=gridoptions, 
        update_mode=GridUpdateMode.GRID_CHANGED, 
        enable_enterprise_modules= True,
        #fit_columns_on_grid_load=True,
        height = 650,
        width=805,
        allow_unsafe_jscode=True,
        theme='alpine'
        )

    return grid_table

def get_tariff_table_comparison_signupPartner(results, selected_date):


    jscode = """function(params){{if( (params.value > 0) &  (params.value < {val})){{return {{'color':'black','backgroundColor':'orange'}}}}}}""".format(val='0.01')
    st.write(jscode)
    cellstyle_jscode_datatotal = JsCode(jscode)

    top_n_strom_tarife = results.copy()
    top_n_strom_tarife = top_n_strom_tarife[top_n_strom_tarife.date == selected_date][['plz', 'providerName', 'tariffName', 'dataunit_vx', 'dataunit_c24', 'datafixed_vx', 'datafixed_c24', 'datatotal_vx', 'datatotal_c24', 'delta_datatotal']]
    top_n_strom_tarife = top_n_strom_tarife.drop_duplicates(['providerName'])
    st.write(top_n_strom_tarife)

    gd = GridOptionsBuilder.from_dataframe(top_n_strom_tarife)
    gd.configure_pagination(enabled=True)
    gd.configure_default_column(editable=False, groupable=True)
    gd.configure_selection(selection_mode='single', use_checkbox=False)
    #gd.configure_column("date", type=["customDateTimeFormat"], custom_format_string='yyyy-MM-dd')
    gd.configure_column("plz", header_name="PLZ")
    gd.configure_column("dataunit_vx", header_name="Arbeitspreis VX")
    #gd.configure_column("dataunit_c24", header_name="Arbeitspreis C24")

    gd.configure_column("datafixed_vx", header_name="Grundpreis VX")
    gd.configure_column("datafixed_c24", header_name="Grundpreis C24")

    gd.configure_column("datatotal_vx", header_name="Jahreskosten VX")
    gd.configure_column("datatotal_c24", header_name="Jahreskosten C24")
    gd.configure_column("delta_datatotal", header_name="Delta Jahreskosten", cellStyle=cellstyle_jscode_datatotal)

    gd.configure_column("providerName", header_name="Versorger")
    gd.configure_column("tariffName", header_name="Tarif Name")
    
    #um date picker einzufügen: https://discuss.streamlit.io/t/ag-grid-component-with-input-support/8108/349?page=17
    gridoptions = gd.build()

    grid_table = AgGrid(top_n_strom_tarife, 
        gridOptions=gridoptions, 
        update_mode=GridUpdateMode.GRID_CHANGED, 
        enable_enterprise_modules= True,
        #fit_columns_on_grid_load=True,
        height = 650,
        width=805,
        allow_unsafe_jscode=True,
        theme='alpine'
        )

    return grid_table

def seperate_data(data_all, selected_date_e, seperation_var='priceGuaranteeNormalized', selection_short=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], selection_long=[12, 13, 14, 15, 16, 24]):
    
    data_all = data_all[data_all.date == selected_date_e]

    color_discrete_map = {'Kurze Preisgarantie': '#4650DF', 'Lange Preisgarantie': '#FC6E44', 
                          'Vertragslaufzeit >= 12': '#4650DF', 'Vertragslaufzeit < 12': '#FC6E44',
                          'Check24': '#4650DF', 'Verivox': '#FC6E44',
                          'Öko':'#4650DF', 'Nicht-Öko':'#FC6E44',
                          'Verbrauch: 3000':'#4650DF',
                          'Grundversorgunstarife':'#4650DF','Sondertrarife':'#FC6E44'}
    
    if(seperation_var == 'Vertragslaufzeit'):
        seperation_var = 'contractDurationNormalized'
        left_color= '#4650DF'
        right_color='#FC6E44'
    elif(seperation_var == 'Preisgarantie'):
        left_color= '#4650DF'
        right_color='#FC6E44'
        seperation_var='priceGuaranteeNormalized'
    elif(seperation_var == 'Öko Tarif/ Konventioneller Tarif'):
        left_color= '#4650DF'
        right_color='#FC6E44'
        seperation_var = 'dataeco'
    elif(seperation_var == 'Partner'):
        left_color= '#4650DF'
        right_color='#FC6E44'
        seperation_var = 'signupPartner'
    elif(seperation_var== 'Kein Unterscheidungsmerkmal'):
        left_color= '#4650DF'
        right_color='None'
        seperation_var = 'None'
    elif(seperation_var == 'Grundversorgungstarif'):
        left_color= '#4650DF'
        right_color='#FC6E44'
        seperation_var = 'commoncarrier'


    if((seperation_var == 'priceGuaranteeNormalized')  | (seperation_var == 'contractDurationeNormalized')):
        left_data = data_all[data_all[seperation_var].isin(selection_short)]
        right_data = data_all[data_all[seperation_var].isin(selection_long)]
        
        return left_data, right_data, left_color, right_color
    elif(seperation_var == 'dataeco'):
        return data_all[data_all[seperation_var] == True], data_all[data_all[seperation_var] == False], left_color, right_color
    elif(seperation_var == 'signupPartner'):
        return data_all[data_all[seperation_var] == 'c24'], data_all[data_all[seperation_var] == 'vx'], left_color, right_color
    elif(seperation_var == 'commoncarrier'):
        return data_all[data_all[seperation_var] == 'yes'], data_all[data_all[seperation_var] == 'no'], left_color, right_color
    elif(seperation_var == 'None'):
        return data_all, None, left_color, right_color

def create_desitiy_chart(data_all,maximum, selected_date_e, selected_variable, color):
    selectd_variable_dict = {
        "Arbeitspreis": "dataunit",
        "dataunit": "Arbeitspreis",
        "Grundpreis": "datafixed",
        "kWh Preis":"kwh_price",
        "Jahreskosten":"Jahreskosten"}

    data_all = data_all[data_all.date == selected_date_e]
    

    data_all = data_all.drop_duplicates(['providerName', 'tariffName', 'kwh_price'])

    counts, bins = np.histogram(data_all[selectd_variable_dict[selected_variable]], bins=range(0, int(maximum), 5))
    bins = 0.5 * (bins[:-1] + bins[1:])
    
    fig = px.bar(x=bins, y=counts, labels={'x':selectd_variable_dict[selected_variable], 'y':'count'})
    fig.update_traces(marker_color=color)

    return fig

def create_plz_tariff_densitiy_chart(data_all, selected_date_e, color):
    selectd_variable_dict = {
        "Arbeitspreis": "dataunit",
        "dataunit": "Arbeitspreis",
        "Grundpreis": "datafixed",
        "kWh Preis":"kwh_price",
        "Jahreskosten":"Jahreskosten"}
    restults_at_date = data_all[data_all['date'] == selected_date_e]
    tariffs = restults_at_date.groupby("plz")["tariffName"].unique().agg(list)

    restults_at_date["Tariffs"] = restults_at_date["plz"].map(tariffs[tariffs.str.len().ge(1)])
    restults_at_date = restults_at_date.drop_duplicates('plz')
    restults_at_date["TariffsCount"] = [len(tariffs) for tariffs in restults_at_date["Tariffs"]]

    restults_at_date = restults_at_date[['plz','Tariffs', 'TariffsCount']]
    
    fig = px.bar(x=restults_at_date['plz'].astype(str), y=restults_at_date['TariffsCount'], labels={'x':'Postleitzahl', 'y':'Anzahl Tarife'})
    fig.update_traces(marker_color='#E3D1FF')
    #fig.update_xaxes(type='category')
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={'type':'category',
            'categoryorder':'total descending'}
    )
    #fig.update_traces(marker_color=color)

    return fig

def summarize_tariff(results, selected_variable):

    selectd_variable_dict = {
        "Arbeitspreis": "dataunit",
        "dataunit": "Arbeitspreis",
        "Grundpreis": "datafixed",
        "datafixed":"Grundpreis",
        "Jahreskosten":"Jahreskosten"}

    results[selectd_variable_dict[selected_variable]+'_mean'] = results.groupby('date')[selectd_variable_dict[selected_variable]].transform('mean')
    #st.write(results[['date', 'dataunit', 'dataunit_mean']])
    return results
                
def create_tarif_chart(source, selected_variable):

    selectd_variable_dict = {
        "Arbeitspreis": "dataunit",
        "dataunit": "Arbeitspreis",
        "Grundpreis": "datafixed",
        "datafixed":"Grundpreis",
        "Jahreskosten":"Jahreskosten"}

    variable = selectd_variable_dict[selected_variable]+'_mean'

    min_y = source[variable].min() - (0.02 * source[variable].min())
    max_y = source[variable].max() + (0.02 * source[variable].max() )

    tarif_y_domain = [min_y, max_y]
    #st.write(list(tarif_y_domain))

    tarif_chart = alt.Chart(source).mark_circle(size=50).encode(
                    x = alt.X('date:T', axis=alt.Axis(format="%d %b %y", grid=False, title='Datum', labelAngle=45)),
                    y = alt.Y(variable+':Q', scale=alt.Scale(domain=list(tarif_y_domain)),axis= alt.Axis(title=selected_variable, offset=10)),
                    tooltip=['date:T',variable+':Q']
                ).properties(width=550, height=150, title='')
    return tarif_chart

def create_tarif_summary_section(results, grid_table_df, index, selected_variable):
    index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
    tariffName = grid_table_df.iloc[index]['tariffName']
    providerName = grid_table_df.iloc[index]['providerName']
    tarif_df = results[(results.tariffName == tariffName) & (results.providerName == providerName)]
    tarif_df = summarize_tariff(tarif_df, selected_variable)
    tarif_chart = create_tarif_chart(tarif_df, selected_variable)

    st.write('<div style="text-align: center">Entwicklung - Tarif \"{tariffName}\" vom Anbieter \"{providerName}\"'.format(tariffName = tariffName, providerName = providerName), unsafe_allow_html=True)
    st.altair_chart(tarif_chart)

#### HEADER REGION

empty_left_head, row1_1,center_head, row1_2, empty_right_head= st.columns((1,4, 1,4,1))

with row1_1:
    st.title(" Strom 🔌 & 🔥 Gas - Dashboard ")
    
with row1_2:
    st.write(
        """Hier noch eine kurze Beschreibung des Dashboard einfügen. """
    )

    dateset_description_expander = st.expander('Datensatz', expanded=False)

    with dateset_description_expander:
        st.write(
            """
        **Was**\n
        Für fünf Postleitzahlen werden die Tarife für jeweils zwei unterschiedliche Energiemengen automatisiert abgerufen. \n\n
        **Rhythmus:**\n
        Die automatisierte Abfrage erfolgt einmal pro Woche. Es gibt keinen festen Tag dafür. \n
        **Postleitzahlen und Verbrauchsmengen:**\n\n
        **10245 Berlin** - 1.300 kWh Strom, 3.000 kWh Strom, 9.000 kWh Gas, 15.000 kWh Gas\n
        **99425 Weimar** - 1.300 kWh Strom, 3.000 kWh Strom, 9.000 kWh Gas, 15.000 kWh Gas\n
        **33100 Paderborn** - 1.300 kWh Strom, 3.000 kWh Strom, 9.000 kWh Gas, 15.000 kWh Gas\n
        **50670 Köln** - 1.300 kWh Strom, 3.000 kWh Strom, 9.000 kWh Gas, 15.000 kWh Gas\n
        **49661 Cloppenburg** - 1.300 kWh Strom, 3.000 kWh Strom, 9.000 kWh Gas, 15.000 kWh Gas\n
        """
        )

empty_left_chart_button, chart_left1_chart_button, chart_center_chart_button,chart_right1_chart_button, empty_right_chart_button = st.columns([1,0.5, 9,0.5, 1])
tab1, tab2, tab3, tab4  = chart_center_chart_button.tabs(["🔌 🔥 Energietyp", "Attributauswahl", "🍎🍏 Vergleichen", "Annotationen"])

### MENU AUSWAHL REGION

#division_selection_column, division_value_selection_column = selection_menu_container.columns([1,3])
selected_variable = tab2.selectbox(
    'Welches Attribut möchtest du anschauen?',
    ('Arbeitspreis', 'kWh Preis','Jahreskosten'),
    index=1)

mean_median_btn = tab2.selectbox(
        "Wie möchtest du aggregieren?",
        ("Durchschnitt", "Median", "Minimum", "Maximum", "Standardabweichung"),
        index=1)

top_n = tab1.selectbox(
            'Agregiere über Top N günstigste Tarife?',
            ['1','3', '5', '10', 'Alle'],
            index=3)
        
seperation_var = tab3.selectbox('Nach welchem Merkmal möchtest du aufteilen?',
    ('Kein Unterscheidungsmerkmal',"Grundversorgungstarif", 'Vertragslaufzeit', 'Preisgarantie', 'Öko Tarif/ Konventioneller Tarif','Partner'),
    index=0,
    help="Gebe hier ein nach welhes Attribut du trennen möchtest: Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At")
            
selection_slider = 12
selection_short = [0,0.5, 1, 1.5]
selection_long = [12,24]

if( (seperation_var =='Vertragslaufzeit') | (seperation_var =='Preisgarantie')  ):
    price_gurarantee_list = electricity_results_3000['priceGuaranteeNormalized'].unique()

    min_slider = sorted(price_gurarantee_list)[1]
    max_slider = sorted(price_gurarantee_list)[-2]

    selection_slider = tab3.slider('Teile '+seperation_var+ ' ab:', int(min_slider), int(max_slider), 12, step=1,
        help="Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At") 
    short = [float(item) for item in price_gurarantee_list if item < float(selection_slider)]
    long =  [float(item) for item in price_gurarantee_list if item >= float(selection_slider)]
    

    selection_short = tab3.multiselect(
    'Kurze  Preisgarantie',
    sorted(short),
    sorted(short))

    selection_long = tab3.multiselect(
    'Lange  Preisgarantie',
    sorted(long),
    sorted(long))

#### ANNOTATION REGION

empty_events1,center_events1, empty_right_events1= st.columns((1,10,1))

events_df = load_events_df()
annotation_container = tab4.expander('Ereignisse 📰🌟 - Hier kannst du Ereinisse in die Zeitachse der Grafiken einblenden oder entfernen', expanded=False)

with annotation_container:
    st.info('Ereignisse werden als vertikale Annotationslienien oder Intervalle auf die Zeitachse der Grafiken eingeblendet. Dies unterschtüzt das Storrytelling Charater der Grafik und das Betrachten von bestimmten Entwicklungen in Zusammenhang mit Ereignissen.')

    gd = GridOptionsBuilder.from_dataframe(events_df)
    gd.configure_pagination(enabled=True)
    gd.configure_default_column(editable=True, groupable=True)
    gd.configure_selection(selection_mode='multiple', use_checkbox=True)
    gd.configure_column("start", type=["customDateTimeFormat"], custom_format_string='yyyy-MM-dd')
    #um date picker einzufügen: https://discuss.streamlit.io/t/ag-grid-component-with-input-support/8108/349?page=17
    gridoptions = gd.build()

    grid_table = AgGrid(events_df, 
    gridOptions=gridoptions, 
    update_mode=GridUpdateMode.GRID_CHANGED, 
    enable_enterprise_modules= True,
    fit_columns_on_grid_load=True,
    #height = 300,
    width='100%',
    allow_unsafe_jscode=True,
    theme='alpine'
     )

    sel_row = grid_table['selected_rows']

    inxexes_of_selected = []
    for i, event in enumerate(sel_row):
        #st.write(event['ereignis'])
        inxexes_of_selected.append(int(events_df.loc[(events_df.start == event['start']) & (events_df.end == event['end']) ].index[0]))
        events_df.loc[(events_df.start == event['start']) & (events_df.end == event['end']), 'ereignis' ] = event['ereignis']
        #st.write(events_df)

    selected_events = events_df.iloc[inxexes_of_selected]

## ENDE ANNOTATION REGION
electricity_chart_column, gas_chart_column = st.columns(2) 

empty_chartHeader_left, chart_header_line_left1,chart_header_line_left11, chart_header_middle,  empty_chart_header_line_right1,  empty_chart_header_line_right11,empty_chart_header_right = st.columns([1,0.5,3,  3,  3,0.5, 1])

ohne_laufzeit_1300, ohne_laufzeit_9000, ohne_laufzeit_3000, ohne_laufzeit_15000 = None, None, None, None
mit_laufzeit_1300, mit_laufzeit_9000, mit_laufzeit_3000, mit_laufzeit_15000 = None, None, None, None

empty_left, chart_left1, chart_center,chart_right1, empty_right = st.columns([1,0.5, 9,0.5, 1])

with chart_center_chart_button:
    energy_type = tab1.selectbox(
                'Energie Typ',
                ['Strom','Gas'],
                index=0)
        
    if(energy_type == 'Strom'):
        ohne_laufzeit_3000, mit_laufzeit_3000, ohne_laufzeit_3000_all, mit_laufzeit_3000_all, summary_3000 = summarize(electricity_results_3000, seperation_var, selection_short, selection_long,'3000',selected_variable, top_n=top_n)
        summary_e = summary_3000
    elif(energy_type == 'Gas'):
        ohne_laufzeit_15000, mit_laufzeit_15000, ohne_laufzeit_15000_all, mit_laufzeit_15000_all, summary_15000 = summarize(gas_results_15000, seperation_var,selection_short, selection_long,'15000',selected_variable, top_n=top_n)
        summary_g = summary_15000


if( (mean_median_btn == 'Maximum') | (mean_median_btn == 'Minimum') ):
    if(seperation_var != 'Kein Unterscheidungsmerkmal'):
        if((top_n != 'Alle')  & (top_n != '1') ):
            chart_center_chart_button.write(mean_median_btn+' der '+selected_variable+' aller top '+top_n+' günstigsten '+energy_type+' Tarife unterschieden nach '+seperation_var)
        elif(top_n == 'Alle'):
            chart_center_chart_button.write(mean_median_btn+' der '+selected_variable+' aller '+energy_type+' Tarife unterschieden nach '+seperation_var)
        elif(top_n == '1'):
            chart_center_chart_button.write(mean_median_btn+' '+selected_variable+' der günstigsten '+energy_type+' Tarife unterschieden nach '+seperation_var)
    else:
        if((top_n != 'Alle')  & (top_n != '1') ):
            chart_center_chart_button.write(mean_median_btn+' der '+selected_variable+' aller top '+top_n+' günstigsten '+energy_type+' Tarife')
        elif(top_n == 'Alle'):
            chart_center_chart_button.write(mean_median_btn+' der '+selected_variable+' aller '+energy_type+' Tarife')
        elif(top_n == '1'):
            chart_center_chart_button.write(mean_median_btn+' '+selected_variable+' der günstigsten '+energy_type+' Tarife')
elif((mean_median_btn == "Durchschnitt") | (mean_median_btn == 'Median') ):
    if(seperation_var != 'Kein Unterscheidungsmerkmal'):
        if((top_n != 'Alle')  & (top_n != '1') ):
            chart_center_chart_button.info(mean_median_btn+'e der '+selected_variable+' aller top '+top_n+' günstigsten '+energy_type+' Tarife unterschieden nach '+seperation_var)
        elif(top_n == 'Alle'):
            chart_center_chart_button.info(mean_median_btn+'e der' +selected_variable+' aller '+energy_type+' Tarife unterschieden nach '+seperation_var)
        elif(top_n == '1'):
            chart_center_chart_button.info(mean_median_btn+'e der '+selected_variable+' aller günstigsten '+energy_type+' Tarife unterschieden nach '+seperation_var)
    else:
        if((top_n != 'Alle')  & (top_n != '1') ):
            chart_center_chart_button.info(mean_median_btn+'e der top '+top_n+' günstigsten '+energy_type+' Tarife')
        elif(top_n == 'Alle'):
            chart_center_chart_button.info(mean_median_btn+'e aller '+energy_type+' Tarife')
        elif(top_n == '1'):
            chart_center_chart_button.info(mean_median_btn+'e der günstigsten '+energy_type+' Tarife')
elif((mean_median_btn == "Standardabweichung")):
    if(seperation_var != 'Kein Unterscheidungsmerkmal'):
        if((top_n != 'Alle')  & (top_n != '1') ):
            chart_center_chart_button.write(mean_median_btn+'en der '+selected_variable+' aller top '+top_n+' günstigsten '+energy_type+' Tarife unterschieden nach '+seperation_var)
        elif(top_n == 'Alle'):
            chart_center_chart_button.write(mean_median_btn+'en aller '+energy_type+' Tarife unterschieden nach '+seperation_var)
        elif(top_n == '1'):
            chart_center_chart_button.write(mean_median_btn+'en der günstigsten '+energy_type+' Tarife unterschieden nach '+seperation_var)
    else:
        if((top_n != 'Alle')  & (top_n != '1') ):
            chart_center_chart_button.write(mean_median_btn+'en der top '+top_n+' günstigsten '+energy_type+' Tarife')
        elif(top_n == 'Alle'):
            chart_center_chart_button.write(mean_median_btn+'en aller '+energy_type+' Tarife')
        elif(top_n == '1'):
            chart_center_chart_button.write(mean_median_btn+'en der günstigsten '+energy_type+' Tarife')

if(energy_type == 'Strom'):
    chart_header = "**{energy_selection}verträge ({selected_variable})**".format(selected_variable=selected_variable, energy_selection='Strom')
    energy_line_chart_e = create_plotly_chart(summary_e, mean_median_btn, int(selection_slider),  selected_variable=selected_variable, events_df=selected_events,seperation_var=seperation_var)
    chart_center.write(chart_header)
    with chart_center:
        output = plotly_events(energy_line_chart_e)

    if(len(output) > 0):
        empty_left_densitiy_chart,empty_left1_densitiy_chart,  chart_center_density,empty_right1_densitiy_chart, empty_right_density_chart = st.columns([1,0.5,9,0.5, 1])

        maximum = electricity_results_3000[selectd_variable_dict[selected_variable]].max() + 5
        left_data, right_data, left_color, right_color = seperate_data(electricity_results_3000, output[0]['x'], seperation_var, selection_short, selection_long)

        histogramm_left = create_desitiy_chart(left_data,maximum, output[0]['x'],selected_variable, left_color)
        selected_date_dt = dt.strptime(output[0]['x'], '%Y-%m-%d')
        selected_date = dt.strftime(selected_date_dt, '%b %d, %Y')

        chart_center_density.write('Am '+selected_date+':')

        num_provider = electricity_results_3000[electricity_results_3000['date'] == selected_date_dt].providerName.unique()
        provider_grouped = electricity_results_3000[electricity_results_3000['date'] == selected_date_dt].groupby('providerName').mean()

        summaryTab, providerTab, tariffTab, resultsTab  = chart_center_density.tabs([" Zusammenfassung", " Zusammenfassung (Anbieter)", " Zusammenfassung (Tarife)", " Zusammenfassung (Anfragenergenbisse)"])
        providerTab.write('Anzahl Anbieter: '+str(len(num_provider)))

        restults_at_date = electricity_results_3000[electricity_results_3000['date'] == selected_date_dt]

        if(seperation_var!="Kein Unterscheidungsmerkmal"):
            histogramm_right = create_desitiy_chart(right_data,maximum, selected_date_dt,selected_variable, right_color)

            with summaryTab:

                total_results = str(len(restults_at_date))
                without_duplicates = str(len(restults_at_date.drop_duplicates(['providerName', 'tariffName', 'kwh_price'])))
                st.info('Folgendes Histogramm zeigt die Verteilung der Suchanfaragenergebnisse am '+selected_date+'. Es zeigt nicht die Verteilung aller Anfragenegebnisse (Insgesammt '+total_results+'). Duplikate die den gleihen Anbieternamen, Tarifnamen und kWh Preis haben werden entfernt (Insgesammt '+without_duplicates+').')
                
                st.plotly_chart(histogramm_left, use_container_width=True)
                st.plotly_chart(histogramm_right, use_container_width=True)
        else:
            with summaryTab:
                total_results = str(len(restults_at_date))
                without_duplicates = str(len(restults_at_date.drop_duplicates(['providerName', 'tariffName', 'kwh_price'])))
                st.info('Folgendes Histogramm zeigt die Verteilung der Suchanfaragenergebnisse am '+selected_date+'. Es zeigt nicht die Verteilung aller Anfragenegebnisse (Insgesammt '+total_results+'). Duplikate die den gleihen Anbieternamen, Tarifnamen und kWh Preis haben werden entfernt (Insgesammt '+without_duplicates+').')
                
                summaryTab.plotly_chart(histogramm_left, use_container_width=True)


                tariffs = restults_at_date.groupby("plz")["tariffName"].unique().agg(list)

                restults_at_date["Tariffs"] = restults_at_date["plz"].map(tariffs[tariffs.str.len().ge(1)])
                restults_at_date = restults_at_date.drop_duplicates('plz')
                restults_at_date["TariffsCount"] = [len(tariffs) for tariffs in restults_at_date["Tariffs"]]

                restults_at_date = restults_at_date[['plz','Tariffs', 'TariffsCount']]


                st.write('Verteilung der Tarife über die Postleitzahlen:')
                st.write('Durchschnittlich '+str(restults_at_date['TariffsCount'].median())+' Tarife pro Postleitzahl mit '+str(int(restults_at_date['TariffsCount'].std()))+' Standardabweichung.')
                st.write('Am wenigsten Tarife in: '+restults_at_date.iloc[restults_at_date['TariffsCount'].argmin()]['plz']+' mit '+str(restults_at_date.iloc[restults_at_date['TariffsCount'].argmin()]['TariffsCount'])+' Tarifen.')
                st.write('Die meisten Tarife in: '+restults_at_date.iloc[restults_at_date['TariffsCount'].argmax()]['plz']+' mit '+str(restults_at_date.iloc[restults_at_date['TariffsCount'].argmax()]['TariffsCount'])+' Tarifen.')
                plz_tariff_density_chart = create_plz_tariff_densitiy_chart(electricity_results_3000,  selected_date_dt, 'blue')
                plz_tariffs_selection = plotly_events(plz_tariff_density_chart)



                selected_date_tariffs = electricity_results_3000[(electricity_results_3000['plz'] == plz_tariffs_selection[0]['x']) & (electricity_results_3000['date'] == selected_date_dt)]
                #maximum = selected_date_tariffs[selectd_variable_dict[selected_variable]].max() + 5

                st.write('Verteilung der Tarife in '+plz_tariffs_selection[0]['x']+' am '+selected_date)
                plz_tariffs_density_chart = create_desitiy_chart(selected_date_tariffs,maximum, selected_date_dt,selected_variable, 'blue')
                st.plotly_chart(plz_tariffs_density_chart, use_container_width=True)
                tariffs_on_date_df = electricity_results_3000[(electricity_results_3000['plz'] == plz_tariffs_selection[0]['x']) & (electricity_results_3000['date'] == selected_date_dt)].groupby(['providerName','tariffName', 'signupPartner']).agg('mean').reset_index()
                st.write('Tarife die am '+selected_date+' in '+plz_tariffs_selection[0]['x']+' angezeigt wurden:')
                get_provider_table_for_plz(tariffs_on_date_df)
                #st.write(tariffs_on_date_df)

        with providerTab:
            
            
            grid_table = get_provider_table(electricity_results_3000,  selected_date_dt)
            
            
elif(energy_type == 'Gas'):
    chart_header = "**{energy_selection}verträge ({selected_variable})**".format(selected_variable=selected_variable, energy_selection='Gas')
    chart_center.write(chart_header)
    energy_line_chart_g = create_plotly_chart(summary_g, mean_median_btn, int(selection_slider),  selected_variable=selected_variable, events_df=selected_events, seperation_var=seperation_var)
        
    with chart_center:
        output = plotly_events(energy_line_chart_g)

    if(len(output) > 0):
        empty_left_densitiy_chart,empty_left1_densitiy_chart, left_densitiy_chart, chart_center_density,right_densitiy_chart,empty_right1_densitiy_chart, empty_right_density_chart = st.columns([1,0.5,4, 1,4,0.5, 1])

        maximum = gas_results_15000[selectd_variable_dict[selected_variable]].max() + 5
        left_data, right_data, left_color, right_color = seperate_data(gas_results_15000, output[0]['x'], seperation_var, selection_short, selection_long)

        histogramm_left = create_desitiy_chart(left_data,maximum, output[0]['x'],selected_variable, left_color)

        chart_center_density.write(output[0]['x']['Tariffs'])

        if(seperation_var!="Kein Unterscheidungsmerkmal"):
            histogramm_right = create_desitiy_chart(right_data,maximum, output[0]['x'],selected_variable, right_color)
            left_densitiy_chart.plotly_chart(histogramm_left, use_container_width=True)
            right_densitiy_chart.plotly_chart(histogramm_right, use_container_width=True)
        else:
            left_densitiy_chart.plotly_chart(histogramm_left, use_container_width=True)

empty_left_tariff_table_chart,empty_left1_tariff_table_chart, left_tariff_table_chart, chart_center_tariff_table,right_tariff_table_chart,empty_right1_tariff_table_chart, empty_right_tariff_table_chart = st.columns([1,0.5,0.01, 9,0.01,0.5, 1])
if(len(output) > 0):
    with resultsTab:
        if(energy_type == 'Strom'):
            get_tariff_table(electricity_results_3000, output[0]['x'])
        elif(energy_type == 'Gas'):
            get_tariff_table(gas_results_15000, output[0]['x'])


