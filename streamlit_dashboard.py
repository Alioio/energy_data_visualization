import numpy as np
import pandas as pd
import polars as pl
import os
import re
import datetime
import timeit
from datetime import datetime as dt
from datetime import date, timedelta
import altair as alt
import altair_catplot as altcat
alt.renderers.set_embed_options(tooltip={"theme": "dark"})
alt.data_transformers.disable_max_rows()
import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import threading
import concurrent.futures
import json


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

    all_dates = all_dates[(all_dates.plz == 10245) | 
                        (all_dates.plz == 99425) |  
                        (all_dates.plz == 33100)  |  
                        (all_dates.plz == 50670) |  
                        (all_dates.plz == 49661)]
    print('DATASET 1 UNIQUE PLZs: ',all_dates.plz.unique())

    ## Filter nach Verbrauch und wandle dataunit in float um
    all_dates = all_dates.drop_duplicates(['date', 'providerName', 'tariffName', 'signupPartner', 'plz'])

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

    '''
    gas_path = 'data/{energy_type}'.format(energy_type=energy_type)
    files = os.listdir(gas_path)
    print(files)

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

    all_dates = all_dates[(all_dates.plz == 10245) | 
                        (all_dates.plz == 99425) |  
                        (all_dates.plz == 33100)  |  
                        (all_dates.plz == 50670) |  
                        (all_dates.plz == 49661)]
    print('DATASET 1 UNIQUE PLZs: ',all_dates.plz.unique())

    ## Filter nach Verbrauch und wandle dataunit in float um
    all_dates = all_dates.drop_duplicates(['date', 'providerName', 'tariffName', 'signupPartner', 'plz'])
    all_dates = all_dates[['date', 'providerName','tariffName', 'signupPartner', 'dataunit', 'contractDurationNormalized', 'priceGuaranteeNormalized', 'dataeco']]
    
    all_dates['dataunit'] = all_dates['dataunit'].str.replace(',','.').astype(float)

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
    #wa_df = wa_df[wa_df.date < all_dates.date.min()]
    wa_df = wa_df.drop_duplicates(['date', 'providerName', 'tariffName', 'signupPartner', 'plz'])
    
    #all_dates = pd.concat([wa_df, all_dates])
    all_dates = wa_df.copy()
    ###

    data_types_dict = {'date':'<M8[ns]', 'providerName':str, 'tariffName':str,'signupPartner':str, 'dataunit':float, 'dataeco':bool, 'plz':str, 'datafixed':float, 'Jahreskosten':float}
    all_dates = all_dates.astype(data_types_dict)

    print('MIT DEM EINLESEN DER 5 PLZ DATEN FERTIG ',energy_type)
    return all_dates[['date', 'providerName', 'tariffName', 'signupPartner', 'plz', 'dataunit',  'datafixed','Jahreskosten','contractDurationNormalized', 'priceGuaranteeNormalized', 'dataeco']]


with concurrent.futures.ThreadPoolExecutor() as executor:
    electricity_reader_thread_3000 = executor.submit(read_energy_data, 'electricity', '3000')
    electricity_reader_thread_1300 = executor.submit(read_energy_data, 'electricity', '1300')
    gas_reader_thread_15000 = executor.submit(read_energy_data, 'gas', '15000')
    gas_reader_thread_9000 = executor.submit(read_energy_data, 'gas', '9000')

    #electricity_reader_thread_100_3000 = executor.submit(read_energy_data_100, 'electricity', '3000')
    #electricity_reader_thread_100_1300 = executor.submit(read_energy_data_100, 'electricity', '1300')
    #gas_reader_thread_100_15000 = executor.submit(read_energy_data_100, 'gas', '15000')
    #gas_reader_thread_100_9000 = executor.submit(read_energy_data_100, 'gas', '9000')

    
    electricity_results_3000 = electricity_reader_thread_3000.result()
    electricity_results_1300 = electricity_reader_thread_1300.result()
    gas_results_15000 = gas_reader_thread_15000.result()
    gas_results_9000 = gas_reader_thread_9000.result()

    #electricity_results_100_3000 = electricity_reader_thread_100_3000.result()
    #electricity_results_100_1300 = electricity_reader_thread_100_1300.result()
    #gas_results_100_15000 = gas_reader_thread_100_15000.result()
    #gas_results_100_9000 = gas_reader_thread_100_9000.result()

def summarize_polars(results, seperation_var='priceGuaranteeNormalized',seperation_value=12, consumption='unknown',selected_variable='dataunit', top_n = '10'):

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


    variables_dict = {
        "Arbeitspreis": "dataunit",
        "Grundpreis": "datafixed",
        "Jahreskosten": "Jahreskosten"
        }



    results_pl = pl.from_pandas(results)
    global_summary = results_pl.groupby(['date']).count()
    global_summary = global_summary.rename({'count':'count_global'})
    #global_summary = global_summary.to_pandas()

    agg_functions_all = [ pl.mean(variables_dict[selected_variable]).alias('mean_all'), pl.median(variables_dict[selected_variable]).alias('median_all'), pl.std(variables_dict[selected_variable]).alias('std_all'), pl.min(variables_dict[selected_variable]).alias('min_all'), pl.max(variables_dict[selected_variable]).alias('max_all'), pl.count(variables_dict[selected_variable]).alias('count_all')]
    agg_functions = [ pl.mean(variables_dict[selected_variable]).alias('mean'), pl.median(variables_dict[selected_variable]).alias('median'), pl.std(variables_dict[selected_variable]).alias('std'), pl.min(variables_dict[selected_variable]).alias('min'), pl.max(variables_dict[selected_variable]).alias('max'), pl.count(variables_dict[selected_variable]).alias('count')]
    
    if( (seperation_var == 'contractDurationNormalized') | (seperation_var == 'priceGuaranteeNormalized') ):

        results = pl.from_pandas(results)

        ohne_laufzeit = results.filter(pl.col(seperation_var) < seperation_value)
        ohne_laufzeit_all = ohne_laufzeit

        summary_ohne_laufzeit_all = ohne_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions_all)
        #st.write(summary_ohne_laufzeit_all)
        #summary_ohne_laufzeit_all.columns =  [ 'mean_all', 'median_all','std_all', 'min_all', 'max_all', 'count_all']
  
        ohne_laufzeit = ohne_laufzeit.sort(['date', 'plz', variables_dict[selected_variable]], [False, False, False])
        ohne_laufzeit = ohne_laufzeit.with_column(pl.lit(1).alias("rank2"))
        #st.write(ohne_laufzeit)

        out = ohne_laufzeit.select(
        [
            pl.col("rank2").cumsum().over(["date", "plz"]).alias("rank")
        ]
        )

        ohne_laufzeit = pl.concat([ohne_laufzeit, out], how="horizontal")
        ohne_laufzeit = ohne_laufzeit[['date','plz', 'providerName','tariffName','signupPartner','dataunit', 'datafixed', 'Jahreskosten', 'dataeco','priceGuaranteeNormalized', 'contractDurationNormalized', 'rank']]
        if(top_n != 'Alle'):
            ohne_laufzeit = ohne_laufzeit.filter(pl.col('rank') <= int(top_n))

        summary_ohne_laufzeit = ohne_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
        #summary_ohne_laufzeit.columns =  [ 'mean', 'median','std', 'min', 'max', 'count']
        #summary_ohne_laufzeit['date'] = summary_ohne_laufzeit.index
        summary_ohne_laufzeit = summary_ohne_laufzeit.with_column(pl.lit('Verbrauch: '+consumption+'\n'+sep_var_readable+' < '+str(seperation_value)).alias('beschreibung') )
        #st.write(summary_ohne_laufzeit)
        
        #mit_laufzeit  = high_consume[high_consume['contractDurationNormalized'] > 11]
        mit_laufzeit = results.filter(pl.col(seperation_var) >= seperation_value)
        mit_laufzeit_all = mit_laufzeit
        summary_mit_laufzeit_all = mit_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions_all)
        #summary_mit_laufzeit_all.columns =  [ 'mean_all', 'median_all','std_all', 'min_all', 'max_all', 'count_all']

        
        mit_laufzeit = mit_laufzeit.sort(['date', 'plz', variables_dict[selected_variable]], [False, False, False])
        mit_laufzeit = mit_laufzeit.with_column(pl.lit(1).alias("rank2"))
        #st.write(ohne_laufzeit)

        out = mit_laufzeit.select(
        [
            pl.col("rank2").cumsum().over(["date", "plz"]).alias("rank")
        ]
        )

        mit_laufzeit = pl.concat([mit_laufzeit, out], how="horizontal")
        mit_laufzeit = mit_laufzeit[['date','plz', 'providerName','tariffName','signupPartner','dataunit', 'datafixed', 'Jahreskosten', 'dataeco','priceGuaranteeNormalized', 'contractDurationNormalized', 'rank']]
        if(top_n != 'Alle'):
            mit_laufzeit = mit_laufzeit.filter(pl.col('rank') <= int(top_n))

        summary_mit_laufzeit = mit_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
        #summary_mit_laufzeit.columns =  [ 'mean', 'median','std', 'min', 'max', 'count']
        #summary_mit_laufzeit['date'] = summary_mit_laufzeit.index
        summary_mit_laufzeit = summary_mit_laufzeit.with_column(pl.lit('Verbrauch: '+consumption+'\n'+sep_var_readable+' >= '+str(seperation_value)).alias('beschreibung') )

        summary = pl.concat([summary_mit_laufzeit, summary_ohne_laufzeit])
        summary_all = pl.concat([summary_mit_laufzeit_all, summary_ohne_laufzeit_all])
        print('all: ',len(summary_all))
        print('summary: ',len(summary))
        summary_all = summary_all[['mean_all', 'median_all', 'std_all', 'min_all', 'max_all', 'count_all']]
        summary = pl.concat([summary, summary_all], how="horizontal")

        print('summary before merge: ',len(summary),'   ',summary.columns)

        #summary = summary.reset_index(drop=True).copy()

        summary = summary.join(global_summary, on="date", how="left")

        #summary_all = summary_all.to_pandas()
        summary = summary.to_pandas()
        mit_laufzeit = mit_laufzeit.to_pandas()
        ohne_laufzeit = ohne_laufzeit.to_pandas()
        mit_laufzeit_all = mit_laufzeit_all.to_pandas()
        ohne_laufzeit_all = ohne_laufzeit_all.to_pandas()

        #summary = pl.merge(left=summary,
        #         right=global_summary,
        #         how='left',
        #         on='date')


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
        summary_ohne_laufzeit['beschreibung'] = 'Verbrauch: '+consumption+' und Öko'
        
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
        summary_mit_laufzeit['beschreibung'] = 'Verbrauch: '+consumption+' und Nicht-Öko'

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
        summary_ohne_laufzeit['beschreibung'] = 'Verbrauch: '+consumption+' Verivox'
        
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
        summary_mit_laufzeit['beschreibung'] = 'Verbrauch: '+consumption+' Check24'

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


    return ohne_laufzeit, mit_laufzeit, ohne_laufzeit_all, mit_laufzeit_all, summary

def summarize(results, seperation_var='priceGuaranteeNormalized',seperation_value=12, consumption='unknown',selected_variable='dataunit', top_n = '10'):

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


    variables_dict = {
        "Arbeitspreis": "dataunit",
        "Grundpreis": "datafixed",
        "Jahreskosten": "Jahreskosten"
        }



    results_pl = pl.from_pandas(results)
    global_summary = results_pl.groupby(['date']).count()
    global_summary = global_summary.rename({'count':'count_global'})
    global_summary = global_summary.to_pandas()

    agg_functions = {
        variables_dict[selected_variable]:
        [ 'mean', 'median','std', 'min', 'max', 'count']
    }
    
    if( (seperation_var == 'contractDurationNormalized') | (seperation_var == 'priceGuaranteeNormalized') ):

        ohne_laufzeit  = results[results[seperation_var] < seperation_value]
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
        summary_ohne_laufzeit['beschreibung'] = 'Verbrauch: '+consumption+'\n'+sep_var_readable+' < '+str(seperation_value) 
        
        #mit_laufzeit  = high_consume[high_consume['contractDurationNormalized'] > 11]
        mit_laufzeit  = results[results[seperation_var] >= seperation_value]
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
        summary_mit_laufzeit['beschreibung'] = 'Verbrauch: '+consumption+'\n'+sep_var_readable+' >= '+str(seperation_value)

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
        summary_ohne_laufzeit['beschreibung'] = 'Verbrauch: '+consumption+' und Öko'
        
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
        summary_mit_laufzeit['beschreibung'] = 'Verbrauch: '+consumption+' und Nicht-Öko'

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
        summary_ohne_laufzeit['beschreibung'] = 'Verbrauch: '+consumption+' Verivox'
        
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
        summary_mit_laufzeit['beschreibung'] = 'Verbrauch: '+consumption+' Check24'

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


    return ohne_laufzeit, mit_laufzeit, ohne_laufzeit_all, mit_laufzeit_all, summary


def create_chart(summary, aggregation='mean', seperation_value=12, date_interval=['2022-07-17', '2022-10-17'], widtht=500, height=180,selected_variable='dataunit', events_df=None, energy_type='gas', seperation_var='priceGuaranteeNormalized'):

    if(seperation_var== 'Kein Unterscheidungsmerkmal'):
        seperation_var = 'None'

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
        "std":"Standardabweichung"
        }

    aggregation = aggregation_dict[aggregation]

    ## Definitionsbereich der Y achse
    min = np.floor(summary[aggregation].min() - (0.025*summary[aggregation].min()))
    max = np.ceil( summary[aggregation].max() + (0.025*summary[aggregation].max()))
    domain1 = np.linspace(min, max, 2, endpoint = True)
    
    #chart view scaling
    chart_min = summary[(summary.date >= pd.to_datetime(date_interval[0])) & (summary.date <= pd.to_datetime(date_interval[1])) ][aggregation].min() 
    chart_max = summary[(summary.date >= pd.to_datetime(date_interval[0])) & (summary.date <= pd.to_datetime(date_interval[1])) ][aggregation].max()
    
    chart_min = np.floor(chart_min - (0.025*chart_min))
    chart_max = np.ceil( chart_max + (0.025*chart_max))
    domain2 = np.linspace(chart_min, chart_max, 2, endpoint = True)

    #count view scaling
    
    x_init = pd.to_datetime(date_interval).astype(int) / 1E6
    interval = alt.selection_interval(encodings=['x'],init = {'x':x_init.to_list()})
    
    chart_max = summary['count_global'].max()
    chart_max = np.ceil( chart_max + (1.2*chart_max))
    domain3 = np.linspace(0, chart_max, 2, endpoint = True)
    
    source = summary.copy()
    
    selection = alt.selection_multi(fields=['beschreibung'], bind='legend')
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['date'], empty='none')
    
    interval_y = alt.selection_interval(encodings=['y'], bind="scales")

    ## Eregignisse

    rule = alt.Chart(events_df[events_df.intervall == False]).mark_rule(
        color="gray",
        strokeWidth=2,
        strokeDash=[12, 6]
    ).encode(
        x= alt.X('start:T', scale=alt.Scale(domain=interval.ref()))
    )

    rect = alt.Chart(events_df[events_df.intervall == True]).mark_rect(opacity=0.3, color= 'gray').encode(
    x= alt.X('start:T', scale=alt.Scale(domain=interval.ref())),
    x2='end:T',
    tooltip='ereignis:N'
    )

    events_text = alt.Chart(events_df[events_df.intervall == False]).mark_text(
        align='left',
        baseline='middle',
        dx=0.25*height*-1,
        dy=-7,
        size=11,
        angle=270,
        color='gray'
    ).encode(
        x= alt.X('start:T', scale=alt.Scale(domain=interval.ref())),
        text='ereignis',
        tooltip='tooltip'
    )

    ## Visualisierung:
    y_axis_title = selected_variable

    if(selected_variable == 'Arbeitspreis'):
        y_axis_title = 'ct/kWh'
    elif(selected_variable == 'Grundpreis'):
        y_axis_title = '€/Monat'
    elif(selected_variable == 'Jahreskosten'):
        y_axis_title = '€ im ersten Jahr'
    else:
        y_axis_title = 'etwas anderes'

    print(source.beschreibung.unique())

    print('SEP VAR: ',seperation_var,'   ',seperation_value)
    if((energy_type == 'gas') & (seperation_var != 'Öko Tarif/ Konventioneller Tarif') & (seperation_var != 'Partner') & (seperation_var != 'None')):
        rng = ['#4650DF','#FC6E44', '#006E78', '#20B679']
        dom = [source[source.beschreibung.str.contains('15000') & source.beschreibung.str.contains('<')].iloc[0].beschreibung,
              source[source.beschreibung.str.contains('15000') & source.beschreibung.str.contains('>=')].iloc[0].beschreibung,
              source[source.beschreibung.str.contains('9000') & source.beschreibung.str.contains('<')].iloc[0].beschreibung,
              source[source.beschreibung.str.contains('9000') & source.beschreibung.str.contains('>=')].iloc[0].beschreibung]
    elif((energy_type == 'electricity') & (seperation_var != 'Öko Tarif/ Konventioneller Tarif') & (seperation_var != 'Partner')& (seperation_var != 'None')):
        rng = ['#4650DF','#FC6E44', '#006E78', '#20B679']
        dom = [source[source.beschreibung.str.contains('3000') & source.beschreibung.str.contains('<')].iloc[0].beschreibung,
              source[source.beschreibung.str.contains('3000') & source.beschreibung.str.contains('>=')].iloc[0].beschreibung,
              source[source.beschreibung.str.contains('1300') & source.beschreibung.str.contains('<')].iloc[0].beschreibung,
              source[source.beschreibung.str.contains('1300') & source.beschreibung.str.contains('>=')].iloc[0].beschreibung]
    elif((energy_type == 'gas') & (seperation_var == 'Öko Tarif/ Konventioneller Tarif')):
        rng = ['#4650DF','#FC6E44', '#006E78', '#20B679']
        dom = [source[source.beschreibung.str.contains('15000') & ~source.beschreibung.str.contains('Nicht-Öko')].iloc[0].beschreibung,
              source[source.beschreibung.str.contains('15000') & source.beschreibung.str.contains('Nicht-Öko')].iloc[0].beschreibung,
              source[source.beschreibung.str.contains('9000') & ~source.beschreibung.str.contains('Nicht-Öko')].iloc[0].beschreibung,
              source[source.beschreibung.str.contains('9000') & source.beschreibung.str.contains('Nicht-Öko')].iloc[0].beschreibung]
    elif((energy_type == 'electricity')  & (seperation_var == 'Öko Tarif/ Konventioneller Tarif')):
        rng = ['#4650DF','#FC6E44', '#006E78', '#20B679']
        dom = [source[source.beschreibung.str.contains('3000') & ~source.beschreibung.str.contains('Nicht-Öko')].iloc[0].beschreibung,
              source[source.beschreibung.str.contains('3000') & source.beschreibung.str.contains('Nicht-Öko')].iloc[0].beschreibung,
              source[source.beschreibung.str.contains('1300') & ~source.beschreibung.str.contains('Nicht-Öko')].iloc[0].beschreibung,
              source[source.beschreibung.str.contains('1300') & source.beschreibung.str.contains('Nicht-Öko')].iloc[0].beschreibung]
    elif((energy_type == 'gas') & (seperation_var == 'Partner')):
        rng = ['#4650DF','#FC6E44', '#006E78', '#20B679']
        dom = [source[source.beschreibung.str.contains('15000') & source.beschreibung.str.contains('Check24')].iloc[0].beschreibung,
              source[source.beschreibung.str.contains('15000') & source.beschreibung.str.contains('Verivox')].iloc[0].beschreibung,
              source[source.beschreibung.str.contains('9000') & source.beschreibung.str.contains('Check24')].iloc[0].beschreibung,
              source[source.beschreibung.str.contains('9000') & source.beschreibung.str.contains('Verivox')].iloc[0].beschreibung]
    elif((energy_type == 'electricity')  & (seperation_var == 'Partner')):
        rng = ['#4650DF','#FC6E44', '#006E78', '#20B679']
        dom = [source[source.beschreibung.str.contains('3000') & source.beschreibung.str.contains('Check24')].iloc[0].beschreibung,
              source[source.beschreibung.str.contains('3000') & source.beschreibung.str.contains('Verivox')].iloc[0].beschreibung,
              source[source.beschreibung.str.contains('1300') & source.beschreibung.str.contains('Check24')].iloc[0].beschreibung,
              source[source.beschreibung.str.contains('1300') & source.beschreibung.str.contains('Verivox')].iloc[0].beschreibung]
    elif((energy_type == 'electricity')  & (seperation_var == 'None')):
        rng = ['#4650DF', '#20B679']
        dom = [source[source.beschreibung.str.contains('3000')].iloc[0].beschreibung,
              source[source.beschreibung.str.contains('1300')].iloc[0].beschreibung]
    elif((energy_type == 'gas')  & (seperation_var == 'None')):
        rng = ['#4650DF', '#20B679']
        dom = [source[source.beschreibung.str.contains('15000')].iloc[0].beschreibung,
              source[source.beschreibung.str.contains('9000')].iloc[0].beschreibung]


    base = alt.Chart(source).mark_line(size=3).encode(
        #x= alt.X('date:T',axis= alt.Axis(grid=False, title='Datum')),
        y = alt.Y(aggregation+':Q', axis = alt.Axis(title=y_axis_title, offset= 5)),
        x= alt.X('date:T',axis= alt.Axis(grid=False, title='Datum 📅')),
        #y = alt.Y('median:Q', axis = alt.Axis(title='Arbeitspreis (ct/kWh)')),
        color=alt.Color('beschreibung:N', scale=alt.
                    Scale(domain=dom, range=rng))
    )
    
    chart = base.encode(
        x=alt.X('date:T',axis= alt.Axis(grid=False, title=''), scale=alt.Scale(domain=interval.ref())),
        y=alt.Y(aggregation+':Q', axis = alt.Axis(title=y_axis_title,  offset= 5), scale=alt.Scale(domain=list(domain2))),
        tooltip = alt.Tooltip(['date:T', aggregation+':Q', 'beschreibung:N']),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
    ).properties(
        width=widtht,
        height=height
    )

    #count_selector = alt.selection(type='single', encodings=['x'])

    count_chart = base.mark_bar(size=6.8).encode(
        x=alt.X('date:T',axis= alt.Axis(grid=False, title=''), scale=alt.Scale(domain=interval.ref())),
        y=alt.Y('count_all:Q', axis = alt.Axis(title='# Absolut'), scale=alt.Scale(domain=domain3)),
        color=alt.Color('beschreibung:N', scale=alt.Scale(domain=dom, range=rng)),
        opacity=alt.condition(nearest, alt.value(1), alt.value(0.5)),
        order=alt.Order(
        'count_all:Q',
      sort='descending'
    )
    ).properties(
        width=widtht,
        height=85
    ).add_selection(
        nearest
    )

    count_chart_normalized = base.mark_bar(size=6.8).encode(
        x=alt.X('date:T',axis= alt.Axis(grid=False, title=''), scale=alt.Scale(domain=interval.ref())),
        y=alt.Y('count_all:Q', axis = alt.Axis(title='# (Normalisiert)'), stack="normalize"),
        color=alt.Color('beschreibung:N', scale=alt.Scale(domain=dom, range=rng)),
        opacity=alt.condition(nearest, alt.value(1), alt.value(0.5)),
        order=alt.Order(
        'count_all:Q',
      sort='descending'
    )
    ).properties(
        width=widtht,
        height=60
    ).add_selection(
        nearest
    )
    
    view = base.encode(
        y = alt.Y(aggregation+':Q', axis = alt.Axis(title=y_axis_title),scale=alt.Scale(domain=list(domain1))),
    ).add_selection(
        interval
    ).properties(
        width=widtht,
        height=60,
    )

    ###############

    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(source).mark_point().encode(
        x='date:T',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )

    # Draw points on the line, and highlight based on selection
    points = chart.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Draw text labels near the points, and highlight based on selection
    text = chart.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, aggregation+':Q', alt.value(' '), format=".2f")
    )

    count_text = alt.Chart(source).mark_text(align='left', size=15).encode(
        text=alt.condition(nearest, 'count_all:Q', alt.value(' ')),
        y=alt.Y('row_number:O',axis=None),
        color=alt.Color('beschreibung:N', scale=alt.
                    Scale(domain=dom, range=rng))
    ).transform_filter(
        nearest
    ).transform_window(
        row_number='row_number()'
    ).properties(
        width=60,
        height=60
    )

    count_text_date = alt.Chart(source).mark_text(align='left', size=25).encode(
        text=alt.condition(nearest, 'date:T', alt.value(' ')),
        color=alt.value('#243039')
        #y=alt.Y('row_number:O',axis=None)
    ).transform_filter(
        nearest
    ).transform_window(
        row_number='row_number()'
    ).properties(
        width=60,
        height=60
    )


    # Draw a rule at the location of the selection
    rules = alt.Chart(source).mark_rule(color='gray').encode(
        x='date:T',
    ).transform_filter(
        nearest
    )

    # Put the five layers into a chart and bind the data
    
    main_view = alt.layer(
        chart , selectors, points, rules, text
    ).properties(
        width=widtht,
        height=height
    )

    print('im CREATE CHART: ',aggregation,'  ',aggregation_dict[aggregation])
    count_chart_view = alt.vconcat(count_chart ,
                                   count_chart_normalized,
                                   count_text_date ,
                                    (count_text.properties(title=alt.TitleParams(text='Anzahl Anfragenergebnisse', align='left')) | 
                                        count_text.encode(text=alt.condition(nearest, aggregation+':Q', alt.value(' '), format=".2f")).properties(title=alt.TitleParams(text=aggregation_dict[aggregation], align='left')) | 
                                        count_text.encode(text='beschreibung:N').properties(title=alt.TitleParams(text='Beschreibung', align='left'))))
    
    annotationen = rule + events_text + rect

    main_view = (main_view + annotationen)

    final_view = main_view.add_selection(
    selection
    ).interactive(bind_x=False)  & view & count_chart_view
    #& ranked_text

    final_view = final_view.configure_legend(
  orient='top',
  labelFontSize=10
)

    return final_view, rng, dom

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
    top_n_strom_tarife = top_n_strom_tarife[top_n_strom_tarife.date == selected_date][['plz', 'providerName', 'tariffName', 'dataunit', 'datafixed', 'datatotal', 'contractDurationNormalized', 'priceGuaranteeNormalized']]
        
    gd = GridOptionsBuilder.from_dataframe(top_n_strom_tarife)
    gd.configure_pagination(enabled=True)
    gd.configure_default_column(editable=False, groupable=True)
    gd.configure_selection(selection_mode='single', use_checkbox=False)
    #gd.configure_column("date", type=["customDateTimeFormat"], custom_format_string='yyyy-MM-dd')
    gd.configure_column("plz", header_name="PLZ")
    gd.configure_column("dataunit", header_name="Arbeitspreis")
    #gd.configure_column("dataunit_c24", header_name="Arbeitspreis C24")

    gd.configure_column("datafixed", header_name="Grundpreis", cellStyle=cellstyle_jscode_numeric)

    gd.configure_column("datatotal", header_name="Jahreskosten")

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


def get_tariff_table_comparison_signupPartner(results, selected_date):


    jscode = """function(params){{if( (params.value > 0) &  (params.value < {val})){{return {{'color':'black','backgroundColor':'orange'}}}}}}""".format(val='0.01')
    st.write(jscode)
    cellstyle_jscode_datatotal = JsCode(jscode)

    top_n_strom_tarife = results.copy()
    top_n_strom_tarife = top_n_strom_tarife[top_n_strom_tarife.date == selected_date][['plz', 'providerName', 'tariffName', 'dataunit_vx', 'dataunit_c24', 'datafixed_vx', 'datafixed_c24', 'datatotal_vx', 'datatotal_c24', 'delta_datatotal']]
        
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

def create_desitiy_chart(data_all,selected_date_e, selected_variable, rng,dom):
    data_all = data_all[data_all.date == selected_date_e]

    data_all = data_all.drop_duplicates(['providerName', 'tariffName', 'signupPartner', 'Jahreskosten', 'dataunit', 'datafixed'])

    print(data_all)
    selectd_variable_dict = {
        "Arbeitspreis": "dataunit",
        "dataunit": "Arbeitspreis",
        "Grundpreis": "datafixed",
        "datafixed":"Grundpreis",
        "Jahreskosten":"Jahreskosten"}

    max_dataunit = data_all[data_all.date == selected_date_e][selectd_variable_dict[selected_variable]].max()
    min_dataunit = data_all[data_all.date == selected_date_e][selectd_variable_dict[selected_variable]].min()

    density_chart_e = alt.Chart(data_all).mark_bar().encode(
        x= alt.X(selectd_variable_dict[selected_variable]+':Q',bin=True,axis= alt.Axis(grid=False, title=selected_variable)),
        y=alt.Y('count()', axis = alt.Axis(  offset= 5, title='Anzahl')),
        color=alt.Color('Beschreibung:N', scale=alt.Scale(domain=dom, range=rng) , legend=None),
        column = alt.Column('Beschreibung:N', title=''),
        tooltip=['count():Q',selectd_variable_dict[selected_variable]+':Q', 'providerName', 'tariffName', 'plz']
    ).properties(width=220, height=70, title='').interactive()

    return density_chart_e

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
                    x = alt.X('date:T', axis=alt.Axis(format="%y %b", grid=False, title='Datum')),
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

    st.write('<div style="text-align: center">Tarif \"{tariffName}\" vom Anbieter \"{providerName}\" für 1.3K kWh Verbrauch'.format(tariffName = tariffName, providerName = providerName), unsafe_allow_html=True)
    st.altair_chart(tarif_chart)

#### HEADER REGION

empty_left_head, row1_1,center_head, row1_2, empty_right_head= st.columns((1,4, 1,4,1))

with row1_1:
    st.title(" Strom 🔌 & 🔥 Gas - Dashboard ")
    
with row1_2:
    st.write(
        """
    ##
    **Dieses Dashboard ist zum Explorieren von Strom- und Gastarifdaten**. Hier einen kuzen Abschnitt einfügen welches diesen Dashboard beschreibt.
    Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. 
    At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.
    """
    )

    dateset_description_expander = st.expander('Datensatz', expanded=False)

    with dateset_description_expander:
        st.write(
            """
        ##
        **Datensatz:** Hier die Beschreibung der Quellen und Datensätze einfügen.
        """
        )

### END HEADER REGION
empty_breakline1,center_breakline1, empty_right_breakline1= st.columns((1,10,1))
center_breakline1.markdown("""---""")

### MENU AUSWAHL REGION
selection_menu_container = st.container()
empty_left_menu, time_selection_column,empty_menu1, attribute_selection_column,empty_menu2, left1_division_expander, empty_right_menu = selection_menu_container.columns([1, 1.5, 0.2, 3.6, 0.25, 4.4, 1])


#division_selection_column, division_value_selection_column = selection_menu_container.columns([1,3])

##Zeitintervallauswahl
today = date.today()
tree_months_ago = today - timedelta(days=90)
date_interval = [tree_months_ago, today]

time_selection = time_selection_column.selectbox(
    label='Zeitraum',
    options=('1 Monat', '3 Monat', '1 Jahr', 'Eigener Zeitraum'),
    index=1)

if(time_selection == '1 Monat'):
    with time_selection_column:
        tree_months_ago = today - timedelta(days=30)
        date_interval = [tree_months_ago, today]
elif(time_selection == '3 Monat'):
    with time_selection_column:
        tree_months_ago = today - timedelta(days=90)
        date_interval = [tree_months_ago, today]
elif(time_selection == '1 Jahr'):
    with time_selection_column:
        tree_months_ago = today - timedelta(days=365)
        date_interval = [tree_months_ago, today]
elif(time_selection == 'Eigener Zeitraum'):
    with time_selection_column:
        date_interval = st.date_input(label='',
                    value=(tree_months_ago, 
                            today),
                    key='#date_range',
                    help="Start-und End Datum: Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At")

#plz_list = electricity_results_3000['plz'].unique().tolist()
#print(electricity_results_3000['plz'].unique())
#plz_list.append('Alle')

#with time_selection_column:
#    st.multiselect(
#            'Tarife aus welchen Postleitzahlen soll enthalten sein?',
#            plz_list,
#            default=['Alle'])


#attribute_selection_column.write("**Attributauswahl**")

selected_variable = attribute_selection_column.selectbox(
    'Welches Attribut möchtest du anschauen?',
    ('Arbeitspreis', 'Grundpreis', 'Jahreskosten'),
    index=0)


mean_median_btn = attribute_selection_column.radio(
        ("").format(selected_variable=selected_variable),
        options=["Durchschnitt", "Median", "Minimum", "Maximum", "Standardabweichung"],
    )

with time_selection_column:
    top_n = st.selectbox(
                'Agregiere über Top N günstigste Tarife?',
                ['1','3', '5', '10', 'Alle'],
                index=3)


#empty_left_division_expander,left1_division_expander,center_division_expander,right1_division_expander, empty_right_division_expander= st.columns((1,4,1,4,1))
#division_expander = left1_division_expander.expander('Weiteres Unterscheidungsmerkmal 🍎🍏 - Hier kannst du ein weiteres Unterscheidungsmerkmal an welches du die Tarife aufteilen möchtest auswählen.', expanded=False)
        
seperation_var = left1_division_expander.selectbox('Nach welches Attribut möchtest du aufteilen?',
    ('Kein Unterscheidungsmerkmal', 'Vertragslaufzeit', 'Preisgarantie', 'Öko Tarif/ Konventioneller Tarif','Partner'),
    index=0,
    help="Gebe hier ein nach welhes Attribut du trennen möchtest: Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At")
            
selection_slider = 12

if( (seperation_var =='Vertragslaufzeit') |(seperation_var =='Preisgarantie')  ):
    selection_slider = left1_division_expander.slider('Ab welchen Wert für das Attribut '+seperation_var+ ' teilen?', 0, 24, 12, step=3,
    help="Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At")

### ENDE MENU AUSWAHL REGION

        
empty_breakline2,center_breakline2, empty_right_breakline2= st.columns((1,10,1))
center_breakline2.markdown("""---""")

#### ANNOTATION REGION

empty_events1,center_events1, empty_right_events1= st.columns((1,10,1))

events_df = load_events_df()
annotation_container = center_events1.expander('Ereignisse 📰🌟 - Hier kannst du Ereinisse in die Zeitachse der Grafiken einblenden oder entfernen', expanded=False)

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

energy_type_selections = ['Strom', 'Gas']
electricity_chart_column, gas_chart_column = st.columns(2) 

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left;} </style>', unsafe_allow_html=True)
st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)

#st.radio("",("Durchschnitt","Median"))
empty_chartHeader_left, chart_header_line_left1, chart_header_middle,  empty_chart_header_line_right1, empty_chart_header_right = st.columns([1,3.5,  3,  3.5, 1])

chart_header_line_left1.markdown("""---""")

empty_chart_header_line_right1.markdown("""---""")


if(seperation_var != 'Kein Unterscheidungsmerkmal'):
    short_description=('Preisentwicklung - {selected_variable} und {seperation_var} der Strom - und Gastarife').format(selected_variable=selected_variable, seperation_var=seperation_var).upper()
    long_description=('Die oberen zwei Grafiken zeigen die Entwicklung der Tarife bezüglich {selected_variable} und {seperation_var}. Im dritten Grafik ist die Anzahl der Suchanfragenergebnisse visualisiert.').format(selected_variable=selected_variable, seperation_var=seperation_var)
else:
    short_description=('Preisentwicklung - {selected_variable} der Strom - und Gastarife').format(selected_variable=selected_variable, seperation_var=seperation_var).upper()
    long_description=('Die oberen zwei Grafiken zeigen die Entwicklung der Tarife bezüglich: {selected_variable}. Im dritten Grafik ist die Anzahl der Suchanfragenergebnisse visualisiert.').format(selected_variable=selected_variable)


with chart_header_middle:
    st.write('  ')
    st.write('<div style="text-align: center"><b>'+short_description+'</b></div>', unsafe_allow_html=True)

empty_chartHeader_left1,  chart_header_middle1,   empty_chart_header_right1 = st.columns([1,  10,  1])
main_chart_container = chart_header_middle1.container()
        
main_chart_container.write(('Die oberen zwei Grafiken zeigen die Entwicklung der Tarife bezüglich: {selected_variable}. Im dritten Grafik ist die Anzahl der Suchanfragenergebnisse visualisiert.').format(selected_variable=selected_variable))

ohne_laufzeit_1300, ohne_laufzeit_9000, ohne_laufzeit_3000, ohne_laufzeit_15000 = None, None, None, None
mit_laufzeit_1300, mit_laufzeit_9000, mit_laufzeit_3000, mit_laufzeit_15000 = None, None, None, None

empty_left, e_chart, middle, g_chart, empty_right = st.columns([1, 4,1,4,1])

dates = electricity_results_3000[(electricity_results_3000.date >= pd.to_datetime(date_interval[0])) & (electricity_results_3000.date <= pd.to_datetime(date_interval[1]))].date.unique()

if(len(date_interval) == 2):
    print(top_n,'  ',type(top_n))
    e_median_date= electricity_results_3000[(electricity_results_3000.date == pd.to_datetime(date_interval[0])) & (electricity_results_3000.date <= pd.to_datetime(date_interval[1]))].date.median()
    chart_header = "**{energy_selection}verträge ({selected_variable})**".format(selected_variable=selected_variable, energy_selection='Strom')
    
    ohne_laufzeit_3000, mit_laufzeit_3000, ohne_laufzeit_3000_all, mit_laufzeit_3000_all, summary_3000 = summarize(electricity_results_3000, seperation_var, int(selection_slider),'3000',selected_variable, top_n=top_n)
    
    ohne_laufzeit_1300, mit_laufzeit_1300, ohne_laufzeit_1300_all, mit_laufzeit_1300_all, summary_1300 = summarize(electricity_results_1300, seperation_var, int(selection_slider),'1300', selected_variable, top_n=top_n) 
    
           
    summary = pd.concat([summary_3000, summary_1300])
    e_chart.write(chart_header)
    energy_line_chart_e, rng_e, dom_e = create_chart(summary, mean_median_btn, int(selection_slider), date_interval=date_interval, selected_variable=selected_variable, events_df=selected_events,energy_type='electricity', seperation_var=seperation_var)
    e_chart.altair_chart(energy_line_chart_e, use_container_width=True)

    
    g_median_date = gas_results_15000[(gas_results_15000.date >= pd.to_datetime(date_interval[0])) & (gas_results_15000.date <= pd.to_datetime(date_interval[1]))].date.median()
    chart_header = "**{energy_selection}verträge ({selected_variable})**".format(selected_variable=selected_variable, energy_selection='Gas')
    ohne_laufzeit_9000, mit_laufzeit_9000, ohne_laufzeit_9000_all, mit_laufzeit_9000_all,  summary_9000 = summarize(gas_results_9000, seperation_var,int(selection_slider),'9000',selected_variable, top_n=top_n)
    
    ohne_laufzeit_15000, mit_laufzeit_15000, ohne_laufzeit_15000_all, mit_laufzeit_15000_all, summary_15000 = summarize(gas_results_15000, seperation_var,int(selection_slider),'15000',selected_variable, top_n=top_n)
    
        
    summary = pd.concat([summary_9000, summary_15000])
    g_chart.write(chart_header)
    energy_line_chart_e, rng_g, dom_g = create_chart(summary, mean_median_btn, int(selection_slider), date_interval=date_interval, selected_variable=selected_variable, events_df=selected_events,energy_type='gas', seperation_var=seperation_var)
    g_chart.altair_chart(energy_line_chart_e, use_container_width=True)

## ENDE CHART REGION

empty_column_left, empty_left_colum1, tarif_list_menu_column_previous, tarif_list_menu_current, tarif_list_menu_next, empty_right_column1, empty_column_right  = st.columns([1, 3.7, 0.8, 1, 0.8, 3.7, 1]) 
electricity_tarif_list_column, gas_tarif_listchart_column = st.columns(2) 

tariff_list_empty_left, electricity_tarif_list_column, tariff_list_middle, gas_tarif_listchart_column, tariff_list_empty_right = st.columns([1, 4,1,4,1])

with empty_left_colum1:
    st.write('   ')
    st.markdown("""---""")
with empty_right_column1:
    st.write('   ')
    st.markdown("""---""")
    
dates = electricity_results_3000[(electricity_results_3000.date >= pd.to_datetime(date_interval[0])) & (electricity_results_3000.date <= pd.to_datetime(date_interval[1]))].date.unique()
dates = pd.to_datetime(dates).strftime("%b %d, %Y")


with tarif_list_menu_current: 
    selected_date_e = st.selectbox(
                        '',
                        (dates),
                        index= (len(dates)+1)//2)

if(len(dates[np.where(np.asarray( dates)< selected_date_e)]) > 0):
    with tarif_list_menu_column_previous:
            st.write('   ')
            #st.write('   ')
            st.write('   ')
            prev_date = dates[np.where(np.asarray( dates)< selected_date_e)][-1:][0]
            prev_date = pd.to_datetime(prev_date).strftime("%b %d, %Y")
            st.button(' {prev_date} << '.format(prev_date=prev_date), disabled=True)
            

if(len(dates[np.where(np.asarray( dates)> selected_date_e)]) > 0):
    with tarif_list_menu_next:
            st.write('   ')
            #st.write('   ')
            st.write('   ')
            next_date = dates[np.where(np.asarray( dates)> selected_date_e)][:1][0]
            next_date = pd.to_datetime(next_date).strftime("%b %d, %Y")
            st.button(' >> {next_date} '.format(next_date=next_date), disabled=True)

sep_line_empty_left2, sep_line2_center, sep_line_empty_right = st.columns([1, 10, 1])                   
sep_line2_center.markdown("""---""")

with electricity_tarif_list_column:
    tariff_list_expander_3000e = st.expander('Stromtarife mit 3000 kWh Verbrauch', expanded=True)
    
    with tariff_list_expander_3000e: 
        #st.write('Tarif mit teuerster Arbeitspreis: ')
        #st.write(electritcity_all.loc[electritcity_all['dataunit'] == electritcity_all['dataunit'].max()][['providerName', 'tariffName', 'Jahreskosten', 'dataunit','datafixed', 'dataeco' ]])

        #st.write('Tarif mit teuerster Jahreskosten: ')
        #st.write(electritcity_all.loc[electritcity_all['Jahreskosten'] == electritcity_all['Jahreskosten'].max()][['providerName', 'tariffName', 'Jahreskosten', 'dataunit','datafixed', 'dataeco' ]])

        #st.write('Tarif mit teuerste Grundpreis: ')
        #st.write(electritcity_all.loc[electritcity_all['datafixed'] == electritcity_all['datafixed'].max()][['providerName', 'tariffName', 'Jahreskosten', 'dataunit','datafixed', 'dataeco' ]])
        st.write('<div style="text-align: center"><b>'+'Zusammenfassung aller Tarife mit 3K Verbrauch am '+str(selected_date_e)+'</b></div>', unsafe_allow_html=True)
        st.write(' ')
        st.write('<div style="text-align: center"><b>'+'Histogramm'+'</b></div>', unsafe_allow_html=True)
        st.write(' ')
        if(seperation_var != 'Kein Unterscheidungsmerkmal'):
            ohne_laufzeit_3000_all['Beschreibung'] = dom_e[0]
            mit_laufzeit_3000_all['Beschreibung'] = dom_e[1]

            electritcity_all = pd.concat([ohne_laufzeit_3000_all,  mit_laufzeit_3000_all])

            density_chart_e = create_desitiy_chart(electritcity_all, selected_date_e, selected_variable, rng_e, dom_e)
            st.altair_chart(density_chart_e)
        

        if( (seperation_var == 'Preisgarantie') | (seperation_var == 'Vertragslaufzeit') ):
            st.write('<div style="text-align: center"><b>'+'Top {top_n} Tarife am {selected_date}'.format(top_n=top_n, selected_date=selected_date_e)+'</b></div>', unsafe_allow_html=True)
            st.write('<div  style="background-color:'+rng_e[0]+'; text-align: center; width:auto;">{seperation_var} < {selection_slider}</div>'.format(top_n=top_n, selected_date=selected_date_e, seperation_var=seperation_var, selection_slider=selection_slider), unsafe_allow_html=True)
            
            grid_table, grid_table_df = get_table(ohne_laufzeit_3000,selected_date_e, rng_e[0], dom_e[0])
            sel_row = grid_table['selected_rows']

            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(electritcity_all, grid_table_df, index, selected_variable)

            st.write('<div style="background-color:'+rng_e[1]+';  text-align: center">{seperation_var} >= {selection_slider}:'.format(top_n=top_n, selected_date=selected_date_e, seperation_var=seperation_var, selection_slider=selection_slider), unsafe_allow_html=True)
            grid_table, grid_table_df = get_table(mit_laufzeit_3000,selected_date_e, rng_e[1], dom_e[1])
            sel_row = grid_table['selected_rows']

            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(electritcity_all, grid_table_df, index, selected_variable)


        elif(seperation_var =='Öko Tarif/ Konventioneller Tarif'):
            st.write('<div style="text-align: center"><b>'+'Top {top_n} Tarife am {selected_date}'.format(top_n=top_n, selected_date=selected_date_e)+'</b></div>', unsafe_allow_html=True)
            st.write('<div style="background-color:'+rng_e[1]+'; text-align: center">Nicht-Öko Tarife'.format(top_n=top_n, selected_date=selected_date_e, seperation_var=seperation_var, selection_slider=selection_slider), unsafe_allow_html=True)
            grid_table, grid_table_df = get_table(ohne_laufzeit_3000,selected_date_e, rng_e[1], dom_e[1])
            sel_row = grid_table['selected_rows']

            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(electritcity_all, grid_table_df, index, selected_variable)

            st.write('<div style="background-color:'+rng_e[0]+'; text-align: center">Öko Tarife'.format(top_n=top_n, selected_date=selected_date_e, seperation_var=seperation_var, selection_slider=selection_slider), unsafe_allow_html=True)
            grid_table, grid_table_df = get_table(mit_laufzeit_3000,selected_date_e, rng_e[0], dom_e[0])
            sel_row = grid_table['selected_rows']

            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(electritcity_all, grid_table_df, index, selected_variable)

        elif(seperation_var =='Partner'):
            st.write('<div style="text-align: center"><b>'+'Top {top_n} Tarife am {selected_date}'.format(top_n=top_n, selected_date=selected_date_e)+'</b></div>', unsafe_allow_html=True)
            st.write('<div style="background-color:'+rng_e[1]+'; text-align: center">Von Verivox'.format(top_n=top_n, selected_date=selected_date_e, seperation_var=seperation_var, selection_slider=selection_slider), unsafe_allow_html=True)
            grid_table, grid_table_df = get_table(ohne_laufzeit_3000,selected_date_e, rng_e[1], dom_e[1])
            sel_row = grid_table['selected_rows']

            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(electritcity_all, grid_table_df, index, selected_variable)

            st.write('<div style="background-color:'+rng_e[0]+'; text-align: center">Von Check24'.format(top_n=top_n, selected_date=selected_date_e, seperation_var=seperation_var, selection_slider=selection_slider), unsafe_allow_html=True)
            grid_table, grid_table_df = get_table(mit_laufzeit_3000,selected_date_e, rng_e[0], dom_e[0])
            sel_row = grid_table['selected_rows']

            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(electritcity_all, grid_table_df, index, selected_variable)

        elif(seperation_var == 'Kein Unterscheidungsmerkmal'):

            mit_laufzeit_3000_all['Beschreibung'] = dom_e[0]

            density_chart_e = create_desitiy_chart(mit_laufzeit_3000_all, selected_date_e, selected_variable, rng_e, dom_e)

            st.altair_chart(density_chart_e)
            st.write('<div style="text-align: center"><b>'+'Top {top_n} Tarife am {selected_date}'.format(top_n=top_n, selected_date=selected_date_e)+'</b></div>', unsafe_allow_html=True)
            st.write('<div style="background-color:'+rng_e[0]+'; text-align: center">Mit 3K kWh Verbrauch', unsafe_allow_html=True)
            grid_table, grid_table_df = get_table(mit_laufzeit_3000,selected_date_e, rng_e[0], dom_e[0])
            sel_row = grid_table['selected_rows']

            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(mit_laufzeit_3000_all, grid_table_df, index, selected_variable)

    tariff_list_expander_1300e = st.expander('Stromtarife mit 1300 kWh Verbrauch', expanded=False)
    
    with tariff_list_expander_1300e:

        st.write('<div style="text-align: center"><b>'+'Zusammenfassung aller Tarife mit 3K Verbrauch am '+str(selected_date_e)+'</b></div>', unsafe_allow_html=True)
        st.write(' ')
        st.write('<div style="text-align: center"><b>'+'Histogramm'+'</b></div>', unsafe_allow_html=True)
        st.write(' ')

        if(seperation_var != 'Kein Unterscheidungsmerkmal'):
            ohne_laufzeit_1300_all['Beschreibung'] = dom_e[2]
            mit_laufzeit_1300_all['Beschreibung'] = dom_e[3]

            electritcity_all = pd.concat([ohne_laufzeit_1300_all,  mit_laufzeit_1300_all])

            density_chart_e = create_desitiy_chart(electritcity_all, selected_date_e, selected_variable, rng_e, dom_e)

            st.altair_chart(density_chart_e)

        if( (seperation_var == 'Preisgarantie') | (seperation_var == 'Vertragslaufzeit') ):
            st.write('<div style="text-align: center"><b>'+'Top {top_n} Tarife am {selected_date}'.format(top_n=top_n, selected_date=selected_date_e)+'</b></div>', unsafe_allow_html=True)
            st.write('<div style="background-color:'+rng_e[2]+'; text-align: center">{seperation_var} < {selection_slider}'.format(top_n=top_n, selected_date=selected_date_e, seperation_var=seperation_var, selection_slider=selection_slider), unsafe_allow_html=True)
            grid_table, grid_table_df = get_table(ohne_laufzeit_1300,selected_date_e, rng_e[2], dom_e[2])
            sel_row = grid_table['selected_rows']

            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(electritcity_all, grid_table_df, index, selected_variable)

            st.write('<div style="background-color:'+rng_e[3]+'; text-align: center">{seperation_var} >= {selection_slider}'.format(top_n=top_n, selected_date=selected_date_e, seperation_var=seperation_var, selection_slider=selection_slider), unsafe_allow_html=True)
            grid_table, grid_table_df = get_table(mit_laufzeit_1300,selected_date_e, rng_e[3], dom_e[3])
            sel_row = grid_table['selected_rows']

            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(electritcity_all, grid_table_df, index, selected_variable)

        elif(seperation_var =='Öko Tarif/ Konventioneller Tarif'):

            st.write('<div style="text-align: center"><b>'+'Top {top_n} Tarife am {selected_date}'.format(top_n=top_n, selected_date=selected_date_e)+'</b></div>', unsafe_allow_html=True)
            st.write('<div style="background-color:'+rng_e[3]+'; text-align: center">Nicht-Öko Tarife'.format(top_n=top_n, selected_date=selected_date_e, seperation_var=seperation_var, selection_slider=selection_slider), unsafe_allow_html=True)
            grid_table, grid_table_df = get_table(ohne_laufzeit_1300,selected_date_e, rng_e[3], dom_e[3])
            sel_row = grid_table['selected_rows']

            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(electritcity_all, grid_table_df, index, selected_variable)

            st.write('<div style="background-color:'+rng_e[2]+'; text-align: center">Öko Tarife'.format(top_n=top_n, selected_date=selected_date_e, seperation_var=seperation_var, selection_slider=selection_slider), unsafe_allow_html=True)
            grid_table, grid_table_df = get_table(mit_laufzeit_1300,selected_date_e, rng_e[2], dom_e[2])
            sel_row = grid_table['selected_rows']

            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(electritcity_all, grid_table_df, index, selected_variable)

        elif(seperation_var =='Partner'):
            st.write('<div style="text-align: center"><b>'+'Top {top_n} Tarife am {selected_date}'.format(top_n=top_n, selected_date=selected_date_e)+'</b></div>', unsafe_allow_html=True)
            st.write('<div style="background-color:'+rng_e[3]+'; text-align: center">Von Verivox', unsafe_allow_html=True)
            grid_table, grid_table_df = get_table(ohne_laufzeit_1300,selected_date_e, rng_e[3], dom_e[3])
            sel_row = grid_table['selected_rows']

            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(electritcity_all, grid_table_df, index, selected_variable)

            st.write('<div style="background-color:'+rng_e[2]+'; text-align: center">Von Check24', unsafe_allow_html=True)
            grid_table, grid_table_df = get_table(mit_laufzeit_1300,selected_date_e, rng_e[2], dom_e[2])
            sel_row = grid_table['selected_rows']

            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(electritcity_all, grid_table_df, index, selected_variable)

        elif(seperation_var == 'Kein Unterscheidungsmerkmal'):
            mit_laufzeit_1300_all['Beschreibung'] = dom_e[1]

            density_chart_e = create_desitiy_chart(mit_laufzeit_1300_all, selected_date_e, selected_variable, rng_e, dom_e)

            st.altair_chart(density_chart_e)

            st.write('<div style="text-align: center"><b>'+'Top {top_n} Tarife am {selected_date}'.format(top_n=top_n, selected_date=selected_date_e)+'</b></div>', unsafe_allow_html=True)
            st.write('<div style="background-color:'+rng_e[1]+'; text-align: center">Mit 1.3K kWh Verbrauch', unsafe_allow_html=True)
            grid_table, grid_table_df = get_table(mit_laufzeit_1300,selected_date_e, rng_e[1], dom_e[1])
            sel_row = grid_table['selected_rows']

            #st.write(sel_row)
            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(mit_laufzeit_1300_all, grid_table_df, index, selected_variable)

with gas_tarif_listchart_column:
    tariff_list_expander_15000g = st.expander('Tarifliste - Gasarife mit 1500 kWh Verbrauch', expanded=True)
    
    with tariff_list_expander_15000g:

        st.write('<div style="text-align: center"><b>'+'Zusammenfassung aller Tarife mit 15K kWh Verbrauch am '+str(selected_date_e)+'</b></div>', unsafe_allow_html=True)
        st.write('<div style="text-align: center"><b>'+'Histogramm'+'</b></div>', unsafe_allow_html=True)
        st.write(' ')

        if(seperation_var != 'Kein Unterscheidungsmerkmal'):
            ohne_laufzeit_15000_all['Beschreibung'] = dom_g[0]
            mit_laufzeit_15000_all['Beschreibung'] = dom_g[1]

            gas_all = pd.concat([ohne_laufzeit_15000_all,  mit_laufzeit_15000_all])
            density_chart_g = create_desitiy_chart(gas_all, selected_date_e, selected_variable, rng_g, dom_g)
            st.altair_chart(density_chart_g)

        if( (seperation_var == 'Preisgarantie') | (seperation_var == 'Vertragslaufzeit') ):
            #st.info('Hier ist gedacht die Tarife aufzulisten die oben im Barchart ausgewählt sind')
            st.write('<div style="text-align: center"><b>'+'Top {top_n} Tarife am {selected_date}'.format(top_n=top_n, selected_date=selected_date_e)+'</b></div>', unsafe_allow_html=True)
            st.write('<div style="background-color:'+rng_g[0]+'; text-align: center">{seperation_var} < {selection_slider}'.format(top_n=top_n, selected_date=selected_date_e, seperation_var=seperation_var, selection_slider=selection_slider), unsafe_allow_html=True)
            grid_table, grid_table_df = get_table(ohne_laufzeit_15000,selected_date_e, rng_g[0], dom_g[0])
            sel_row = grid_table['selected_rows']

            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(gas_all, grid_table_df, index, selected_variable)

            st.write('<div style="background-color:'+rng_g[1]+'; text-align: center">{seperation_var} >= {selection_slider}'.format(top_n=top_n, selected_date=selected_date_e, seperation_var=seperation_var, selection_slider=selection_slider), unsafe_allow_html=True)
            grid_table, grid_table_df = get_table(mit_laufzeit_15000,selected_date_e, rng_g[1], dom_g[1])
            sel_row = grid_table['selected_rows']

            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(gas_all, grid_table_df, index, selected_variable)
        elif(seperation_var =='Öko Tarif/ Konventioneller Tarif'):
            st.write('<div style="text-align: center"><b>'+'Top {top_n} Tarife am {selected_date}'.format(top_n=top_n, selected_date=selected_date_e)+'</b></div>', unsafe_allow_html=True)
            st.write('<div style="background-color:'+rng_g[1]+'; text-align: center">Nicht-Öko Tarife', unsafe_allow_html=True)
            grid_table, grid_table_df = get_table(ohne_laufzeit_15000,selected_date_e, rng_g[1], dom_g[1])
            sel_row = grid_table['selected_rows']

            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(gas_all, grid_table_df, index, selected_variable)

            st.write('<div style="background-color:'+rng_g[0]+'; text-align: center">Öko Tarife', unsafe_allow_html=True)
            grid_table, grid_table_df = get_table(mit_laufzeit_15000,selected_date_e, rng_g[0], dom_g[0])
            sel_row = grid_table['selected_rows']

            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(gas_all, grid_table_df, index, selected_variable)

        elif(seperation_var =='Partner'):
            st.write('<div style="text-align: center"><b>'+'Top {top_n} Tarife am {selected_date}'.format(top_n=top_n, selected_date=selected_date_e)+'</b></div>', unsafe_allow_html=True)
            st.write('<div style="background-color:'+rng_g[1]+'; text-align: center">Von Verivox', unsafe_allow_html=True)
            grid_table, grid_table_df = get_table(ohne_laufzeit_15000,selected_date_e, rng_g[1], dom_g[1])
            sel_row = grid_table['selected_rows']

            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(gas_all, grid_table_df, index, selected_variable)

            st.write('<div style="background-color:'+rng_g[0]+'; text-align: center">Von Check24', unsafe_allow_html=True)
            grid_table, grid_table_df = get_table(mit_laufzeit_15000,selected_date_e, rng_g[0], dom_g[0])
            sel_row = grid_table['selected_rows']

            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(gas_all, grid_table_df, index, selected_variable)
                
        elif(seperation_var == 'Kein Unterscheidungsmerkmal'):

            mit_laufzeit_15000_all['Beschreibung'] = dom_g[0]

            density_chart_e = create_desitiy_chart(mit_laufzeit_15000_all, selected_date_e, selected_variable, rng_g, dom_g)

            st.altair_chart(density_chart_e)
            st.write('<div style="text-align: center"><b>'+'Top {top_n} Tarife am {selected_date}'.format(top_n=top_n, selected_date=selected_date_e)+'</b></div>', unsafe_allow_html=True)
            st.write('<div style="background-color:'+rng_g[0]+'; text-align: center">Mit 15K kWh Verbrauch', unsafe_allow_html=True)
            grid_table, grid_table_df = get_table(mit_laufzeit_15000,selected_date_e, rng_g[0], dom_g[0])
            sel_row = grid_table['selected_rows']

            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(mit_laufzeit_15000_all, grid_table_df, index, selected_variable)

    tariff_list_expander_9000g = st.expander('Tarifliste - Gastarife mit 9000 kWh Verbrauch', expanded=False)
    
    with tariff_list_expander_9000g:


        st.write('<div style="text-align: center"><b>'+'Zusammenfassung aller Tarife mit 9K kWh Verbrauch am '+str(selected_date_e)+'</b></div>', unsafe_allow_html=True)
        st.write('<div style="text-align: center"><b>'+'Histogramm'+'</b></div>', unsafe_allow_html=True)
        st.write(' ')

        if(seperation_var != 'Kein Unterscheidungsmerkmal'):
            ohne_laufzeit_9000_all['Beschreibung'] = dom_g[2]
            mit_laufzeit_9000_all['Beschreibung'] = dom_g[3]

            gas_all = pd.concat([ohne_laufzeit_9000_all,  mit_laufzeit_9000_all])

            density_chart_g = create_desitiy_chart(gas_all, selected_date_e, selected_variable, rng_g, dom_g)

            st.altair_chart(density_chart_g)

        if( (seperation_var == 'Preisgarantie') | (seperation_var == 'Vertragslaufzeit') ):
            #st.info('Hier ist gedacht die Tarife aufzulisten die oben im Barchart ausgewählt sind')
            st.write('<div style="text-align: center"><b>'+'Top {top_n} Tarife am {selected_date}'.format(top_n=top_n, selected_date=selected_date_e)+'</b></div>', unsafe_allow_html=True)
            st.write('<div style="background-color:'+rng_g[2]+'; text-align: center">{seperation_var} < {selection_slider}'.format(top_n=top_n, selected_date=selected_date_e, seperation_var=seperation_var, selection_slider=selection_slider), unsafe_allow_html=True)
            grid_table, grid_table_df = get_table(ohne_laufzeit_9000,selected_date_e, rng_g[2], dom_g[2])
            sel_row = grid_table['selected_rows']

            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(gas_all, grid_table_df, index, selected_variable)

            st.write('<div style="background-color:'+rng_g[3]+'; text-align: center">{seperation_var} >= {selection_slider}'.format(top_n=top_n, selected_date=selected_date_e, seperation_var=seperation_var, selection_slider=selection_slider), unsafe_allow_html=True)
            grid_table, grid_table_df = get_table(mit_laufzeit_9000,selected_date_e, rng_g[3], dom_g[3])
            sel_row = grid_table['selected_rows']

            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(gas_all, grid_table_df, index, selected_variable)

        elif(seperation_var =='Öko Tarif/ Konventioneller Tarif'):
            st.write('<div style="text-align: center"><b>'+'Top {top_n} Tarife am {selected_date}'.format(top_n=top_n, selected_date=selected_date_e)+'</b></div>', unsafe_allow_html=True)
            st.write('<div style="background-color:'+rng_g[3]+'; text-align: center">Nicht-Öko Tarife', unsafe_allow_html=True)
            grid_table, grid_table_df = get_table(ohne_laufzeit_9000,selected_date_e, rng_g[3], dom_g[3])
            sel_row = grid_table['selected_rows']

            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(gas_all, grid_table_df, index, selected_variable)
                
            st.write('<div style="background-color:'+rng_g[2]+'; text-align: center">Öko Tarife', unsafe_allow_html=True)
            grid_table, grid_table_df = get_table(mit_laufzeit_9000,selected_date_e, rng_g[2], dom_g[2])
            sel_row = grid_table['selected_rows']

            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(gas_all, grid_table_df, index, selected_variable)

        elif(seperation_var =='Partner'):
            st.write('<div style="text-align: center"><b>'+'Top {top_n} Tarife am {selected_date}'.format(top_n=top_n, selected_date=selected_date_e)+'</b></div>', unsafe_allow_html=True)
            st.write('<div style="background-color:'+rng_g[3]+'; text-align: center">Von Verivox', unsafe_allow_html=True)
            grid_table, grid_table_df = get_table(ohne_laufzeit_9000,selected_date_e, rng_g[3], dom_g[3])
            sel_row = grid_table['selected_rows']

            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(gas_all, grid_table_df, index, selected_variable)

            st.write('<div style="background-color:'+rng_g[2]+'; text-align: center">Von Check24', unsafe_allow_html=True)
            grid_table, grid_table_df = get_table(mit_laufzeit_9000,selected_date_e, rng_g[2], dom_g[2])
            sel_row = grid_table['selected_rows']

            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(gas_all, grid_table_df, index, selected_variable)
        elif(seperation_var == 'Kein Unterscheidungsmerkmal'):

            mit_laufzeit_9000_all['Beschreibung'] = dom_g[1]

            density_chart_e = create_desitiy_chart(mit_laufzeit_9000_all, selected_date_e, selected_variable, rng_g, dom_g)

            st.altair_chart(density_chart_e)
            st.write('<div style="text-align: center"><b>'+'Top {top_n} Tarife am {selected_date}'.format(top_n=top_n, selected_date=selected_date_e)+'</b></div>', unsafe_allow_html=True)
            st.write('<div style="background-color:'+rng_g[1]+'; text-align: center">Mit 9K kWh Verbrauch', unsafe_allow_html=True)
            grid_table, grid_table_df = get_table(mit_laufzeit_9000,selected_date_e, rng_g[1], dom_g[1])
            sel_row = grid_table['selected_rows']

            if(len(sel_row) > 0 ):
                index = sel_row[0]['_selectedRowNodeInfo']['nodeRowIndex']
                create_tarif_summary_section(mit_laufzeit_9000_all, grid_table_df, index, selected_variable)



#empty_tariffSummary_Header_left1,  tariffSummary_chart_header_middle1,   empty_tariffSummary_header_right1 = st.columns([1,  10,  1])
#empty_tariffSummary_table_left1,  tariffSummary_chart_table_middle1,   empty_tariffSummary_table_right1 = st.columns([1,  10,  1])

#tariffSummary_chart_header_middle1.write('<div style="text-align: center"><b>Übersicht über aller Tarife - Vergleich Check24 und Verivox</b></div>', unsafe_allow_html=True)
#all_tafiffs_100 = electricity_results_100_3000[1]
#all_tafiffs_100 = all_tafiffs_100[all_tafiffs_100['date'] == selected_date_e]

#grid_table = get_tariff_table(all_tafiffs_100, selected_date_e)

#get_tariff_table(electricity_results_100_3000[1], selected_date_e)


#Ungewöhnlich hohe Grundkosten +3x Std gelb. Über 1000€ oder unter 0 rot
#dataunit unter 0 rot. Über +3x std gelb

#Vertragsverlängerung über 1.5 oder unter 0 rot



#tariffSummary_chart_table_middle1.write(grid_table)

##########################################################


#javascript integriegen um screen weite zu lesen:
#https://www.youtube.com/watch?v=TqOGBOHHxrU


#print(high_consume.dtypes)
#tariff_summary, boxplot = summarize_tariffs(electricity_results_3000)
#st.write(tariff_summary)

#main_chart_container.altair_chart(boxplot)

#gasimportdaten
#https://www.bundesnetzagentur.de/DE/Fachthemen/ElektrizitaetundGas/Versorgungssicherheit/aktuelle_gasversorgung_/_svg/Gasimporte/Gasimporte.html
#wetter und verbrauch daten
#https://www.bundesnetzagentur.de/DE/Gasversorgung/aktuelle_gasversorgung/_svg/GasverbrauchSLP_monatlich/Gasverbrauch_SLP_M.html;jsessionid=BC4D6020F61B843F1C0FB52C4384DE6E
#<div tabindex="0" role="button" aria-expanded="true" class="streamlit-expanderHeader st-ae st-bw st-ag st-ah st-ai st-aj st-bx st-by st-bz st-c0 st-c1 st-c2 st-c3 st-ar st-as st-c4 st-c5 st-b3 st-c6 st-c7 st-c8 st-b4 st-c9 st-ca st-cb st-cc st-cd">Tarifliste - Gasarife mit 1500 kWh Verbrauch<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false" fill="currentColor" xmlns="http://www.w3.org/2000/svg" color="inherit" class="e1fb0mya1 css-fblp2m ex0cdmw0"><path fill="none" d="M0 0h24v24H0V0z"></path><path d="M12 8l-6 6 1.41 1.41L12 10.83l4.59 4.58L18 14l-6-6z"></path></svg></div>
#<div tabindex="0" role="button" aria-expanded="true" class="streamlit-expanderHeader st-ae st-bw st-ag st-ah st-ai st-aj st-bx st-by st-bz st-c0 st-c1 st-c2 st-c3 st-ar st-as st-c4 st-c5 st-b3 st-c6 st-c7 st-c8 st-b4 st-c9 st-ca st-cb st-cc st-cd">Tarifliste - Stromtarife mit 3000 kWh Verbrauch<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false" fill="currentColor" xmlns="http://www.w3.org/2000/svg" color="inherit" class="e1fb0mya1 css-fblp2m ex0cdmw0"><path fill="none" d="M0 0h24v24H0V0z"></path><path d="M12 8l-6 6 1.41 1.41L12 10.83l4.59 4.58L18 14l-6-6z"></path></svg></div>