import numpy as np
import pandas as pd
import os
import re
import datetime
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
        return value_with_current_unit
    elif(currentunit == 'year'):
        return value_with_current_unit *12
    if(currentunit == 'week'):
        return value_with_current_unit *0.25
    if(currentunit == 'day'):
        return value_with_current_unit / 30
    if( (( currentunit == 'nan' or currentunit == 'once' or currentunit == 'indefinitely') & value_with_current_unit == 0)):
        return value_with_current_unit
    else:
        return int(-1)

def set_plz(ID):
  if(ID==3):
    return '10245'
  elif(ID==7):
    return '99425'
  elif(ID==11):
    return '33100'
  elif(ID==15):
    return '50670'
  elif(ID==19):
    return '71771'

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

    #path = Path(__file__).parents[1] / 'data/wa_{energy_type}.xlsx'
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

#electricity_loader_thread = threading.Thread(target=read_energy_data, args = ['electricity'], name='electricity_loader_thread')
#gas_loader_thread = threading.Thread(target=read_energy_data, args = ('gas'), name='gas_loader_thread')
#high_consume = electricity_loader_thread.start()

with concurrent.futures.ThreadPoolExecutor() as executor:
    electricity_reader_thread_3000 = executor.submit(read_energy_data, 'electricity', '3000')
    electricity_reader_thread_1300 = executor.submit(read_energy_data, 'electricity', '1300')
    gas_reader_thread_15000 = executor.submit(read_energy_data, 'gas', '15000')
    gas_reader_thread_9000 = executor.submit(read_energy_data, 'gas', '9000')
    electricity_results_3000 = electricity_reader_thread_3000.result()
    electricity_results_1300 = electricity_reader_thread_1300.result()
    gas_results_15000 = gas_reader_thread_15000.result()
    gas_results_9000 = gas_reader_thread_9000.result()

#gas_loader_thread.start()
#electricity_loader_thread.join()
#high_consume_gas = gas_loader_thread.join()

#high_consume = read_energy_data('gas')

#@st.cache(ttl=24*60*60)
def summarize(results, seperation_var='priceGuaranteeNormalized',seperation_value=12, consumption='unknown',selected_variable='dataunit', top_n = 10):

    sep_var_readable = seperation_var
    if(seperation_var == 'Vertragslaufzeit'):
        seperation_var = 'contractDurationNormalized'
    elif(seperation_var == 'Preisgarantie'):
        seperation_var='priceGuaranteeNormalized'
    elif(seperation_var == 'Öko Tarif/ Konventioneller Tarif'):
        seperation_var = 'dataeco'
    elif(seperation_var == 'Partner'):
        seperation_var = 'signupPartner'
    
    variables_dict = {
        "Arbeitspreis": "dataunit",
        "Grundpreis": "datafixed",
        "Jahreskosten": "Jahreskosten"
        }

    agg_functions = {
        variables_dict[selected_variable]:
        [ 'mean', 'median','std', 'min', 'max', 'count']
    }
    

    if( (seperation_var == 'contractDurationNormalized') | (seperation_var == 'priceGuaranteeNormalized') ):


        ohne_laufzeit  = results[results[seperation_var] < seperation_value]

        ohne_laufzeit = ohne_laufzeit.groupby(['date','plz']).apply(
        lambda df: df.nsmallest(10, columns='dataunit')
        ).reset_index(drop=True)

        summary_ohne_laufzeit = ohne_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
        summary_ohne_laufzeit.columns =  [ 'mean', 'median','std', 'min', 'max', 'count']
        summary_ohne_laufzeit['date'] = summary_ohne_laufzeit.index
        summary_ohne_laufzeit['beschreibung'] = 'Verbrauch: '+consumption+'\n'+sep_var_readable+' < '+str(seperation_value) 
        
        #mit_laufzeit  = high_consume[high_consume['contractDurationNormalized'] > 11]
        mit_laufzeit  = results[results[seperation_var] >= seperation_value]
        mit_laufzeit = mit_laufzeit.groupby(['date','plz']).apply(
        lambda df: df.nsmallest(10, columns='dataunit')
        ).reset_index(drop=True)

        summary_mit_laufzeit = mit_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
        summary_mit_laufzeit.columns =  [ 'mean', 'median','std', 'min', 'max', 'count']
        summary_mit_laufzeit['date'] = summary_mit_laufzeit.index
        summary_mit_laufzeit['beschreibung'] = 'Verbrauch: '+consumption+'\n'+sep_var_readable+' >= '+str(seperation_value)

        summary = pd.concat([summary_mit_laufzeit, summary_ohne_laufzeit])
    elif(seperation_var == 'dataeco'):
        ohne_laufzeit  = results[results[seperation_var] == True]

        ohne_laufzeit = ohne_laufzeit.groupby(['date','plz']).apply(
        lambda df: df.nsmallest(10, columns='dataunit')
        ).reset_index(drop=True)

        
        summary_ohne_laufzeit = ohne_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
        summary_ohne_laufzeit.columns =  [ 'mean', 'median','std', 'min', 'max', 'count']
        summary_ohne_laufzeit['date'] = summary_ohne_laufzeit.index
        summary_ohne_laufzeit['beschreibung'] = 'Verbrauch: '+consumption+' und Öko Tarif' 
        
        #mit_laufzeit  = high_consume[high_consume['contractDurationNormalized'] > 11]
        mit_laufzeit  = results[results[seperation_var] ==  False]

        mit_laufzeit  = results[results[seperation_var] >= seperation_value]
        mit_laufzeit = mit_laufzeit.groupby(['date','plz']).apply(
        lambda df: df.nsmallest(10, columns='dataunit')
        ).reset_index(drop=True)

        summary_mit_laufzeit = mit_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
        summary_mit_laufzeit.columns =  [ 'mean', 'median','std', 'min', 'max', 'count']
        summary_mit_laufzeit['date'] = summary_mit_laufzeit.index
        summary_mit_laufzeit['beschreibung'] = 'Verbrauch: '+consumption+' und Nicht-Öko Tarif'
        summary = pd.concat([summary_mit_laufzeit, summary_ohne_laufzeit])
    elif(seperation_var == 'signupPartner'):
        ohne_laufzeit  = results[results[seperation_var] == 'vx']

        ohne_laufzeit = ohne_laufzeit.groupby(['date','plz']).apply(
        lambda df: df.nsmallest(10, columns='dataunit')
        ).reset_index(drop=True)

        summary_ohne_laufzeit = ohne_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
        summary_ohne_laufzeit.columns =  [ 'mean', 'median','std', 'min', 'max', 'count']
        summary_ohne_laufzeit['date'] = summary_ohne_laufzeit.index
        summary_ohne_laufzeit['beschreibung'] = 'Verbrauch: '+consumption+' Verivox' 
        
        #mit_laufzeit  = high_consume[high_consume['contractDurationNormalized'] > 11]
        mit_laufzeit  = results[results[seperation_var] ==  'c24']

        mit_laufzeit  = results[results[seperation_var] >= seperation_value]
        mit_laufzeit = mit_laufzeit.groupby(['date','plz']).apply(
        lambda df: df.nsmallest(10, columns='dataunit')
        ).reset_index(drop=True)
        
        summary_mit_laufzeit = mit_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
        summary_mit_laufzeit.columns =  [ 'mean', 'median','std', 'min', 'max', 'count']
        summary_mit_laufzeit['date'] = summary_mit_laufzeit.index
        summary_mit_laufzeit['beschreibung'] = 'Verbrauch: '+consumption+' Check24'

        summary = pd.concat([summary_mit_laufzeit, summary_ohne_laufzeit])

    return summary

#@st.cache(ttl=24*60*60)
def create_chart(summary,  aggregation='mean', seperation_value=12, date_interval=['2022-07-17', '2022-10-17'], widtht=700, height=280,selected_variable='dataunit', events_df=None, energy_type='gas', seperation_var='priceGuaranteeNormalized'):

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
    chart_max = summary[(summary.date >= pd.to_datetime(date_interval[0])) & (summary.date <= pd.to_datetime(date_interval[1])) ]['count'].max()
    
    chart_max = np.ceil( chart_max + (1.05*chart_max))
    domain3 = np.linspace(0, chart_max, 2, endpoint = True)
    
    source = summary.copy()
    #, color='#4650DF'
    x_init = pd.to_datetime(date_interval).astype(int) / 1E6
    interval = alt.selection_interval(encodings=['x'],init = {'x':x_init.to_list()})
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

    print('SEP VAR: ',seperation_var,'   ',seperation_value)
    if((energy_type == 'gas') & (seperation_var != 'Öko Tarif/ Konventioneller Tarif') & (seperation_var != 'Partner')):
        rng = ['#4650DF','#FC6E44', '#006E78', '#20B679']
        dom = [source[source.beschreibung.str.contains('15000') & source.beschreibung.str.contains('<')].iloc[0].beschreibung,
              source[source.beschreibung.str.contains('15000') & source.beschreibung.str.contains('>=')].iloc[0].beschreibung,
              source[source.beschreibung.str.contains('9000') & source.beschreibung.str.contains('<')].iloc[0].beschreibung,
              source[source.beschreibung.str.contains('9000') & source.beschreibung.str.contains('>=')].iloc[0].beschreibung]
    elif((energy_type == 'electricity') & (seperation_var != 'Öko Tarif/ Konventioneller Tarif') & (seperation_var != 'Partner')):
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


    base = alt.Chart(source).mark_line(size=3).encode(
        #x= alt.X('date:T',axis= alt.Axis(grid=False, title='Datum')),
        y = alt.Y(aggregation+':Q', axis = alt.Axis(title=y_axis_title, offset= 5)),
        x= alt.X('date:T',axis= alt.Axis(grid=False, title='Datum 📅')),
        #y = alt.Y('median:Q', axis = alt.Axis(title='Arbeitspreis (ct/kWh)')),
        color=alt.Color('beschreibung:N', scale=alt.
                    Scale(domain=dom, range=rng))
       # strokeDash=alt.condition(
       # alt.datum.date < alt.expr.toDate('2022-05-19T00:00sadfs:00'),
       # alt.value([3, 3]),  # dashed line: 5 pixels  dash + 5 pixels space
       # alt.value([0])
    #)
    )

    chart = base.encode(
        x=alt.X('date:T',axis= alt.Axis(grid=False, title=''), scale=alt.Scale(domain=interval.ref())),
        y=alt.Y(aggregation+':Q', axis = alt.Axis(title=y_axis_title,  offset= 5), scale=alt.Scale(domain=list(domain2))),
        tooltip = alt.Tooltip(['date:T', aggregation+':Q', 'count:Q', 'beschreibung:N']),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
      #  strokeDash=alt.condition(
      #  alt.datum.date < alt.expr.toDate('2022-05-19T00:00:fasdf00'),
      #  alt.value([3, 3]),  # dashed line: 5 pixels  dash + 5 pixels space
      #  alt.value([0])
   # )
    ).properties(
        width=widtht,
        height=height
    )

    #count_selector = alt.selection(type='single', encodings=['x'])

    count_chart = base.mark_bar(size=6.8).encode(
        #x=alt.X('date:T',axis= alt.Axis(grid=False, title=''), scale=alt.Scale(domain=interval.ref())),
        #y=alt.Y('mean:Q', axis = alt.Axis(title='Arbeitspreis (ct/kWh)')),
        x=alt.X('date:T',axis= alt.Axis(grid=False, title=''), scale=alt.Scale(domain=interval.ref())),
        y=alt.Y('count:Q', axis = alt.Axis(title='Anzahl Ergenbisse'),scale=alt.Scale(domain=list(domain3))),
        color=alt.Color('beschreibung:N', scale=alt.Scale(domain=dom, range=rng)),
        opacity=alt.condition(nearest, alt.value(1), alt.value(0.5)),
        tooltip = alt.Tooltip(['date:T', aggregation+':Q', 'count:Q', 'beschreibung:N']),
        order=alt.Order(
        # Sort the segments of the bars by this field
        'count:Q',
      sort='descending'
    )
    ).properties(
        width=widtht,
        height=200
    )
    #.add_selection(count_selector)

    
    # Base chart for data tables
    #ranked_text = alt.Chart(source).mark_text(align='right').encode(
    #    y=alt.Y('row_number:O',axis=None)
    #).transform_filter(
    #    count_selector
    #).transform_window(
    #    row_number='row_number()'
    #).transform_filter(
    #    'datum.row_number < 15'
    #)
    
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
        text=alt.condition(nearest, 'count:Q', alt.value(' ')),
        y=alt.Y('row_number:O',axis=None),
        color=alt.Color('beschreibung:N', scale=alt.
                    Scale(domain=dom, range=rng))
    ).transform_filter(
        nearest
    ).transform_window(
        row_number='row_number()'
    ).properties(
        width=80,
        height=80
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
        width=80,
        height=80
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
        width=80,
        height=80
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

    count_chart_view = alt.vconcat(count_chart ,count_text_date ,(count_text.properties(title=alt.TitleParams(text='Anzahl Anfragenergebnisse', align='left')) | count_text.encode(text=alt.condition(nearest, aggregation+':Q', alt.value(' '), format=".2f")).properties(title=alt.TitleParams(text=aggregation_dict[aggregation], align='left')) | count_text.encode(text='beschreibung:N').properties(title=alt.TitleParams(text='Beschreibung', align='left'))))
    
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

    #final_view.save('D:\energy_data_visualization\energy_chart.html')
    return final_view

def summarize_tariffs(results, date='2022-02-24'):
    tariffs_at_date = results[results.date == '2022-02-24']
    tariff_summary = tariffs_at_date.drop_duplicates(['date', 'providerName', 'tariffName'])

    boxplot = altcat.catplot(tariff_summary,
               height=350,
               width=450,
               mark='point',
               box_mark=dict(strokeWidth=2, opacity=0.6),
               whisker_mark=dict(strokeWidth=2, opacity=0.9),
               encoding=dict(x=alt.X('dataeco:N', title=None),
                             y=alt.Y('dataunit:Q',scale=alt.Scale(zero=False)),
                             color=alt.Color('providerName:N', legend=None)),
               transform='jitterbox',
              jitter_width=0.5)

    return tariff_summary, boxplot

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

#### HEADER REGION

row1_1, row1_2 = st.columns((2, 3))

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

st.markdown("""---""")

### MENU AUSWAHL REGION
selection_menu_container = st.container()
time_selection_column, attribute_selection_column = selection_menu_container.columns([1,2])

#attribute_selection_menu_container = st.container()
#attribute_selection_column, attribute_aggregation_type_column = selection_menu_container.columns([1,3])

#division_selection_container = st.container()
division_selection_column, division_value_selection_column = selection_menu_container.columns([1,3])

##Zeitintervallauswahl
today = date.today()
tree_months_ago = today - timedelta(days=90)
date_interval = [tree_months_ago, today]

#time_selection_column.write('**Zeitraum**')

time_selection_column.write("**Zeitraum:**")
time_selection = time_selection_column.selectbox(
    label=' ',
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

plz_list = electricity_results_3000['plz'].unique().tolist()
print(electricity_results_3000['plz'].unique())
plz_list.append('Alle')

with time_selection_column:
    st.multiselect(
            'Tarife aus welchen Postleitzahlen soll enthalten sein?',
            plz_list,
            default=['Alle'])

#Energietypauswahl
#energy_type_selections = selection_dropdown_column[0].multiselect(
#    'What are your favorite colors',
#    ['Strom','Gas', 'Wetter', 'Spotmarktpreise', 'Börsenpreise', 'Erzeugung Enerneuerbare'],
#    default=['Strom', 'Gas'])

attribute_selection_column.write("**Attributauswahl**")

selected_variable = attribute_selection_column.selectbox(
    'Welches Attribut möchtest du anschauen?',
    ('Arbeitspreis', 'Grundpreis', 'Jahreskosten'),
    index=0)


mean_median_btn = attribute_selection_column.radio(
        ("Wie möchtest du den {selected_variable} der Tarife aggregieren?").format(selected_variable=selected_variable),
        options=["Durchschnitt", "Median", "Minimum", "Maximum", "Standardabweichung"],
    )

with attribute_selection_column:
    top_n = st.multiselect(
                'Tarife aus welchen Postleitzahlen soll enthalten sein?',
                [1, 5, 10, 15],
                default=['Alle'])

division_expander = st.expander('Weiteres Unterscheidungsmerkmal 🍎🍏 - Hier kannst du ein weiteres Unterscheidungsmerkmal an welches du die Tarife aufteilen möchtest auswählen.', expanded=False)

with division_expander:
    st.info(('Gebe ein weiteres Unterscheidungsmerkmal ein welchest du betrachten möchtest. \nZ.B.: Vergleiche die Entwicklung von {selected_variable} für Tarife mit **mit langer Preisgarantie** Tarife **mit kurzer Preisgarantie**.').format(selected_variable=selected_variable))

    sep_var_col, sep_val_col = st.columns(2)
        
    seperation_var = sep_var_col.selectbox('Nach welches Attribut möchtest du aufteilen?',
    ('Vertragslaufzeit', 'Preisgarantie', 'Öko Tarif/ Konventioneller Tarif','Partner', 'Anbieter'),
    index=1,
    help="Gebe hier ein nach welhes Attribut du trennen möchtest: Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At")
            
    selection_slider = 12

    if( (seperation_var =='Vertragslaufzeit') |(seperation_var =='Preisgarantie')  ):
        selection_slider = sep_val_col.slider('Ab welchen Wert für das Attribut '+seperation_var+ ' teilen?', 0, 24, 12, step=3,
        help="Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At")
    elif((seperation_var =='Anbieter')):
        col1, col2 = st.columns(2)

        gas = gas_results_15000.copy()
        gas['type'] = 'gas'
        electricity = electricity_results_3000.copy()
        electricity['type'] = 'electricity'

        all = pd.concat([gas, electricity]).drop_duplicates(['providerName'])

        col1.write(all)
        col2.write(len(all))

### ENDE MENU AUSWAHL REGION

st.markdown("""---""")

#### ANNOTATION REGION

events_df = load_events_df()
annotation_container = st.expander('Ereignisse 📰🌟 - Hier kannst du Ereinisse in die Zeitachse der Grafiken einblenden oder entfernen', expanded=False)

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

st.markdown("""---""")
## ENDE ANNOTATION REGION

main_chart_container = st.container()
energy_type_selections = ['Strom', 'Gas']
electricity_chart_column, gas_chart_column = st.columns(2) 

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left;} </style>', unsafe_allow_html=True)
st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)
#st.radio("",("Durchschnitt","Median"))

main_chart_container.write(('**Preisentwicklung - {selected_variable} und {seperation_var} der Strom - und Gastarife.**').format(selected_variable=selected_variable, seperation_var=seperation_var).upper())
main_chart_container.write(('Die oberen zwei Grafiken zeigen die Entwicklung der Tarife bezüglich {selected_variable} und {seperation_var}. Im dritten Grafik ist die Anzahl der Suchanfragenergebnisse visualisiert.').format(selected_variable=selected_variable, seperation_var=seperation_var))

if(len(date_interval) == 2):
    with electricity_chart_column:
        chart_header = "**{energy_selection}verträge ({selected_variable})**".format(selected_variable=selected_variable, energy_selection='Strom')
        summary_3000 = summarize(electricity_results_3000, seperation_var, int(selection_slider),'3000',selected_variable, top_n=top_n)
        summary_1300 = summarize(electricity_results_1300, seperation_var, int(selection_slider),'1300', selected_variable, top_n=top_n)
        summary = pd.concat([summary_3000, summary_1300])
        st.write(chart_header)
        energy_line_chart_e = create_chart(summary,mean_median_btn, int(selection_slider), date_interval=date_interval, selected_variable=selected_variable, events_df=selected_events,energy_type='electricity', seperation_var=seperation_var)
        st.altair_chart(energy_line_chart_e, use_container_width=True)

        tariff_list_expander = st.expander('Tarife', expanded=False)

        with tariff_list_expander:
            st.info('Hier ist gedacht die Tarife aufzulisten die oben im Barchart ausgewählt sind')
    with gas_chart_column:
        chart_header = "**{energy_selection}verträge ({selected_variable})**".format(selected_variable=selected_variable, energy_selection='Gas')
        summary_9000 = summarize(gas_results_9000, seperation_var,int(selection_slider),'9000',selected_variable, top_n=top_n)
        summary_15000 = summarize(gas_results_15000, seperation_var,int(selection_slider),'15000',selected_variable, top_n=top_n)
        summary = pd.concat([summary_9000, summary_15000])
        st.write(chart_header)
        energy_line_chart_e = create_chart(summary,mean_median_btn, int(selection_slider), date_interval=date_interval, selected_variable=selected_variable, events_df=selected_events,energy_type='gas', seperation_var=seperation_var)
        st.altair_chart(energy_line_chart_e, use_container_width=True)

        tariff_list_expander = st.expander('Tarife', expanded=False)

        with tariff_list_expander:
            st.info('Hier ist gedacht die Tarife aufzulisten die oben im Barchart ausgewählt sind')


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