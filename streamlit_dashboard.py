import numpy as np
import pandas as pd
import os
import re
import datetime
from datetime import datetime as dt
from datetime import date, timedelta
import altair as alt
import altair_catplot as altcat
import streamlit as st
import threading
import concurrent.futures
import json

def foo(bar):
    print('hello {}'.format(bar))
    return 'foo'


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

@st.cache(ttl=24*60*60)
def read_energy_data(energy_type, verbrauch):
    ## Lese alle Dateien und füge sie zu einer Liste zusammen

    '''
    gas_path = 'D:\energy_data_visualization\data\{energy_type}'.format(energy_type=energy_type)
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
def summarize(results, seperation_var='priceGuaranteeNormalized',seperation_value=12, consumption='unknown',selected_variable='dataunit'):

    sep_var_readable = seperation_var
    if(seperation_var == 'Vertragslaufzeit'):
        seperation_var = 'contractDurationNormalized'
    elif(seperation_var == 'Preisgarantie'):
        seperation_var='priceGuaranteeNormalized'
    
    variables_dict = {
        "Arbeitspreis": "dataunit",
        "Grundpreis": "datafixed",
        "Jahreskosten": "Jahreskosten"
        }

    agg_functions = {
        variables_dict[selected_variable]:
        [ 'mean', 'median','std', 'min', 'max', 'count']
    }
    
    ohne_laufzeit  = results[results[seperation_var] < seperation_value]
    summary_ohne_laufzeit = ohne_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
    summary_ohne_laufzeit.columns =  [ 'mean', 'median','std', 'min', 'max', 'count']
    summary_ohne_laufzeit['date'] = summary_ohne_laufzeit.index
    summary_ohne_laufzeit['beschreibung'] = 'Verbrauch: '+consumption+'\n'+sep_var_readable+' < '+str(seperation_value) 
    
    #mit_laufzeit  = high_consume[high_consume['contractDurationNormalized'] > 11]
    mit_laufzeit  = results[results[seperation_var] >= seperation_value]
    summary_mit_laufzeit = mit_laufzeit[ ['date','providerName','tariffName','signupPartner', variables_dict[selected_variable]]].groupby(['date']).agg(agg_functions)
    summary_mit_laufzeit.columns =  [ 'mean', 'median','std', 'min', 'max', 'count']
    summary_mit_laufzeit['date'] = summary_mit_laufzeit.index
    summary_mit_laufzeit['beschreibung'] = 'Verbrauch: '+consumption+'\n'+sep_var_readable+' >= '+str(seperation_value)

    summary = pd.concat([summary_mit_laufzeit, summary_ohne_laufzeit])
    return summary

#@st.cache(ttl=24*60*60)
def create_chart(summary,  aggregation='mean', seperation_var='priceGuaranteeNormalized', seperation_value=12, date_interval=['2022-07-17', '2022-10-17'], widtht=800, height=300,selected_variable='dataunit'):

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
    
    chart_max = np.ceil( chart_max + (0.8*chart_max))
    domain3 = np.linspace(0, chart_max, 2, endpoint = True)
    
    source = summary.copy()
    
    x_init = pd.to_datetime(date_interval).astype(int) / 1E6
    interval = alt.selection_interval(encodings=['x'],init = {'x':x_init.to_list()})
    selection = alt.selection_multi(fields=['beschreibung'], bind='legend')
    
    interval_y = alt.selection_interval(encodings=['y'], bind="scales")

    ## Eregignisse
    events_df = pd.DataFrame([
        {
            "start": "2022-07-01",
            "ereignis": "Strom: Abschaffung der EEG Umlage",
            "tooltip":"EEG Umlage wurde abgeschafft."
        },
        {
            "start": "2022-02-24",
            "ereignis": "Kriegsbegin in der Ukraine",
            "tooltip":"Invasion in der Ukraine"
        },
        {
            "start": "2022-10-01",
            "ereignis": "Senkung der Mehrwertsteuer für Gas",
            "tooltip":"Mehrwertsteuer für Gas wurde von 19% auf 7% gesenkt."
        }
    ])

    rule = alt.Chart(events_df).mark_rule(
        color="gray",
        strokeWidth=2,
        strokeDash=[12, 6]
    ).encode(
        x= alt.X('start:T', scale=alt.Scale(domain=interval.ref()))
    )

    events_text = alt.Chart(events_df).mark_text(
        align='left',
        baseline='middle',
        dx=-7,
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
    
    base = alt.Chart(source).mark_line(size=2).encode(
        #x= alt.X('date:T',axis= alt.Axis(grid=False, title='Datum')),
        y = alt.Y(aggregation+':Q', axis = alt.Axis(title=y_axis_title)),
        x= alt.X('date:T',axis= alt.Axis(grid=False, title='Datum')),
        #y = alt.Y('median:Q', axis = alt.Axis(title='Arbeitspreis (ct/kWh)')),
        color='beschreibung:N'
    )

    chart = base.encode(
        #x=alt.X('date:T',axis= alt.Axis(grid=False, title=''), scale=alt.Scale(domain=interval.ref())),
        #y=alt.Y('mean:Q', axis = alt.Axis(title='Arbeitspreis (ct/kWh)')),
        x=alt.X('date:T',axis= alt.Axis(grid=False, title=''), scale=alt.Scale(domain=interval.ref())),
        y=alt.Y(aggregation+':Q', axis = alt.Axis(title=y_axis_title),scale=alt.Scale(domain=list(domain2))),
        tooltip = alt.Tooltip(['date:T', aggregation+':Q', 'count:Q', 'beschreibung:N']),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
    ).properties(
        width=widtht,
        height=height
    )

    count_chart = base.mark_bar().encode(
        #x=alt.X('date:T',axis= alt.Axis(grid=False, title=''), scale=alt.Scale(domain=interval.ref())),
        #y=alt.Y('mean:Q', axis = alt.Axis(title='Arbeitspreis (ct/kWh)')),
        x=alt.X('date:T',axis= alt.Axis(grid=False, title=''), scale=alt.Scale(domain=interval.ref())),
        y=alt.Y('count:Q', axis = alt.Axis(title='Anzahl Ergenbisse'),scale=alt.Scale(domain=list(domain3))),
        color='beschreibung:N',
        tooltip = alt.Tooltip(['date:T', aggregation+':Q', 'count:Q', 'beschreibung:N'])
    ).properties(
        width=widtht,
        height=60
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

    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['date'], empty='none')

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
        text=alt.condition(nearest, 'median:Q', alt.value(' '))
    )

    count_text = count_chart.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, 'count:Q', alt.value(' ')),
        color='beschreibung:N'
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

    count_chart_view = alt.layer(
        count_chart , selectors,  rules, count_text
    ).properties(
        width=widtht,
        height=height
    )

    annotationen = rule + events_text

    main_view = main_view + annotationen

    final_view = main_view.add_selection(
    selection
    ).interactive(bind_x=False) & count_chart_view & view 

    final_view = final_view.configure_legend(
  orient='bottom',
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

selection_menu_container = st.container()
selection_dropdown_column = selection_menu_container.columns([2,1,2])
main_chart_container = st.container()

today = date.today()
tree_months_ago = today - timedelta(days=90)
date_interval = [tree_months_ago, today]

date_interval = selection_dropdown_column[1].date_input(label='Date Range: ',
            value=(tree_months_ago, 
                    today),
            key='#date_range',
            help="Start-und End Datum: Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At")

mean_median_btn = selection_dropdown_column[1].radio(
        "Wie möchtest du die Tarifdaten aggregieren?",
        options=["mean", "median"],
    )

seperation_var = selection_dropdown_column[2].selectbox(
    'Nach welches Attribut möchtest du trennen?',
    ('Vertragslaufzeit', 'Preisgarantie', 'Öko Tarif/ Konventioneller Tarif'),
    help="Gebe hier ein nach welhes Attribut du trennen möchtest: Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At")

selection_slider = selection_dropdown_column[2].slider('Ab welchen Wert für die Variable '+seperation_var+ ' möchtest die Daten teilen?', 0, 24, 12, step=3,
    help="Start-und End Datum: Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At")

selected_variable = selection_dropdown_column[0].selectbox(
    'Welches Attribut möchtest du anschauen?',
    ('Arbeitspreis', 'Grundpreis', 'Jahreskosten'))

energy_type_selections = selection_dropdown_column[0].multiselect(
    'What are your favorite colors',
    ['Strom','Gas'],
    default=['Strom'])

chart_columns = main_chart_container.columns(len(energy_type_selections)) 

for i, energy_selection in enumerate(energy_type_selections):
    if((energy_selection == 'Strom')):
        summary_3000 = summarize(electricity_results_3000, seperation_var, int(selection_slider),'3000',selected_variable)
        summary_1300 = summarize(electricity_results_1300, seperation_var, int(selection_slider),'1300', selected_variable)
        summary = pd.concat([summary_3000, summary_1300])
    elif((energy_selection == 'Gas')):
        summary_9000 = summarize(gas_results_9000, seperation_var,int(selection_slider),'9000',selected_variable)
        summary_15000 = summarize(gas_results_15000, seperation_var,int(selection_slider),'15000',selected_variable)
        summary = pd.concat([summary_9000, summary_15000])

#summary = summarize(high_consume, seperation_var,int(selection_slider))

    if(len(date_interval) == 2):
        energy_line_chart_e = create_chart(summary,mean_median_btn, int(selection_slider), date_interval=date_interval, selected_variable)
        chart_columns[i].altair_chart(energy_line_chart_e)

#print(high_consume.dtypes)
#tariff_summary, boxplot = summarize_tariffs(high_consume)

#main_chart_container.altair_chart(boxplot)





