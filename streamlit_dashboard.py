import numpy as np
import pandas as pd
import os
import re
import datetime
import altair as alt
import altair_catplot as altcat
import streamlit as st


st.set_page_config(page_title="Energy Dashboard",
                    page_icon=":bar_chart:",
                    layout="wide")


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
def read_energy_data():
    ## Lese alle Dateien und füge sie zu einer Liste zusammen
    gas_path = 'D:\energy_data_visualization\data\electricity'
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
    #### lese die Daten der wöchentlichen Abfrage zu den 5 Städten
    wa_df = pd.read_excel('D:\energy_data_visualization\data\wa_strom.xlsx') 
    #wa_df = pd.read_excel('D:\energy_data_visualization\data\wa_gas.xlsx') 

    wa_df = wa_df[(wa_df.ID == 1) | 
    (wa_df.ID == 7) |
    (wa_df.ID == 11) |
    (wa_df.ID == 15) |
    (wa_df.ID == 19) ]

    ###
    wa_df['Einheit Vertragslaufzeit'].fillna('nan')
    wa_df['Einheit Garantielaufzeit'].fillna('nan')

    wa_df['contractDurationNormalized'] = wa_df.apply(lambda row : unit_to_month(row['Einheit Vertragslaufzeit'], row['Vertragslaufzeit']), axis = 1)
    wa_df['priceGuaranteeNormalized'] = wa_df.apply(lambda row : unit_to_month(row['Einheit Garantielaufzeit'], row['Garantielaufzeit']), axis = 1)

    wa_df.rename(columns = {'Datum':'date','Anbieter':'providerName','Tarifname':'tariffName', 'Partner':'signupPartner', 'Arbeitspreis':'dataunit', 'Öko':'dataeco'}, inplace = True)
    wa_df['plz'] = wa_df.apply(lambda row : set_plz(row['ID']), axis = 1)
    wa_df = wa_df[wa_df.date < all_dates.date.min()]
    wa_df = wa_df.drop_duplicates(['date', 'providerName', 'tariffName', 'signupPartner', 'plz'])
    
    all_dates = pd.concat([wa_df, all_dates])
    ###

    data_types_dict = {'date':'<M8[ns]', 'providerName':str, 'tariffName':str,'signupPartner':str, 'dataunit':float, 'dataeco':bool, 'plz':str }
    all_dates = all_dates.astype(data_types_dict)

    print('MIT DEM EINLESEN DER 5 PLZ DATEN FERTIG')
    return all_dates[['date', 'providerName', 'tariffName', 'signupPartner', 'plz', 'dataunit',  'contractDurationNormalized', 'priceGuaranteeNormalized', 'dataeco']]

high_consume = read_energy_data()


#@st.cache(ttl=24*60*60)
def summarize(high_consume, seperation_var='priceGuaranteeNormalized',seperation_value=12):

    sep_var_readable = seperation_var
    if(seperation_var == 'Vertragslaufzeit'):
        seperation_var = 'contractDurationNormalized'
    elif(seperation_var == 'Preisgarantie'):
        seperation_var='priceGuaranteeNormalized'
    

    agg_functions = {
        'dataunit':
        [ 'mean', 'median','std', 'min', 'max', 'count']
    }
    
    ohne_laufzeit  = high_consume[high_consume[seperation_var] < seperation_value]
    summary_ohne_laufzeit = ohne_laufzeit[ ['date','providerName','tariffName','signupPartner', 'dataunit']].groupby(['date']).agg(agg_functions)
    summary_ohne_laufzeit.columns =  [ 'mean', 'median','std', 'min', 'max', 'count']
    summary_ohne_laufzeit['date'] = summary_ohne_laufzeit.index
    summary_ohne_laufzeit['beschreibung'] = sep_var_readable+' < '+str(seperation_value) 

    #mit_laufzeit  = high_consume[high_consume['contractDurationNormalized'] > 11]
    mit_laufzeit  = high_consume[high_consume[seperation_var] >= seperation_value]
    summary_mit_laufzeit = mit_laufzeit[ ['date','providerName','tariffName','signupPartner', 'dataunit']].groupby(['date']).agg(agg_functions)
    summary_mit_laufzeit.columns =  [ 'mean', 'median','std', 'min', 'max', 'count']
    summary_mit_laufzeit['date'] = summary_mit_laufzeit.index
    summary_mit_laufzeit['beschreibung'] = sep_var_readable+' >= '+str(seperation_value)

    summary = pd.concat([summary_mit_laufzeit, summary_ohne_laufzeit])
    return summary


#@st.cache(ttl=24*60*60)
def create_chart(summary,  aggregation='mean',seperation_var='priceGuaranteeNormalized', seperation_value=12):
    ## Definitionsbereich der Y achse
    min = np.floor(summary['median'].min() - 5)
    max = np.ceil( summary['median'].max() + 5)
    domain1 = np.linspace(min, max, 2, endpoint = True)
    

    source = summary.copy()
    interval = alt.selection_interval(encodings=['x'])
    single = alt.selection_single()
    #x_init = pd.to_datetime(['2005-03-01', '2005-04-01']).astype(int) / 1E6
    #interval = alt.selection_interval(encodings=['x'], init={'x':list(x_init)})


    interval_y = alt.selection_interval(encodings=['y'], bind="scales")

    ## Eregignisse
    events_df = pd.DataFrame([
        {
            "start": "2022-07-01",
            "ereignis": "Abschaffung der EEG Umlage"
        },
        {
            "start": "2022-02-24",
            "ereignis": "Kriegsbegin in der Ukraine"
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
    )

    ## Visualisierung:
    
    base = alt.Chart(source).mark_line(size=2).encode(
        #x= alt.X('date:T',axis= alt.Axis(grid=False, title='Datum')),
        y = alt.Y(aggregation+':Q', axis = alt.Axis(title='Arbeitspreis (ct/kWh)'),scale=alt.Scale(domain=list(domain1))),
        x= alt.X('date:T',axis= alt.Axis(grid=False, title='Datum')),
        #y = alt.Y('median:Q', axis = alt.Axis(title='Arbeitspreis (ct/kWh)')),
        color='beschreibung:N'
    )

    chart = base.encode(
        #x=alt.X('date:T',axis= alt.Axis(grid=False, title=''), scale=alt.Scale(domain=interval.ref())),
        #y=alt.Y('mean:Q', axis = alt.Axis(title='Arbeitspreis (ct/kWh)')),
        x=alt.X('date:T',axis= alt.Axis(grid=False, title=''), scale=alt.Scale(domain=interval.ref())),
        y=alt.Y(aggregation+':Q', axis = alt.Axis(title='Arbeitspreis (ct/kWh)')),
        tooltip = alt.Tooltip(['date:T', aggregation+':Q', 'count:Q', 'beschreibung:N'])
    ).properties(
        width=1000,
        height=400
    )

    view = base.add_selection(
        interval
    ).properties(
        width=1000,
        height=80,
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

    # Draw a rule at the location of the selection
    rules = alt.Chart(source).mark_rule(color='gray').encode(
        x='date:T',
    ).transform_filter(
        nearest
    )

    # Put the five layers into a chart and bind the data
    
    main_view = alt.layer(
        chart, selectors, points, rules, text
    ).properties(
        width=1000, height=400
    )

    annotationen = rule + events_text

    main_view = main_view + annotationen

    final_view = main_view & view 
    

    #final_view.save('D:\energy_data_visualization\energy_chart.html')
    return final_view

def summarize_tariffs(high_consume, date='2022-02-24'):
    tariffs_at_date = high_consume[high_consume.date == '2022-02-24']
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


#def create_chart2(tariff_summary): 

selection_menu_container = st.container()
selection_dropdown_column = selection_menu_container.columns(2)
main_chart_container = st.container()

seperation_var = selection_dropdown_column[0].selectbox(
    'Nach welches Attribut möchtest du trennen?',
    ('Vertragslaufzeit', 'Preisgarantie', 'Öko Tarif/ Konventioneller Tarif'))


selection_dropdown_column[1].write('Vergleiche nach: '+ seperation_var)

selection_slider = selection_dropdown_column[1].slider('Ab welchen Wert für die Variable '+seperation_var+ ' möchtest die Daten teilen?', 0, 24, 12, step=3)


summary = summarize(high_consume, seperation_var,int(selection_slider))

mean_median_btn = selection_dropdown_column[0].radio(
        "Set selectbox label visibility 👉",
        options=["mean", "median"],
    )
selection_dropdown_column[0].write(str(mean_median_btn))

energy_line_chart = create_chart(summary,mean_median_btn, int(selection_slider))

main_chart_container.altair_chart(energy_line_chart)

#start_date = selection_dropdown_column[0].date_input("Start Date", value=pd.to_datetime("2021-01-31", format="%Y-%m-%d"))
#end_date = selection_dropdown_column[0].date_input("End Date", value=pd.to_datetime("today", format="%Y-%m-%d"))

#start = start_date.strftime("%Y-%m-%d")
#end = end_date.strftime("%Y-%m-%d")

print(high_consume.dtypes)
tariff_summary, boxplot = summarize_tariffs(high_consume)

main_chart_container.altair_chart(boxplot)





