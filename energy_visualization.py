import numpy as np
import pandas as pd
import os
import re
import datetime
import altair as alt



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
high_consume = all_dates
high_consume['dataunit'] = high_consume['dataunit'].str.replace(',','.').astype(float)
print('MIT DEM EINLESEN DER 100 PLZ DATEN FERTIG')
#### lese die Daten der wöchentlichen Abfrage zu den 5 Städten
wa_df = pd.read_excel('D:\energy_data_visualization\data\wa_strom.xlsx')

wa_df = wa_df[(wa_df.ID == 1) | 
(wa_df.ID == 7) |
(wa_df.ID == 11) |
(wa_df.ID == 15) |
(wa_df.ID == 19) ][['Datum', 'Anbieter', 'Tarifname', 'Arbeitspreis', 'Grundpreis', 'Vertragslaufzeit', 'Einheit Vertragslaufzeit']]

print('MIT DEM EINLESEN DER 5 PLZ DATEN FERTIG')

## Welche agregationen möchten wir auf den Arbeitspreis
agg_functions = {
    'Arbeitspreis':
    [ 'mean', 'median','std', 'min', 'max', 'count']
}

## fasse agregationen zusammen 
summary_wa_df = wa_df.groupby(['Datum']).agg(agg_functions)

summary_wa_df.columns =  [ 'mean', 'median','std', 'min', 'max', 'count']
summary_wa_df['date'] = summary_wa_df.index
summary_wa_df['beschreibung'] = 'Abfrage 5 Städte ohne VL unterscheidung'

agg_functions = {
    'dataunit':
    [ 'mean', 'median','std', 'min', 'max', 'count']
}

ohne_laufzeit  = high_consume[high_consume['contractDurationNormalized'] <= 11]
summary_ohne_laufzeit = ohne_laufzeit[ ['date','providerName','tariffName','signupPartner', 'dataunit']].groupby(['date']).agg(agg_functions)
summary_ohne_laufzeit.columns =  [ 'mean', 'median','std', 'min', 'max', 'count']
summary_ohne_laufzeit['date'] = summary_ohne_laufzeit.index
summary_ohne_laufzeit['beschreibung'] = 'Abfrage 100 Städte ohne Laufzeit'  

mit_laufzeit  = high_consume[high_consume['contractDurationNormalized'] > 11]
summary_mit_laufzeit = mit_laufzeit[ ['date','providerName','tariffName','signupPartner', 'dataunit']].groupby(['date']).agg(agg_functions)
summary_mit_laufzeit.columns =  [ 'mean', 'median','std', 'min', 'max', 'count']
summary_mit_laufzeit['date'] = summary_mit_laufzeit.index
summary_mit_laufzeit['beschreibung'] = 'Abfrage 100 Städte mit Laufzeit' 

summary = pd.concat([summary_mit_laufzeit, summary_ohne_laufzeit, summary_wa_df])


## Definitionsbereich der Y achse
min = np.floor(summary['median'].min() - 5)
max = np.ceil( summary['median'].max() + 5)
domain1 = np.linspace(min, max, 2, endpoint = True)

## Eregignisse
presidents = pd.DataFrame([
    {
        "start": "2022-07-01",
        "ereignis": "Abschaffung EEG Umlage"
    },
    {
        "start": "2022-02-24",
        "ereignis": "Kriegsbegin"
    }
])

rule = alt.Chart(presidents).mark_rule(
    color="gray",
    strokeWidth=2,
    strokeDash=[12, 6]
).encode(
    x='start:T'
)


events_text = alt.Chart(presidents).mark_text(
    align='left',
    baseline='middle',
    dx=7,
    dy=-135,
    size=11
).encode(
    x='start:T',
    text='ereignis',
    color=alt.value('#000000')
)

## Visualisierung:
source = summary.copy()
interval = alt.selection_interval(encodings=['x'])
interval_y = alt.selection_interval(encodings=['y'])

base = alt.Chart(source).mark_line(size=2).encode(
    x= alt.X('date:T',axis= alt.Axis(grid=False, title='Datum')),
    y = alt.Y('median:Q', axis = alt.Axis(title='Arbeitspreis (ct/kWh)'),scale=alt.Scale(domain=list(domain1))),
    color='beschreibung:N'
)

chart = base.encode(
    x=alt.X('date:T',axis= alt.Axis(grid=False, title=''), scale=alt.Scale(domain=interval.ref())),
    y=alt.Y('median:Q', axis = alt.Axis(title='Arbeitspreis (ct/kWh)'),scale=alt.Scale(domain=interval_y.ref())),
    tooltip = alt.Tooltip(['date:T', 'median:Q', 'count:Q', 'beschreibung:N'])
).properties(
    width=800,
    height=300
)

view = base.add_selection(
    interval
).add_selection(
    interval_y
).properties(
    width=800,
    height=100,
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
    width=600, height=300
).resolve_scale(
    y='shared'
)

annotationen = rule + events_text

main_view = main_view + annotationen

final_view = alt.vconcat(main_view  & view).properties(
    title={
      "text": ["Entwicklung der Strompreise in Deutschland"], 
      #"subtitle": ["compared to average of 3 Jan - 6 Feb"],
      "color": "black",
      "orient":'top', 
      "anchor":'middle'
      #"subtitleColor": "darkgray"
    }
)

final_view.save('D:\energy_data_visualization\energy_chart.html')