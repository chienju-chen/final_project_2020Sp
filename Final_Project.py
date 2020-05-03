# Import Packages
import numpy as np
import pandas as pd
from pytrends.request import TrendReq
from scipy import signal
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#from kneed import KneeLocator

# Extract confirmed case numbers for target countries
def extract_target_country_case(filename, country_list):
    covid_all = pd.read_csv(filename)
    covid_target = covid_all[covid_all['Country/Region'].isin(country_list)]
    covid_target = covid_target.drop(columns=['Province/State','Lat','Long'])
    # adding up the confirmed case numbers for countires with multiple rows
    for country in country_list:
        covid_multiple_rows = covid_target[covid_target['Country/Region']==country]
        if covid_multiple_rows.shape[0] > 1:
            covid_target = covid_target.drop(covid_multiple_rows.index)
            covid_single_row = covid_multiple_rows.groupby('Country/Region').sum().reset_index()
            covid_target = covid_target.append(covid_single_row).reset_index(drop=True)
        else:
            continue
    return(covid_target)


filename = "data/time_series_covid19_confirmed_global.csv"
country_dict = {'Taiwan*':'TW', 'Korea, South':'KR', 'Italy':'IT', 'Spain':'ES', 'Czechia':'CZ',
                     'US':'US', 'Peru':'PE', 'Iran':'IR', 'Australia':'AU', 'South Africa':'ZA'}
country_list = list(country_dict.keys())
confirmed_case_target = extract_target_country_case(filename, country_list)

# Transpose confirmed_case_target dataframe
df_country = pd.DataFrame(country_dict.items(), columns=['Name','Code'])
confirmed_case_target = pd.merge(confirmed_case_target, df_country, how='left', left_on='Country/Region', right_on='Name')
confirmed_case_tr = confirmed_case_target.transpose()

# set country code as the column names of the transposed dataframe
confirmed_case_tr.columns = confirmed_case_tr.loc['Code']
confirmed_case_tr = confirmed_case_tr.drop(['Country/Region','Name','Code'])

# set date as the index of the transposed dataframe
confirmed_case_tr.index = pd.to_datetime(confirmed_case_tr.index)

# Extract Google Trends of single(multiple) keyword(s) for each target country
# (old)only for extracting keywords together
'''
def extract_google_trends(timeframe, keywords_dict):
    timeframe = timeframe[0]+" "+timeframe[1]
    pytrend = TrendReq(hl='en-US', tz=360)
    result_dict = {}
    for key in list(keywords_dict.keys())[:-1]:
        pytrend.build_payload(kw_list=keywords_dict[key], cat=0, timeframe=timeframe, geo=key)
        result_local = pytrend.interest_over_time().drop(columns=['isPartial'])
        if key == 'US':
            result = result_local
        else:
            result_local.columns = [keyword+"(local)" for keyword in keywords_dict['common']]
            pytrend.build_payload(kw_list=keywords_dict['common'], cat=0, timeframe=timeframe, geo=key)
            result_common = pytrend.interest_over_time().drop(columns=['isPartial'])
            result = pd.concat([result_local, result_common], axis=1)
        result['date'] = result.index.strftime("%Y/%m/%d")
        result_dict[key] = result
    return(result_dict)
'''
# extract Google Trends of single(multiple) keyword(s) for each target country
def extract_google_trends(timeframe, keywords_dict, single_search=True):
    timeframe = timeframe[0] + " " + timeframe[1]
    pytrend = TrendReq(hl='en-US', tz=360)
    result_dict = {}
    for key in list(keywords_dict.keys())[:-1]:
        # extract Google Trends of single keyword for each target country
        if single_search:
            result = pd.DataFrame()
            for word in keywords_dict[key]:
                pytrend.build_payload(kw_list=[word], cat=0, timeframe=timeframe, geo=key)
                if pytrend.interest_over_time().empty:
                    result = pd.concat([result, pd.DataFrame(0, index=result.index, columns=[word])], axis=1)
                result = pd.concat([result, pytrend.interest_over_time()], axis=1)
            # extract Google Trends of English keywords for each target country
            if key != 'US':
                result = result.drop(columns=['isPartial'])
                result.columns = [keyword + "(local)" for keyword in keywords_dict['common']]
                for word in keywords_dict['common']:
                    pytrend.build_payload(kw_list=[word], cat=0, timeframe=timeframe, geo=key)
                    result = pd.concat([result, pytrend.interest_over_time()], axis=1)
            result = result.drop(columns=['isPartial'])

        # extract Google Trends of multiple keywords for each target country
        else:
            pytrend.build_payload(kw_list=keywords_dict[key], cat=0, timeframe=timeframe, geo=key)
            result_local = pytrend.interest_over_time().drop(columns=['isPartial'])
            if key == 'US':
                result = result_local
            # extract Google Trends of English keywords for each target country
            else:
                result_local.columns = [keyword + "(local)" for keyword in keywords_dict['common']]
                pytrend.build_payload(kw_list=keywords_dict['common'], cat=0, timeframe=timeframe, geo=key)
                result_common = pytrend.interest_over_time().drop(columns=['isPartial'])
                result = pd.concat([result_local, result_common], axis=1)
        result_dict[key] = result
    return (result_dict)

timeframe = ('2020-01-01', confirmed_case_tr.index[-1].strftime('%Y-%m-%d'))
keywords_dict = {'TW':['武漢', '冠狀病毒', '武漢肺炎', '肺炎', '新冠肺炎'],
                 'KR':['우한', '코로나바이러스', '우한 폐렴', '폐렴', '신종 코로나바이러스'],
                 'IT': ['Wuhan', 'coronavirus', 'Polmonite di Wuhan', 'Polmonite', 'covid'],
                 'ES': ['Wuhan', 'coronavirus', 'neumonía de Wuhan', 'neumonía', 'covid'],
                 'CZ': ['Wu-chan', 'koronavirus', 'wuchanský koronavirus', 'zápal plic', 'covid'],
                 'US':['Wuhan', 'coronavirus', 'Wuhan coronavirus', 'pneumonia', 'covid'],
                 'PE':['Wuhan', 'coronavirus', 'neumonía de Wuhan', 'neumonía', 'covid'],
                 'IR':['ووهان','کروناویروس','ووهان کروناویروس','سینه‌پهلو','کووید'],
                 'AU':['Wuhan', 'coronavirus', 'Wuhan coronavirus', 'pneumonia','covid'],
                 'ZA':['Wuhan', 'coronavirus', 'Wuhan Koronavirus', 'Longontsteking','covid'],
                 'common':['Wuhan', 'coronavirus', 'Wuhan coronavirus', 'pneumonia', 'covid']}

single_google_trends_dict = extract_google_trends(timeframe, keywords_dict)
multi_google_trends_dict = extract_google_trends(timeframe, keywords_dict, single_search=False)

# Merge confirmed case dataframe to Google Trends query results
# Reference: https://thispointer.com/pandas-how-to-merge-dataframes-by-index-using-dataframe-merge-part-3/
for dictionary in [single_google_trends_dict, multi_google_trends_dict]:
    for key in dictionary:
        df_merged = pd.merge(dictionary[key], confirmed_case_tr[key], how='left', left_index=True, right_index=True)
        df_merged[key] = df_merged[key].fillna(0)
        dictionary[key] = df_merged.rename(columns={key: "Confirmed Case"})

#         # compute the daily increased cases as "Daily Case" column
#         daily_case = df_merged[key]-df_merged[key].shift(1)
#         df_merged = pd.concat([df_merged, daily_case.rename('Daily Case')], axis=1)

#         # compute the change in daily increased cases as "Daily Case Change" column
#         daily_case_change = df_merged[key]-2*df_merged[key].shift(1)+df_merged[key].shift(2)
#         df_merged = pd.concat([df_merged, daily_case_change.rename('Daily Case Change')], axis=1)


# Find the date where the number of confirmed case becomes notable in each country
# Reference: #### (1) https://plotly.com/python/smoothing/ #### (2) https://pypi.org/project/kneed/#input-data
def find_notable_date(confirmed_case):
    # smooth the confirmed case curve
    confirmed_case_smooth = pd.Series(signal.savgol_filter(confirmed_case, 15, 3), index=confirmed_case.index)

    # find the notable date with the maximum change in daily increased case
    daily_case_change_smooth = confirmed_case_smooth - 2 * confirmed_case_smooth.shift(1) + confirmed_case_smooth.shift(
        2)
    notable_date = daily_case_change_smooth.idxmax().strftime("%Y-%m-%d")

    #     # find the notable date, which is the day after the confirmed case curve reaches elbow point using kneed.KneeLocator package
    #     x, y = confirmed_case_smooth.reset_index().index, confirmed_case_smooth
    #     kn = KneeLocator(x, y, curve='convex', direction='increasing', online=True)
    #     notable_date_2 = confirmed_case_smooth.index[kn.knee+1].strftime('%Y-%m-%d')
    return (notable_date)


for key in multi_google_trends_dict:
    confirmed_case = multi_google_trends_dict[key]['Confirmed Case']
    notable_date = find_notable_date(confirmed_case)
    index = datetime.strptime(notable_date, '%Y-%m-%d')
    print("Notable date of", key, ": ", notable_date)
    print(confirmed_case[index - timedelta(5):index], "\n")

#     notable_date_1, notable_date_2 = find_notable_date(confirmed_case)
#     index1 = datetime.strptime(notable_date_1,'%Y-%m-%d')
#     index2 = datetime.strptime(notable_date_2,'%Y-%m-%d')
#     print(key+": ",notable_date_1, notable_date_2)
#     if index1 > index2:
#         print(confirmed_case[index2-timedelta(1):index1],"\n")
#     else:
#         print(confirmed_case[index1-timedelta(1):index2],"\n")


# Plot Google Trends query results (keywords are extrated together) vs. confirmed case number over time
# Reference: #### (1) https://seaborn.pydata.org/tutorial/color_palettes.html
# #### (2) https://matplotlib.org/3.2.1/gallery/shapes_and_collections/fancybox_demo.html
# #### (3) https://www.earthdatascience.org/courses/use-data-open-source-python/use-time-series-data-in-python/date-time-types-in-pandas-python/customize-dates-matplotlib-plots-python/

# old
'''
def plot_GT_case_trends(notable_rate, popularity, country, local=True):
    t = popularity.index.values
    confirmed_case = popularity['Confirmed Case']
    notable_date_1 = popularity['Daily Case Change(Smoothed)'].idxmax().strftime("%Y-%m-%d")
    # confirmed_case_smooth = signal.savgol_filter(confirmed_case, 15, 3)
    #    x,y = confirmed_case.reset_index().index, confirmed_case
    #     kn_old = KneeLocator(x, y, curve='convex', direction='increasing')
    #    kn = KneeLocator(x, y, curve='convex', direction='increasing', online=True)
    #     notable_date_2 = confirmed_case.index[kn_old.knee].strftime('%Y-%m-%d')
    #    notable_date_3 = confirmed_case.index[kn.knee].strftime('%Y-%m-%d')
    notable_date_4 = popularity[popularity['Daily Case'] > confirmed_case[-1] * notable_rate].index[0].strftime(
        '%Y-%m-%d')

    if local:  # when keywords are searched in local language
        popularity = popularity.iloc[:, :5]
        title = "Popularity of Google Search Queries(in local language) & Number of Confirmed Cases Over Time_" + country
    else:  # when keywords are searched in English
        popularity = popularity.iloc[:, 5:-5]
        title = "Popularity of Google Search Queries(in English) & Number of Confirmed Cases Over Time_" + country

    label = list(popularity.columns)
    color = sns.color_palette("husl", len(label))

    # plot Google Trends query trend 
    fig, ax1 = plt.subplots(figsize=(25, 10))
    for i in range(len(label)):
        ax1.plot(t, popularity.iloc[:, i], color=color[i], label=label[i], linewidth=3)

    ax2 = ax1.twinx()
    ax2.plot(t, confirmed_case, color='red', label='# of Confirmed Cases', linewidth=5)
    # ax2.plot(t, confirmed_case_smooth, color='green', linewidth=5)
    ax2.axvline(notable_date_1, color='black', ls='--', lw=3)
    ax2.axvline(notable_date_2, color='blue', ls='--', lw=3)
    # ax2.axvline(notable_date_3, color='green', ls='--', lw=3)
    ax2.axvline(notable_date_4, color='orange', ls='--', lw=3)

    date_form = mdates.DateFormatter("%Y-%m-%d")
    ax1.xaxis.set_major_formatter(date_form)
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax1.set_ylabel('Popularity of Query', fontsize=20)
    ax2.set_ylabel('Number of Confirmed Cases', fontsize=20, color='red')
    ax1.tick_params(axis='x', labelsize=15, labelrotation=45)
    ax1.tick_params(axis='y', labelsize=15)
    ax2.tick_params(axis='y', labelcolor='red', labelsize=15)
    ax1.grid(linewidth=0.5, color='gray')
    fig.legend(fontsize=15, loc=(0.07, 0.7))
    plt.title(title, fontsize=25)
    plt.show()
'''

def plot_GT_case_trends(popularity, country, local=True):
    t = popularity.index.values
    confirmed_case = popularity['Confirmed Case']
    notable_date = find_notable_date(confirmed_case)
    # confirmed_case_smooth = pd.Series(signal.savgol_filter(confirmed_case, 15, 3), index=confirmed_case.index)

    if local:  # when keywords are searched in local language
        popularity = popularity.iloc[:, :5]
        title = "Popularity of Google Search Queries(in local language) & Number of Confirmed Cases Over Time_" + country
    else:  # when keywords are searched in English
        popularity = popularity.iloc[:, 5:-1]
        title = "Popularity of Google Search Queries(in English) & Number of Confirmed Cases Over Time_" + country

    label = list(popularity.columns)
    color = sns.color_palette("husl", len(label))

    # plot the trend of Google Trends queries
    pd.plotting.register_matplotlib_converters()
    fig, ax1 = plt.subplots(figsize=(25, 10))
    for i in range(len(label)):
        ax1.plot(t, popularity.iloc[:, i], color=color[i], label=label[i], linewidth=3)

    # plot the number of confirmed cases & the notable date
    ax2 = ax1.twinx()
    ax2.plot(t, confirmed_case, color='red', label='# of Confirmed Cases', linewidth=5)
    # ax2.plot(t, confirmed_case_smooth, color='green', linewidth=5)
    ax2.axvline(notable_date, color='blue', ls='--', lw=3)
    bbox = dict(fc='white', ec='blue', alpha=0.6)
    ax2.text(notable_date, confirmed_case[-1] * 0.97, notable_date, color='blue', fontsize=20, bbox=bbox, ha='center')

    date_form = mdates.DateFormatter("%Y-%m-%d")
    ax1.xaxis.set_major_formatter(date_form)
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax1.set_ylabel('Popularity of Query', fontsize=20)
    ax2.set_ylabel('Number of Confirmed Cases', fontsize=20, color='red')
    ax1.tick_params(axis='x', labelsize=15, labelrotation=45)
    ax1.tick_params(axis='y', labelsize=15)
    ax2.tick_params(axis='y', labelcolor='red', labelsize=15)
    ax1.grid(linewidth=0.5, color='gray')
    fig.legend(fontsize=15, loc=(0.07, 0.7))
    plt.title(title, fontsize=25)
    plt.show()


for key in multi_google_trends_dict:
    popularity = multi_google_trends_dict[key]
    plot_GT_case_trends(popularity, key)
    if key != 'US':
        plot_GT_case_trends(popularity, key, local=False)

# Plot Google Trends query results (keywords are extrated individually) vs. confirmed case number over time
for key in single_google_trends_dict:
    popularity = single_google_trends_dict[key]
    plot_GT_case_trends(popularity, key)
    if key != 'US':
        plot_GT_case_trends(popularity, key, local=False)

# Bar Plot for Awareness Gap Days and Severity
def find_search_peak_date(multi_google_trends_dict, most_popular_keywords_dict):
    search_peak_dict = {}
    for key in multi_google_trends_dict:
        search_peak_dict[key] = multi_google_trends_dict[key][most_popular_keywords_dict[key]].idxmax()
    return(search_peak_dict)
    # results of search peak of single_google_trends_dict will be the same.

# If have time... need Modification to automatically get the most popular keywords
most_popular_keywords_dict = {'TW': 'Wuhan(local)',
                              'KR': 'Wuhan(local)',
                              'IT': 'coronavirus(local)',
                              'ES': 'coronavirus(local)',
                              'CZ': 'coronavirus(local)',
                              'US': 'coronavirus',
                              'PE': 'coronavirus(local)',
                              'IR': 'coronavirus(local)',
                              'AU': 'coronavirus(local)',
                              'ZA': 'coronavirus(local)'}

search_peak_dict = find_search_peak_date(multi_google_trends_dict, most_popular_keywords_dict)
# print(search_peak_dict)

notable_date_dict = {}
for key in multi_google_trends_dict:
    notable_date_dict[key] = datetime.strptime(notable_date,'%Y-%m-%d')
# print(notable_date_dict)

aware_period_in_days = {}
for key in multi_google_trends_dict:
    aware_days = datetime.date(notable_date_dict[key]) - datetime.date(search_peak_dict[key])
    aware_period_in_days[key] = aware_days.days
# print(aware_period_in_days)

# plt.bar(aware_period_in_days.keys(), aware_period_in_days.values())

# Severity Degree
def popul_density_target_country(filename, country_list_popul):
    popul_density = pd.read_csv(filename)
    popul_target = popul_density[popul_density['Country (or dependency)'].isin(country_dict_popul.values())]
    popul_target = popul_target[['Country (or dependency)', 'Density (P/Km?)', 'Population (2020)']]
    return popul_target

filename = 'data/Countries in the world by population (2020).csv'
# noteice that country name here is different from confirmed cases data
country_dict_popul = {'Taiwan*':'Taiwan', 'Korea, South':'South Korea', 'Italy':'Italy', 'Spain':'Spain', 'Czechia':'Czech Republic (Czechia)',
                     'US':'United States', 'Peru':'Peru', 'Iran':'Iran', 'Australia':'Australia', 'South Africa':'South Africa'}
country_list_popul = list(country_dict_popul.keys())
popul_density_target = popul_density_target_country(filename, country_dict_popul)

popul_density_target.insert(0, "Country/Region",
                            ["US", "Iran", "Italy", "South Africa", "Korea, South",
                             "Spain", "Peru", "Australia", "Taiwan*", "Czechia"])

popul_density_with_c = popul_density_target.merge(confirmed_case_target, left_on = "Country/Region", right_on = "Country/Region")

pandemic_severity = popul_density_with_c['4/24/2020'] / popul_density_with_c['Population (2020)'].div(1000000)
popul_density_with_c.insert(4, "Severity", round(pandemic_severity, 2))

# Reference of adding Column with Dictionary values:
# https://cmdlinetips.com/2018/01/how-to-add-a-new-column-to-using-a-dictionary-in-pandas-data-frame/

popul_density_with_c.insert(4, "Country Code", popul_density_with_c["Country/Region"].map(country_dict))
df_severity = popul_density_with_c[['Country Code', 'Severity']]
severity_dict = df_severity.set_index('Country Code')['Severity'].to_dict()
# Reference:
# pandas.DataFrame.to_dict: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_dict.html
# set_index: https://stackoverflow.com/questions/52547805/how-to-convert-dataframe-to-dictionary-in-pandas-without-index

# Severity Degree
# plt.bar(aware_period_in_days.keys(), severity_dict.values())

# Awareness Gap Days
# plt.bar(aware_period_in_days.keys(), aware_period_in_days.values())

dict_for_hist_plot = {}
for i in aware_period_in_days.keys():
    dict_for_hist_plot[i] = (aware_period_in_days[i], severity_dict[i])

df_for_plot = pd.DataFrame(dict_for_hist_plot).transpose()
df_for_plot.columns = ['Awareness', 'Severity']

# Reference of Double y-axis:
# https://stackoverflow.com/questions/24183101/pandas-bar-plot-with-two-bars-and-two-y-axis
# https://stackoverflow.com/questions/9103166/multiple-axis-in-matplotlib-with-different-scales

fig = plt.figure() # Create matplotlib figure

ax = fig.add_subplot() # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

ax.set_ylim(0, 60)
ax2.set_ylim(0, 5000)
width = 0.3

plot_aw = df_for_plot.Awareness.plot(kind = 'bar', color = 'orange', ax = ax, width = width, position = 1)
plot_se = df_for_plot.Severity.plot(kind = 'bar', color = 'blue', ax = ax2, width = width, position = 0)

ax.set_ylabel('Awareness Gap Days')
ax2.set_ylabel('Severity Degree')

fig.legend(loc = 'upper center')
# ax.legend()
# ax2.legend()
plt.show()

# Hypothesis_2

# Reference of Scatterplot with annotation on each data point:
# https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point

x = list(popul_density_with_c['Density (P/Km?)'])
y = list(popul_density_with_c['Severity'])
c = list(popul_density_with_c['Country Code'])

fig, ax = plt.subplots()
ax.scatter(x, y)

for i, txt in enumerate(c):
    ax.annotate(txt, (x[i], y[i]))

plt.xlabel('Population Density')
plt.ylabel('Severity Degree')