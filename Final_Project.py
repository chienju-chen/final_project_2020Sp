"""
IS 590PR Final Project - Spring 2020

Project Topic:
Analysis on how People’s Awareness and Population Density Affect the Severity of COVID-19

Team Member:
Chien-Ju Chen, Yun-Hsuan Chuang

Hypothesis:
1. For countries that people are aware of COVID-19 earlier, the pandemic will be less serious.
2. Pandemic will be more serious in the countries with a higher population density.

Data Source:
1. COVID-19 confirmed cases：https://github.com/CSSEGISandData/COVID-19
2. Google Trends：https://trends.google.com/trends/
3. Countries in the world by population (2020)：https://www.worldometers.info/world-population/population-by-country/

Study Time Period:
Jan. 1st, 2020 - Apr. 24th, 2020
"""

# Import Packages
import numpy as np
import pandas as pd
from pytrends.request import TrendReq
from scipy import signal
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def extract_target_country_case(filename, country_list):
    """
    Given the global confirmed cases data and a list of target country names of which we want to study,
    extract the number of confirmed cases for each target country.
    :param filename: a string indicating the file name of global confirmed cases data
    :param country_list: a list containing the target country names of which we want to study
    :return: a DataFrame containing the target country name and the corresponding daily number of cumulative
    confirmed cases

    >>> filename = 'data/time_series_covid19_confirmed_global.csv'
    >>> country_code_dict = {'Taiwan*':'TW', 'Korea, South':'KR', 'Italy':'IT', 'Australia':'AU'}
    >>> country_list = list(country_code_dict.keys())
    >>> confirmed_case_target = extract_target_country_case(filename, country_list)
    >>> confirmed_case_target['Country/Region']
    0           Italy
    1    Korea, South
    2         Taiwan*
    3       Australia
    Name: Country/Region, dtype: object
    >>> confirmed_case_target['4/24/20']
    0    192994
    1     10718
    2       428
    3      6677
    Name: 4/24/20, dtype: int64
    >>> confirmed_case_target.shape
    (4, 95)
    """
    covid_all = pd.read_csv(filename)
    covid_target = covid_all[covid_all['Country/Region'].isin(country_list)]
    covid_target = covid_target.drop(columns=['Province/State', 'Lat', 'Long'])

    # add up the confirmed case numbers for countries with multiple rows
    for country in country_list:
        covid_multiple_rows = covid_target[covid_target['Country/Region'] == country]
        if covid_multiple_rows.shape[0] > 1:
            covid_target = covid_target.drop(covid_multiple_rows.index)
            covid_single_row = covid_multiple_rows.groupby('Country/Region').sum().reset_index()
            covid_target = covid_target.append(covid_single_row)
        else:
            continue
    covid_target = covid_target.reset_index(drop=True)
    return (covid_target)


def extract_google_trends(timeframe, keywords_dict):
    """
    For Hypothesis 1 that aim to study the relationship between people's awareness of COVID-19 and pandemic
    severity degree, this function is to extract the search trend of COVID-19 related keywords in each country
    from Google Trends given a certain timeframe and a dictionary of each target country's keywords.
    :param timeframe: a string indicating the time period for extracting the keyword search trend from Google Trends
    :param keywords_dict: a dictionary containing the country code of target countries as keys and the corresponding
    list of COVID-19 related keywords represented in each country's official language as values
    :return: a dictionary containing the country code of target countries as keys and the corresponding search trend
    of COVID-19 related keywords in each country in Dataframe form as values

    >>> timeframe_1 = '2020-01-01 2020-01-10'
    >>> keywords_dict_1 = {'TW':['武漢', '冠狀病毒'], 'KR':['우한', '코로나바이러스'], 'US':['Wuhan', 'coronavirus']}
    >>> search_trend_dict = extract_google_trends(timeframe_1, keywords_dict_1)
    >>> search_trend_dict.keys()
    dict_keys(['TW', 'KR', 'US'])
    >>> search_trend_dict['TW'].columns
    Index(['Wuhan', 'coronavirus'], dtype='object')
    >>> search_trend_dict['TW'].index # doctest: +ELLIPSIS
    DatetimeIndex(['2020-01-01', '2020-01-02', ... '2020-01-09', '2020-01-10'],
                  dtype='datetime64[ns]', name='date', freq=None)
    >>> search_trend_dict['KR'].shape
    (10, 2)
    >>> timeframe_2 = '2020-01-01 2020-01-32'
    >>> search_trend_dict = extract_google_trends(timeframe_2, keywords_dict_1)
    Traceback (most recent call last):
    pytrends.exceptions.ResponseError: The request failed: Google returned a response with code 500.
    >>> keywords_dict_2 = {'TE':['武漢', '冠狀病毒'], 'KR':['우한', '코로나바이러스'], 'US':['Wuhan', 'coronavirus']}
    >>> search_trend_dict = extract_google_trends(timeframe_1, keywords_dict_2)
    Traceback (most recent call last):
    pytrends.exceptions.ResponseError: The request failed: Google returned a response with code 400.
    """
    pytrend = TrendReq(hl='en-US', tz=360)
    search_trend_dict = {}
    for key in list(keywords_dict.keys()):
        pytrend.build_payload(kw_list=keywords_dict[key], cat=0, timeframe=timeframe, geo=key)
        df_search_trend = pytrend.interest_over_time().drop(columns=['isPartial'])

        # rename the columns of df_search_trend to keywords in English
        df_search_trend.columns = keywords_dict['US']
        search_trend_dict[key] = df_search_trend
    return(search_trend_dict)


"""
Reference of smoothing: 
https://plotly.com/python/smoothing/
"""
def find_notable_date(confirmed_case_dict, smooth_window_len=15):
    """
    Given the daily number of confirmed cases of target countries, find the date on which COVID-19 pandemic
    became notable in each country. The notable date is defined as the date on which the maximum change in
    the number of daily confirmed cases happened.
    :param confirmed_case_dict: a dictionary containing the country code of target countries as keys and the
    corresponding COVID-19 daily confirmed cases data in Dataframe form as values
    :param smooth_window_len: a numeric indicating the window length for smoothing the curve of confirmed cases
    :return: a dictionary containing the country code of target countries as keys and the corresponding notable
    date in string form as values

    >>> dates = ['2020-01-01','2020-01-02','2020-01-03','2020-01-04','2020-01-05']
    >>> dates_datetime = [datetime.strptime(d,'%Y-%m-%d') for d in dates]
    >>> confirmed_case_dict = {'TW': pd.DataFrame({'Confirmed Case':[0,2,10,11,14]}, index=dates_datetime)}
    >>> notable_date_dict = find_notable_date(confirmed_case_dict,5)
    >>> notable_date_dict.keys()
    dict_keys(['TW'])
    >>> notable_date_dict['TW']
    '2020-01-03'
    """
    notable_date_dict = {}
    for key in confirmed_case_dict:
        # smooth the confirmed cases curve before finding the notable date to avoid the fluctuations
        # in the curve leading to an unreasonable notable date
        confirmed_case = confirmed_case_dict[key]['Confirmed Case']
        confirmed_case_smooth = pd.Series(signal.savgol_filter(confirmed_case, smooth_window_len, 3), index=confirmed_case.index)
        # find the notable date with the maximum change in daily increased case
        daily_case_change_smooth = confirmed_case_smooth - 2*confirmed_case_smooth.shift(1) \
                                   + confirmed_case_smooth.shift(2)
        notable_date_dict[key] = daily_case_change_smooth.idxmax().strftime("%Y-%m-%d")
    return (notable_date_dict)


"""
Reference:
(1) sns.color_palette: https://seaborn.pydata.org/tutorial/color_palettes.html
(2) Add boxes on a plot: https://matplotlib.org/3.2.1/gallery/shapes_and_collections/fancybox_demo.html
(3) matplotlib.dates(mdates): https://www.earthdatascience.org/courses/use-data-open-source-python/use-time-series
                              -data-in-python/date-time-types-in-pandas-python/customize-dates-matplotlib-plots-python/
"""
def plot_KWsearch_case_trends(GT_case_trend, notable_date, country_code):
    """
    Given the keyword search and confirmed cases trends data, notable date, and country code of a country,
    plot search trend of each keyword and the confirmed cases number over time for that country on the same plot
    :param GT_case_trend: a DataFrame containing the keyword search and confirmed cases trends of a country
    :param notable_date: a string indicating the pandemic notable date of a country
    :param country_code: a string indicating the country code of a country
    :return: None
    """
    t = GT_case_trend.index.values
    confirmed_case = GT_case_trend['Confirmed Case']

    title = "Keyword Search Trends & Number of Confirmed Cases Over Time_" + country_code
    label = list(GT_case_trend.columns)[:-1]
    color = sns.color_palette("husl", len(label))

    # plot the search trend of each keyword
    pd.plotting.register_matplotlib_converters()
    fig, ax1 = plt.subplots(figsize=(25, 10))
    for i in range(len(label)):
        ax1.plot(t, GT_case_trend.iloc[:, i], color=color[i], label=label[i], linewidth=3)

    # plot the number of confirmed cases & mark the notable date
    ax2 = ax1.twinx()
    ax2.plot(t, confirmed_case, color='red', label='# of Confirmed Cases', linewidth=5)
    ax2.axvline(notable_date, color='blue', ls='--', lw=3)
    bbox = dict(fc='white', ec='blue', alpha=0.6)
    ax2.text(notable_date, confirmed_case[-1] * 0.97, notable_date, color='blue', fontsize=20, bbox=bbox, ha='center')

    # adjust labeling on the plot
    date_form = mdates.DateFormatter("%Y-%m-%d")
    ax1.xaxis.set_major_formatter(date_form)
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax1.set_ylabel('Popularity of Keyword Search', fontsize=20)
    ax2.set_ylabel('Number of Confirmed Cases', fontsize=20, color='red')
    ax1.tick_params(axis='x', labelsize=15, labelrotation=45)
    ax1.tick_params(axis='y', labelsize=15)
    ax2.tick_params(axis='y', labelcolor='red', labelsize=15)
    ax1.grid(linewidth=0.5, color='gray')
    fig.legend(fontsize=15, loc=(0.07, 0.7))
    plt.title(title, fontsize=25)
    plt.show()


def find_search_peak_date(search_trend_dict: dict, most_popular_keywords_dict: dict) -> dict:
    """
    To decide the date of each country, find the date when the most popular keyword reach its search peak.
    The return value is in a dictionary type, with country codes as keys and search peak dates as values.
    :param search_trend_dict: dictionary that stores the search trend from Google Trends
    :param most_popular_keywords_dict: dictionary that stores the most popular keyword of each country
    :return: dictionary with country code as keys and search peak date as values

    >>> dates = ['2020-01-01','2020-01-02','2020-01-03','2020-01-04','2020-01-05']
    >>> dates_datetime = [datetime.strptime(d,'%Y-%m-%d') for d in dates]
    >>> search_trend_dict_DT = {'TW': pd.DataFrame({'Wuhan':[0,35,50,95,20], 'coronavirus':[14,43,65,33,56]}, index=dates_datetime)}
    >>> most_popular_keywords_dict_DT = {'TW': 'Wuhan'}
    >>> search_peak_dict = find_search_peak_date(search_trend_dict_DT, most_popular_keywords_dict_DT)
    >>> search_peak_dict['TW']
    Timestamp('2020-01-04 00:00:00')
    """
    search_peak_dict = {}
    for key in search_trend_dict:
        search_peak_dict[key] = search_trend_dict[key][most_popular_keywords_dict[key]].idxmax()
    return(search_peak_dict)


def popul_density_target_country(filename: str, country_list_popul: list, country_dict_popul: dict) -> pd.DataFrame:
    """
    For Hypothesis 2 that aim to investigate the relationship between population density and severity degree,
    this function is to get the population density of each country from specific file.
    :param filename: a string combining file path and file name
    :param country_list_popul: a list of target country names for extracting population data
    :param country_dict_popul: a dictionary for referencing different country names in multiple data sources
    :return: a DataFrame containing country name, density and total number of population

    >>> filename_DT_1 = 'data/Countries in the world by population (2020).csv'
    >>> country_dict_popul_DT_1 = {'Taiwan*':'Taiwan', 'Korea, South':'South Korea', 'Italy':'Italy'}
    >>> country_list_popul_DT_1 = list(country_dict_popul_DT_1.values())
    >>> popul_density_target_country(filename_DT_1, country_dict_popul_DT_1, country_dict_popul_DT_1)
       Country (or dependency)  Density (P/Km?)  Population (2020)
    22                   Italy              206           60461826
    27             South Korea              527           51269185
    56                  Taiwan              673           23816775
    """
    popul_density = pd.read_csv(filename)
    popul_target = popul_density[popul_density['Country (or dependency)'].isin(country_dict_popul.values())]
    popul_target = popul_target[['Country (or dependency)', 'Density (P/Km?)', 'Population (2020)']]
    return popul_target



if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------------------------
    # Extract the number of confirmed cases for each target country
    filename = "data/time_series_covid19_confirmed_global.csv"
    country_code_dict = {'Taiwan*': 'TW', 'Korea, South': 'KR', 'Italy': 'IT', 'Spain': 'ES', 'Czechia': 'CZ',
                         'US': 'US', 'Peru': 'PE', 'Iran': 'IR', 'Australia': 'AU', 'South Africa': 'ZA'}
    country_list = list(country_code_dict.keys())
    confirmed_case_target = extract_target_country_case(filename, country_list)

    # Transpose confirmed_case_target for future merge with the search trend of COVID-19 related keywords
    df_country = pd.DataFrame(country_code_dict.items(), columns=['Name', 'Code'])
    confirmed_case_target = pd.merge(confirmed_case_target, df_country, how='left',
                                     left_on='Country/Region', right_on='Name')
    confirmed_case_tr = confirmed_case_target.transpose()

    # Set country codes as the column names of the transposed dataframe confirmed_case_tr
    confirmed_case_tr.columns = confirmed_case_tr.loc['Code']
    confirmed_case_tr = confirmed_case_tr.drop(['Country/Region', 'Name', 'Code'])

    # Set date as the index of confirmed_case_tr
    confirmed_case_tr.index = pd.to_datetime(confirmed_case_tr.index)

    # ----------------------------------------------------------------------------------------------------------
    # Extract search trend of keywords from Google Trends for each target country
    timeframe = '2020-01-01 ' + confirmed_case_tr.index[-1].strftime('%Y-%m-%d')
    keywords_dict = {'TW': ['武漢', '冠狀病毒', '武漢肺炎', '肺炎', '新冠肺炎'],
                     'KR': ['우한', '코로나바이러스', '우한 폐렴', '폐렴', '신종 코로나바이러스'],
                     'IT': ['Wuhan', 'coronavirus', 'Polmonite di Wuhan', 'Polmonite', 'covid'],
                     'ES': ['Wuhan', 'coronavirus', 'neumonía de Wuhan', 'neumonía', 'covid'],
                     'US': ['Wuhan', 'coronavirus', 'Wuhan coronavirus', 'pneumonia', 'covid'],
                     'IR': ['ووهان', 'کروناویروس', 'ووهان کروناویروس', 'سینه‌پهلو', 'کووید'],
                     'PE': ['Wuhan', 'coronavirus', 'neumonía de Wuhan', 'neumonía', 'covid'],
                     'AU': ['Wuhan', 'coronavirus', 'Wuhan coronavirus', 'pneumonia', 'covid'],
                     'CZ': ['Wu-chan', 'koronavirus', 'wuchanský koronavirus', 'zápal plic', 'covid'],
                     'ZA': ['Wuhan', 'coronavirus', 'Wuhan Koronavirus', 'Longontsteking', 'covid']}

    search_trend_dict = extract_google_trends(timeframe, keywords_dict)

    # ----------------------------------------------------------------------------------------------------------
    # Merge confirmed_case_tr with Google Trends extracting results and save the result back to search_trend_dict
    """
    Reference of merging dataframes by index:
    https://thispointer.com/pandas-how-to-merge-dataframes-by-index-using-dataframe-merge-part-3/
    """
    for key in search_trend_dict:
        df_merged = pd.merge(search_trend_dict[key], confirmed_case_tr[key], how='left', left_index=True,
                             right_index=True)
        df_merged[key] = df_merged[key].fillna(0)
        search_trend_dict[key] = df_merged.rename(columns={key: "Confirmed Case"})

    # ----------------------------------------------------------------------------------------------------------
    # Find the date on which COVID-19 pandemic became notable in each country
    notable_date_dict = find_notable_date(search_trend_dict)

    # ----------------------------------------------------------------------------------------------------------
    # Plot keyword search trends vs. confirmed cases number over time for each country
    for key in search_trend_dict:
        trend = search_trend_dict[key]
        notable_date = notable_date_dict[key]
        plot_KWsearch_case_trends(trend, notable_date, key)

    # ----------------------------------------------------------------------------------------------------------
    # Find the search peak date for evaluating the awareness level of each country
    most_popular_keywords_dict = {'TW': 'Wuhan', 'KR': 'Wuhan', 'IT': 'coronavirus', 'ES': 'coronavirus',
                                  'CZ': 'coronavirus', 'US': 'coronavirus', 'PE': 'coronavirus',
                                  'IR': 'coronavirus', 'AU': 'coronavirus', 'ZA': 'coronavirus'}

    search_peak_dict = find_search_peak_date(search_trend_dict, most_popular_keywords_dict)

    # Calculate and save the awareness level of each country to aware_period_in_days dictionary
    aware_period_in_days = {}
    for key in search_trend_dict:
        notable_date_datetime = datetime.strptime(notable_date_dict[key], '%Y-%m-%d')
        aware_days = datetime.date(notable_date_datetime) - datetime.date(search_peak_dict[key])
        aware_period_in_days[key] = aware_days.days

    # ----------------------------------------------------------------------------------------------------------
    # Extract the total number of population of the target countries
    filename = 'data/Countries in the world by population (2020).csv'
    country_dict_popul = {'Taiwan*': 'Taiwan', 'Korea, South': 'South Korea', 'Italy': 'Italy',
                          'Spain': 'Spain', 'Czechia': 'Czech Republic (Czechia)', 'US': 'United States',
                          'Peru': 'Peru', 'Iran': 'Iran', 'Australia': 'Australia', 'South Africa': 'South Africa'}
    country_list_popul = list(country_dict_popul.values())
    popul_density_target = popul_density_target_country(filename, country_dict_popul, country_dict_popul)

    # Merge confirmed_case_target with population data to evaluate pandemic severity degree
    popul_density_target.insert(0, "Country/Region",
                                ["US", "Iran", "Italy", "South Africa", "Korea, South",
                                 "Spain", "Peru", "Australia", "Taiwan*", "Czechia"])
    popul_density_with_c = popul_density_target.merge(confirmed_case_target, left_on="Country/Region",
                                                      right_on="Country/Region")

    # Calculate the pandemic severity degree of each country by definition: confirmed cases per million people
    # The number of total confirmed cases is the last date of our time period: Apr. 24th, 2020
    pandemic_severity = popul_density_with_c['4/24/20'] / popul_density_with_c['Population (2020)'].div(1000000)
    popul_density_with_c.insert(4, "Severity", round(pandemic_severity, 2))

    # Insert the country code of target countries as index value
    """
    Reference of adding column with dictionary values: 
    https://cmdlinetips.com/2018/01/how-to-add-a-new-column-to-using-a-dictionary-in-pandas-data-frame/
    """
    popul_density_with_c.insert(4, "Country Code", popul_density_with_c["Country/Region"].map(country_code_dict))

    """
    Reference:
    (1) pandas.DataFrame.to_dict: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_dict.html
    (2) set_index: https://stackoverflow.com/questions/52547805/how-to-convert-dataframe-to-dictionary-in-pandas-without-index
    """
    df_severity = popul_density_with_c[['Country Code', 'Severity']]
    severity_dict = df_severity.set_index('Country Code')['Severity'].to_dict()

    # Create a dictionary with country code as keys to store the awareness level and severity degree of each country
    dict_for_hist_plot = {}
    for i in aware_period_in_days.keys():
        dict_for_hist_plot[i] = (aware_period_in_days[i], severity_dict[i])

    df_for_plot = pd.DataFrame(dict_for_hist_plot).transpose()
    df_for_plot.columns = ['Awareness', 'Severity']

    # ----------------------------------------------------------------------------------------------------------
    # Plot the outcome of Hypothesis 1 (Awareness Level vs. Severity Degree) into bar plot
    """
    Reference of Double y - axis:
    (1) https://stackoverflow.com/questions/24183101/pandas-bar-plot-with-two-bars-and-two-y-axis
    (2) https://stackoverflow.com/questions/9103166/multiple-axis-in-matplotlib-with-different-scales
    """
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    ax.set_ylim(0, 60)
    ax2.set_ylim(0, 5000)

    width = 0.3

    # for plotting purpose, convert data from df to arrays
    country = []
    for key in dict_for_hist_plot:
        country.append(key)
    n = df_for_plot.shape[0]

    # for plotting purpose, set data for x, y1, y2
    x = np.arange(len(country))
    y1 = np.zeros(n)
    y2 = np.zeros(n)
    for i in range(n):
        y1[i] = df_for_plot['Awareness'][i]
        y2[i] = df_for_plot['Severity'][i]

    aw = ax.bar(x - width / 2, y1, color='orange', width=width)
    se = ax2.bar(x + width / 2, y2, color='blue', width=width)
    ax.set_xticks(x)
    ax.set_xticklabels(country)
    ax.set_ylabel('Awareness Level')
    ax2.set_ylabel('Severity Degree')
    ax.legend([aw, se], ['Awareness', 'Severity'])
    plt.show()

    # ----------------------------------------------------------------------------------------------------------
    # Plot the outcome of Hypothesis 2 into scatter plot
    """
    Reference of Scatter Plot with annotation on each data point:
    https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point
    """
    x = list(popul_density_with_c['Density (P/Km?)'])
    y = list(popul_density_with_c['Severity'])
    c = list(popul_density_with_c['Country Code'])

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    for i, txt in enumerate(c):
        ax.annotate(txt, (x[i], y[i]))

    plt.xlabel('Population Density\n(people / Km2)')
    plt.ylabel('Severity Degree\n(Confirmed Cases / Million Pople)')
    plt.show()
