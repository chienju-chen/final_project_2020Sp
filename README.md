# IS 590PR Final Project - Spring 2020


## Topic

Analysis on how People’s Awareness and Population Density Affect the Severity of COVID-19


## Team Member

Chien-Ju Chen, Yun-Hsuan Chuang (Github ID: chienju-chen, katyyhc)


## Introduction

This project is aiming to explore possible causes for the different severity of COVID-19 pandemic between countries. We consider **people’s awareness** and **population density** as possible factors and study whether they would affect the severity of pandemic between different countries.


## Hypothesis
1. For countries that people are **aware\* of COVID-19 earlier**, the pandemic will be **less serious**.
2. Pandemic will be **more serious** in the countries with a **higher population density**.

*Assumption: People’s awareness of COVID-19 can be reflected by the search of COVID-19 related keywords on Google, where the search trend for each keyword can be extracted from Google Trends.


## Data Source:

1. GitHub repository of CSEE at Johns Hopkins University (COVID-19 confirmed cases data)： <br>https://github.com/CSSEGISandData/COVID-19

2. Google Trends (awareness data)：<br>https://trends.google.com/trends/
 
3. Worldometer (population density data)： <br>https://www.worldometers.info/world-population/population-by-country/


## Method

### Study Period: Jan. 1st, 2020 - Apr. 24th, 2020

The start date, Jan. 1st, 2020, is selected as the day after WHO China Country Office informed the cases of pneumonia of unknown cause detected in Wuhan City.

### Target Country:

Ten target countries are picked from 5 continents around the world, including both countries with severe and less severe pandemic for studying the two hypotheses:
* United States
* Peru
* Taiwan
* South Korea
* Iran
* Italy
* Spain
* Czech Republic
* Australia
* South Aftica

### COVID-19 Related Keyword:

Five keywords are selected for extracting the search trend from Google Trends, which is used for representing people’s awareness of COVID-19. The keywords are listed below with the reason of being selected:

* “Wuhan”, “coronavirus”, and “Wuhan coronavirus”:
<br>Before WHO announced COVID-19 as its official name, people’s understanding of this disease is that it started from Wuhan, a city in China, and is caused by the coronavirus.

* “pneumonia”: 
<br>Most COVID-19 confirmed cases show symptoms of pneumonia. 

* “covid”: 
<br>Only the alphabetic part of “COVID-19” is used because we found that while some people search the word with a hyphen before the number 19, others search it without a hyphen.

The five keywords are translated to the official language of each target country(shown in the following figure) by using Wikipedia. We then search each translated keyword on Google to ensure that it is the actually keyword used for search. 

![image](https://github.com/chienju-chen/final_project_2020Sp/blob/master/plots/COVID-19_related_keywords.png)

### COVID-19 Awareness Trend:

The COVID-19 awareness trend is visualized by plotting the search trend of each keyword in the same country as the plot shown below, which is an illustration of the awareness trend in the US.

![image](https://github.com/chienju-chen/final_project_2020Sp/blob/master/plots/Keyword%20Search%20Trends_US.png)

### Definition of Terms:

For a better demonstration of the terms used in this project, we plot the confirmed cases curve of US as the bold red curve on the plot of awareness trend in the US:

![image](https://github.com/chienju-chen/final_project_2020Sp/blob/master/plots/Keyword%20Search%20Trends%20_%20Number%20of%20Confirmed%20Cases%20Over%20Time_US.png)

* Severity of Pandemic: 
<br>The number of cumulative confirmed cases per million people in a country on Apr. 24th, 2020.

* Notable Date: 
<br>The date on which the maximum change in the number of daily confirmed cases happens in a country, which is represented as the blue dashed line in the above figure. It indicates the point where the pandemic became notable in a country, which we set as the reference point for comparing how early people in different countries were aware of COVID-19.

* Keyword Search Peak:
<br>The point where the highest search popularity among the keywords happens before the notable date, which is represented as the red circle on the above figure. It indicates the timing when a considerable amount of people in a country were aware of COVID-19.

* Awareness Level:
<br>The number of days after the keyword search peak and before the notable date in a country, which is represented as the gap between the dashed red line and the dashed blue line in the above figure. The wider the gap is, the higher the awareness level is.


## Outcome & Analysis

### Hypothesis 1
For countries that people are **aware of COVID-19 earlier**, the pandemic will be **less serious**.

![image](https://github.com/chienju-chen/final_project_2020Sp/blob/master/plots/bar_plot_hyp.1.png)

The bar plot shows the comparison of Awareness Level and Severity Degree of each country. Hypothesis 1 is shown to be true for 7 countries (TW, KR, IT, ES, US, IR, PE), while the relationship is not very obvious for 3 other countries (AU, CZ, ZA).

* Taiwan(TW) and South Korea(KR) are the earliest aware countries with low severity degree.
* Iran(IR) and Peru(PE) are the 2-tier early aware countries with medium severity degree.
* Italy(IT), Spain(ES) and the United States(US) are the most severe countries with low awareness level.


### Hypothesis 2
Pandemic will be **more serious** in the countries with a **higher population density**.

![image](https://github.com/chienju-chen/final_project_2020Sp/blob/master/plots/scatter_plot_hyp.2.png)

The scatter plot shows the relationship between population density and severity degree of each country. The outcomes of 4 countries (AU, IR, PE, IT) are consistent with Hypothesis 2, while other 4 of them (TW, KR, ES, US) are not. The left 2 countries (CZ, ZA) are still not obvious in trend and might be affected by other factors.

* Australia(AU), Iran(IR) and Peru(PE) are shown to be low in both population density and severity degree. Italy(IT) is highly populated and also severity in pandemic.
* Although Spain(ES) and the United States(US) are low populated, pandemic in these two countries is still serious. 
* Taiwan(TW) and South Korea(KR) are highly populated countries while the severity degree is very low.

In addition, we noticed that the awareness level of Spain(ES) and United States(US) are both very low while that of Taiwan(TW) and Korea(KR) are really high. For these four countries that are inconsistent with Hypothesis 2, their severity degree can be better explained by the awareness level.

### Other Factors

The impact of both awareness level and population density are not very obvious on Czech Republic(CZ) and South Africa(ZA). As a result, we would consider other factors that might affect their pandemic.

Czech has relatively high population density and didn’t aware early, but the pandemic seemed to be under well control. One possible explanation is its policy from the government to mandatorily ask people to wear masks.

For South Africa, basically, the numbers of travel and passengers are low. According to ACI(Airport Council International), the passengers of all the Africa’s airports is only composed 2% of worldwide, which may be the reason for the low severity degree of South Africa.



## Conclusion
According to our analysis, both high awareness level and low population density are relative to a low severity of COVID-19 pandemic. However, the relationship between awareness level and pandemic severity is stronger, which makes us consider awareness level a stronger factor and better explainable to the severity of COVID-19. Therefore, we conclude that both hypotheses are valid, but hypothesis 1 shows a stronger relationship.
