## final_project

### This project is aiming to explore possible causes for the different severity of COVID-19 pandemic between countries.

We consider people’s awareness and population density as possible factors and study whether they would affect the severity of pandemic between different countries.


To scale people’s awareness, we assume the search of keywords related to COVID-19 on Google as the degree of awareness.

The severity of pandemic is defined as the number of confirmed cases per million people.



## Hypothesis: 

1. For countries that people are **aware of COVID-19 earlier**,  the pandemic will be **less serious**.

2. Pandemic will be **more serious** in the countries with a **higher population density**.

 

## Data Source:

COVID-19 confirmed cases：https://github.com/CSSEGISandData/COVID-19

Google Trend：https://trends.google.com/trends/

Countries in the world by population (2020)：https://www.worldometers.info/world-population/population-by-country/

NOAA(National Centers for Environmental Information)：https://www.ncdc.noaa.gov/cdo-web/datasets



## Outcome & Analysis

### **Hypothesis 1**
For countries that people are **aware of COVID-19 earlier**, the pandemic will be **less serious**.

![image](https://github.com/chienju-chen/final_project_2020Sp/blob/master/plots/bar_plot_hyp.1.png)

The bar plot shows the comparison of Awareness Level and Severity Degree of each country. Hypothesis 1 is shown to be true for 7 countries (TW, KR, IT, ES, US, IR, PE), while the relationship is not very obvious for 3 other countries (AU, CZ, ZA).

* Taiwan(TW) and South Korea(KR) are the earliest aware countries with slight severity degree.
* Iran(IR) and Peru(PE) are the 2-tier early aware countries with smooth severity degree.
* Italy(IT), Spain(ES) and the United States(US) are the most severe countries with low awareness level.


### Hypothesis 2
Pandemic will be **more serious** in the countries with a **higher population density**.

![image](https://github.com/chienju-chen/final_project_2020Sp/blob/master/plots/scatter_plot_hyp.2.png)

The scatter plot shows the relationship between Population Density and Severity Degree of each country. The outcomes of 3 countries (AU, IR, PE) are consistent with Hypothesis 2, while 5 of them (TW, KR, IT, ES, US) are opposite to Hypothesis 2. The left 2 countries (CZ, ZA) are still not obvious in trend and might be affected by other factors.

* Australia(AU), Iran(IR) and Peru(PE) are shown to be low in both population density and severity degree.
* Taiwan(TW) and South Korea(KR) are highly populated countries while the severity degree is very low, which could be explained by Hypothesis 1-- both of them are early aware.
* Italy(IT), Spain(ES) and the United States(US) are how populated countries, however, the pandemic is still severe.


### Other Factors

The impact of both Hypothesis 1 and 2 seem to be not very obvious on Czech Republic(CZ) and South Africa(ZA). As a result, we would consider other factors that might affect their pandemic. 

Czech has relatively high population density and didn’t aware early, but the pandemic seemed to be under well control. One special thing of Czech is its policy from the government to mandatorily ask people to wear masks. “#Masks4all” indeed effectively slowed down the outbreak in Czech.

For South Africa, basically, the numbers of travel and passengers are low. According to ACI(Airport Council International), the passengers of all the Africa’s airports is only composed 2% of worldwide. 



## Conclusion
According to our analysis, both Hypothesis 1 and 2 are relative to the severity of COVID-19 pandemic. The relationship between Awareness Level and pandemic is stronger than the relationship between Population Density and pandemic. As a result, Hypothesis 1 is stronger factor and is able to better explain the severity of COVID-19.
