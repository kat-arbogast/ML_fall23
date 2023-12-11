# ML_fall23
This contains all of the code and data used to create the semester long machine learning project. This project will focus on wildfires in the United States. 


## Introduction

Wildfires represent a formidable natural disaster, instilling fear due to their capacity to wreak havoc. These infernos possess the destructive potential to obliterate homes, devastate ecosystems, ravage agricultural lands, and tragically claim lives. Consequently, it is imperative for communities to invest in a comprehensive understanding of wildfires and enhance their predictive capabilities to safeguard both their residents and the surrounding environment. Regrettably, in recent decades, wildfires have surged in severity, unleashing unprecedented devastation upon our landscapes, properties, and human lives. To address this escalating crisis, proactive measures must be taken to delve into the different factors that influence wildfire dynamics, discerning how human activities and natural phenomena interact to either worsen or mitigate the crisis. The core objective of this project is to analyze trends and construct models that highlight and understand the intricacies of external factors, such as weather patterns and human interventions, on the escalating severity of wildfires. Hopefully these efforts will create valuable insights to then develop effective strategies for wildfire prevention and response.

Current efforts made in this area of focus tend to be largely government agencies as this impacts the US as a whole, and also it requires ample research and funding. For example one of the largest databases and analytics groups for wildfires in the United States is the Federal Emergency Management Agency (FEMA) with their National Fire Incident Reporting System (NFIRS), which is accessible to the public here. Another important group to mention is the National Oceanic and Atmospheric Association - National Centers for Environmental Information where one can find summary information from the National Interagency Fire Center (NIFC). These groups are collecting and providing current and historic information on fire incidents. Not only are these organizations impactful in the work they do researching and reporting information on wildfires, they also are multifaceted and help the community gain knowledge about a variety of natural disasters. This machine learning project aims to learn and grow from the current work, and attempt to pull in other resources such as the US Drought Monitor data (which is produced through a partnership between the National Drought Mitigation Center at the University of Nebraska-Lincoln, the United States Department of Agriculture and the National Oceanic and Atmospheric Administration) to develop strong understanding and models for the wildfires in the Unites States. 

The National Fire Protection Association (NFPA) plays a crucial role in sharing knowledge about fires in the United States. The organization regularly publishes comprehensive reports that shed light on various aspects of fires within the country. These reports encompass statistical analyses, economic implications, and delve into the complex issues such as the "Invisible Fire Problem," which specifically addresses the unique fire risks faced by homeless communities as compared to those in traditional housing situations. In recent years, the NFPA has unearthed significant findings regarding fires in the United States. To illustrate, consider the alarming statistics related to heating equipment fires, which were responsible for a staggering toll: "heating equipment fires resulted in an estimated 480 civilian deaths; 1,370 civilian injuries; and one billion dollars in direct property damage each year from 2016 to 2020" (Campbell, 2022). Furthermore, these fires constituted a significant proportion of home structure fires, accounting for approximately 13 percent of such incidents during the 2016-2020 period.

Apart from heating equipment fires, intentional and lightning-induced fires have also proven to be exceptionally destructive and deadly. According to Marty Ahrens (2013), during the period from 2007 to 2011, U.S. local fire departments were responding to an estimated average of 22,600 fires each year ignited by lightning. These fires, on average, resulted in nine civilian deaths, 53 civilian injuries, and a staggering $451 million in direct property damage annually. Similarly, intentional fires have exacted a heavy toll on lives and property, with Richard Campbell (2021) highlighting their devastating impact. These fires were associated with an estimated 400 civilian deaths, 950 civilian injuries, and an economic burden of $815 million in direct property damage annually. In essence, the NFPA's rigorous research and reporting efforts provide invaluable insights into the multifaceted nature of fires in the United States, encompassing their causes, consequences, and the specific vulnerabilities faced by various communities.

Insights derived from wildfire data can offer valuable information on trends and patterns, thereby enabling the development of impactful action plans to mitigate the destructive effects of wildfires. Thus, a machine learning project stands as a promising avenue for further exploration and understanding. Leveraging both text and numerical data spanning from 2000 to 2022, this project will employ a range of machine learning techniques, including unsupervised methods such as k-means and hierarchical clustering to identify patterns and groupings within the data. Additionally, supervised learning models like Naive Bayes, Support Vector Machines, Decision Trees, and Linear Regression analyses will be employed to predict and understand the factors influencing the severity and occurrence of wildfires. By integrating data sources from multiple organizations and resources (including but not limited to: FEMA, NFPA, and the National Drought Mitigation Center), this project aims to build upon existing knowledge and contribute to the development of robust strategies for wildfire prevention and response. As the project progresses, it will delve into the intricate relationships between weather patterns, human interventions, and the evolving landscape of wildfires, seeking to provide insights that go beyond the capabilities of traditional approaches.


## Focused Topic Questions

1. How has the severity of US fires changed over time?

2. Is there a "fire season" like people say? 

3. How might seasonal changes within a year impact fires and their severity?

4. How may human intervention have improved or worsened wildfires in the US?

5. What are the most common human based causes for fires?

6. What are the most common environmental cause for fires?

7. How have climate changes impacted wildfires and their severity? 

8. How effective can different models be in predicting a future year's fire statistics? 

9. What are people's sentiments around wildfires and other climate issues?

10. In terms of severity, how has fires changed in both environmental damage and human life impacts (and why might they differ)?