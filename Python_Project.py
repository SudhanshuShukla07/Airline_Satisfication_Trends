import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


d1=pd.read_csv("D:\\OneDrive\\Desktop\\python\\airline.csv")
print(d1)
print("Information of Dataset: \n")
print(d1.info())
print("Descriptiontion of Dataset: \n")
print(d1.describe())
print("Missing value or not?? \n")
print(d1.isnull().sum())
d1['Arrival Delay']=d1['Arrival Delay'].fillna(112)
print("1st 10 rows of datset: \n")
print(d1.head(10))
print("last 10 rows of datset: \n")
print(d1.tail(15))
print("Shape of DataSet:  \n",d1.shape)
print("Columns of DataSet:  \n",d1.columns)
print("Datatype of DataSet: \n ",d1.dtypes)
print("veryfing Having any null value or not?? \n")
print(d1.isnull().sum())
d1.to_csv("cleaned_airline.csv", index=False)
print("Saved succesfully")



plt.figure(figsize=(12, 6))
sns.countplot(data=d1, x='Satisfaction', hue='Satisfaction', palette='coolwarm',legend=False)
plt.title("Overall Satisfaction Levels")
plt.xlabel("Satisfaction")
plt.ylabel("Number of Passengers")
plt.show()

satis_by_class = d1[d1['Satisfaction'] == 'Satisfied'].groupby('Class').size() / d1.groupby('Class').size() * 100
satis_by_class = satis_by_class.sort_values(ascending=False).reset_index()
satis_by_class.columns = ['Class', 'Satisfied (%)']



plt.figure(figsize=(10, 6))
plt.pie(satis_by_class['Satisfied (%)'], labels=satis_by_class['Class'],
        autopct='%1.1f%%', colors=sns.color_palette('Set2'))
plt.title("Satisfied Passengers by Travel Class", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


df_line = d1.groupby('Age')['Flight Distance'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_line, x='Age', y='Flight Distance', color='purple')
plt.title("Average Flight Distance by Age")
plt.xlabel("Age")
plt.ylabel("Average Flight Distance")
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 6))
sns.countplot(data=d1, x='Customer Type', hue='Satisfaction', palette='viridis')
plt.title("Satisfaction by Customer Type")
plt.xlabel("Customer Type")
plt.ylabel("Number of Passengers")
plt.legend(title='Satisfaction')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=d1, x='Satisfaction', y='Check-in Service',hue='Satisfaction', palette='pastel',legend=False)
plt.title("Check-in Experience vs Satisfaction")
plt.show()


plt.figure(figsize=(10, 8))
sns.boxplot(data=d1, x='Satisfaction', y='Online Boarding',hue='Satisfaction', palette='Set3',legend=False)
plt.title("Online Boarding Experience vs Satisfaction")
plt.show()


features = [
    'In-flight Wifi Service',
    'Food and Drink',
    'Seat Comfort',
    'In-flight Entertainment',
    'Online Boarding',
    'Leg Room Service',
    'Check-in Service',
    'Cleanliness',
    'Gate Location'
]


df_grouped = d1.groupby('Satisfaction')[features].mean().T

df_grouped.plot(kind='bar', figsize=(12, 6), colormap='tab10')
plt.title("Average Ratings of In-Flight Services by Satisfaction")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)
plt.legend(title='Satisfaction')
plt.tight_layout()
plt.show()

dis_satisfied = d1[d1['Satisfaction'] == 'Neutral or Dissatisfied']


dis_sat = dis_satisfied[features].mean().sort_values()

plt.figure(figsize=(10, 6))
dis_sat.plot(kind='barh', color='crimson')
plt.title("Top Factors Driving Dissatisfaction")
plt.xlabel("Average Rating (Lower = Worse)")
plt.tight_layout()
plt.show()

# Heatmap of correlation matrix
plt.figure(figsize=(12, 8))
corr = d1.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.8)
plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.show()






