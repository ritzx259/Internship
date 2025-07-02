import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

df = pd.read_csv("/Users/ritzmk/PycharmProjects/pythonProject/internship/Titanic-Dataset.csv")

print(df.describe(include='all'))
print(df.info())
print(df.isnull().sum())

sns.histplot(df['Age'].dropna(), kde=True)
plt.title('Age Distribution')
plt.show()

sns.histplot(df['Fare'].dropna(), kde=True)
plt.title('Fare Distribution')
plt.show()

sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Age vs Survived')
plt.show()

sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title('Fare by Class')
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

sns.pairplot(df[['Age', 'Fare', 'Pclass', 'Survived']].dropna(), hue='Survived')
plt.show()

fig = px.scatter(df, x='Age', y='Fare', color='Survived', title='Age vs Fare by Survival')
fig.show()

fig = px.histogram(df, x='Sex', color='Survived', barmode='group', title='Sex vs Survival')
fig.show()

fig = px.box(df, x='Pclass', y='Age', color='Survived', title='Age Distribution by Class and Survival')
fig.show()
