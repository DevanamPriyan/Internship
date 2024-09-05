######### TASK 1 #########
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading dataset
file_path = 'G:/internship 2/API_SP.POP.TOTL_DS2_en_csv_v2_2431709.csv'
df = pd.read_csv(file_path, skiprows=4)  # Skip the first few header rows

# Inspecting the dataset to understand its structure
print(df.head())
print(df.columns)

# Select the relevant data for the year 2023
df_2023 = df[['Country Name', '2023']].dropna()

# Renaming columns for easy referencimg
df_2023.columns = ['Country', 'Population_2023']

# Sorting the data by population
df_2023_sorted = df_2023.sort_values(by='Population_2023', ascending=False)

# Bar Chart: Population distribution by country for 2023
plt.figure(figsize=(14, 8))
sns.barplot(x='Country', y='Population_2023', data=df_2023_sorted.head(20))  # Top 20 countries for readability
plt.xticks(rotation=90)
plt.title('Top 20 Countries by Population in 2023')
plt.xlabel('Country')
plt.ylabel('Population')
plt.show()

# Histogram: Distribution of population sizes across all countries in 2023
plt.figure(figsize=(10, 6))
sns.histplot(df_2023['Population_2023'], bins=20, kde=True)
plt.title('Distribution of Population Sizes in 2023')
plt.xlabel('Population')
plt.ylabel('Frequency')
plt.show()

