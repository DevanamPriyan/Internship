######### TASK 2 #########
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'G:/internship 2/Task2/Original/train.csv'
df = pd.read_csv(file_path)

# Information about the dataset
print("Data Overview:")
print(df.head())
print("\nSummary:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Data Cleaning
# Filling the missing values for Age with the median value
df['Age'].fillna(df['Age'].median(), inplace=True)

# Filling up missing values for Embarked with the mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Dropping the Cabin column due to excessive missing values
df.drop(columns=['Cabin'], inplace=True)

# Dropping rows with missing values in Fare.
df.dropna(subset=['Fare'], inplace=True)

# Verifying that there are no missing values left
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Exploratory Data Analysis (EDA)

# Distribution of individual variables
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Survived', data=df)
plt.title('Distribution of Survival')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Pclass', data=df)
plt.title('Distribution of Passenger Class')
plt.xlabel('Pclass')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Sex', data=df)
plt.title('Distribution of Gender')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# Bivariate Analysis - Relationship between two variables
plt.figure(figsize=(10, 6))
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title('Age Distribution by Passenger Class')
plt.xlabel('Pclass')
plt.ylabel('Age')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Pclass')
plt.ylabel('Survival Rate')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Gender')
plt.xlabel('Sex')
plt.ylabel('Survival Rate')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Embarked', y='Survived', data=df)
plt.title('Survival Rate by Port of Embarkation')
plt.xlabel('Embarked')
plt.ylabel('Survival Rate')
plt.show()