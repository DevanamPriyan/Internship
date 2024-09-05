######### TASK 3 #########
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

file_path = 'G:/internship 2/Task 3/bank.csv'

# Try loading the CSV file with different delimiters if unsure of the delimiter(here we see it is semicolon)
data = pd.read_csv(file_path, delimiter=';')  # Assuming a semicolon delimiter

# Define the column names as provided
column_names = [
    'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
    'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y'
]

# Assigning column names
data.columns = column_names

# Displaying first few rows of the formatted dataset
print(data.head())

# Saving the cleaned dataset to a new CSV file
data.to_csv('G:/internship 2/Task 3/formatted_bank.csv', index=False)

#Building Classifier

file_path = 'G:/internship 2/Task 3/formatted_bank.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Encode categorical variables using LabelEncoder
label_encoders = {}
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']

for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Define features (X) and target (y)
X = data.drop('y', axis=1)
y = data['y']

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Print the accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
