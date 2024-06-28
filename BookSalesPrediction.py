# Importing libraries
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, \
    classification_report, accuracy_score, precision_score, f1_score, fbeta_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


def calc_accuracy(method, label_test, pred):
    print("accuracy score for ", method, accuracy_score(label_test, pred))
    print("precision_score for ", method, precision_score(label_test, pred, average='micro'))
    print("f1 score for ", method, f1_score(label_test, pred, average='micro'))
    print("f2 score for ", method, fbeta_score(label_test, pred, average='micro', beta=0.5))
    print("recall score for ", method, recall_score(label_test, pred, average='micro'))
    cm = confusion_matrix(label_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.figure_.savefig("./" + method + ".png")
    report = classification_report(label_test, pred, output_dict=True)
    df = pd.DataFrame.from_dict(report)
    df["method"] = method
    df.to_csv("./" + method + ".csv")
    print(classification_report(label_test, pred))


def categorize_values(value, num_categories=3):
    # Calculate the range of values
    value_range = max(value) - min(value)

    # Calculate the width of each category
    category_width = value_range / num_categories

    # Define the category boundaries
    boundaries = [min(value) + i * category_width for i in range(num_categories)]
    boundaries.append(max(value))

    # Assign categories based on boundaries
    categories = []
    for v in value:
        for i in range(num_categories):
            if boundaries[i] <= v <= boundaries[i + 1]:
                categories.append(i)
                break

    # Map category numbers to labels
    category_labels = {0: 'low', 1: 'medium', 2: 'high'}
    categorized_values = [category_labels[cat] for cat in categories]

    return categorized_values


data = pd.read_csv("Books_Data_Clean.csv")
# split data into features and target and target column name is "units sold"
print(data.columns)
y = categorize_values(data["units sold"])
X = data.drop(
    columns=["units sold", "index", "Book Name", "Author", "Book_average_rating", "Book_ratings_count", "gross sales",
             "publisher revenue", "sales rank"])
"""[ 'Publishing Year',  'language_code',
       'Author_Rating', 'genre',
        'sale price', 
       'Publisher ', ]"""
# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to convert string features to numerical labels
X_train = X_train.apply(LabelEncoder().fit_transform)
y_train = LabelEncoder().fit_transform(y_train)
X_test = X_test.apply(LabelEncoder().fit_transform)
y_test = LabelEncoder().fit_transform(y_test)

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

# Random Forest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# XGBoost Classifier
xgb_classifier = xgb.XGBClassifier()
xgb_classifier.fit(X_train, y_train)

# Predictions
dt_prediction = dt_classifier.predict(X_train)
rf_prediction = rf_classifier.predict(X_train)
xgb_prediction = xgb_classifier.predict(X_train)

print("Decision Tree Prediction:", dt_prediction)
print("Random Forest Prediction:", rf_prediction)
print("XGBoost Prediction:", xgb_prediction)
dt_prediction = dt_classifier.predict(X_test)
rf_prediction = rf_classifier.predict(X_test)
xgb_prediction = xgb_classifier.predict(X_test)
calc_accuracy("Decision Tree", y_test, dt_prediction)
calc_accuracy("Random Forest", y_test, rf_prediction)
calc_accuracy("XGBoost", y_test, xgb_prediction)
