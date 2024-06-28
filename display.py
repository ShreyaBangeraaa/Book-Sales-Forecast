from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, recall_score, fbeta_score, \
    f1_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Sample data for the table
originalData = pd.read_csv("Books_Data_Clean.csv")
if "index" in originalData.columns:
    originalData = originalData.drop(columns=["index"])
columns = list(originalData.columns)
data = originalData.drop(
    columns=["Book Name", "Author", "Book_average_rating", "Book_ratings_count", "gross sales",
             "publisher revenue", "sales rank"])


def calc_accuracy(method, label_test, pred):
    print("accuracy score for ", method, accuracy_score(label_test, pred))
    print("precision_score for ", method, precision_score(label_test, pred, average='micro'))
    print("f1 score for ", method, f1_score(label_test, pred, average='micro'))
    print("f2 score for ", method, fbeta_score(label_test, pred, average='micro', beta=0.5))
    print("recall score for ", method, recall_score(label_test, pred, average='micro'))
    cm = confusion_matrix(label_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("static/" + method + ".png")
    report = classification_report(label_test, pred, output_dict=True)
    df = pd.DataFrame.from_dict(report)
    df.to_csv("static/" + method + ".csv")
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


def train():
    y = categorize_values(data["units sold"])
    X = data.drop(columns=["units sold"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train = X_train.apply(LabelEncoder().fit_transform)
    y_train = LabelEncoder().fit_transform(y_train)
    X_test = X_test.apply(LabelEncoder().fit_transform)
    y_test = LabelEncoder().fit_transform(y_test)

    # Decision Tree Classifier
    dt_classifier_local = DecisionTreeClassifier()
    dt_classifier_local.fit(X_train, y_train)

    # Random Forest Classifier
    rf_classifier_local = RandomForestClassifier()
    rf_classifier_local.fit(X_train, y_train)

    # XGBoost Classifier
    xgb_classifier_local = xgb.XGBClassifier()
    xgb_classifier_local.fit(X_train, y_train)
    return dt_classifier_local, rf_classifier_local, xgb_classifier_local, X_test, y_test


dt_classifier, rf_classifier, xgb_classifier, X_test, y_test = train()
dt_prediction = dt_classifier.predict(X_test)
rf_prediction = rf_classifier.predict(X_test)
xgb_prediction = xgb_classifier.predict(X_test)
calc_accuracy("Decision Tree", y_test, dt_prediction)
calc_accuracy("Random Forest", y_test, rf_prediction)
calc_accuracy("XGBoost", y_test, xgb_prediction)


# Function to display table
def display_table(file="default"):
    if file == "default":
        global data
        return data.to_html()
    else:
        df = pd.read_csv("static/" + file + ".csv")
        return df.to_html()


# Function to add content to the table
def add_to_table(name, age, city):
    global data
    new_row = {"Name": name, "Age": age, "City": city}
    data = data.append(new_row, ignore_index=True)
    return "Content added successfully!"


# Function to display bar chart
def display_bar_chart():
    plt.bar(data["Name"], data["Age"])
    plt.xlabel("Name")
    plt.ylabel("Age")
    plt.title("Age Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        print("im here")
        if 'username' not in session:
            return redirect(url_for('index'))
        return f(*args, **kwargs)

    return decorated_function


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


# Login route
@app.route("/", methods=["GET", "POST"])
def login():
    print("login page")
    if 'username' in session:
        return redirect(url_for('display_table_page'))
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username == "user" and password == "password":
            session['username'] = username
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", message="Invalid username or password.")
    return render_template("login.html", message="")


# Page selection route
@app.route("/page_selection", methods=["GET", "POST"])
def page_selection():
    if request.method == "POST":
        page_option = request.form["page_option"]
        if page_option == "Display Table":
            return redirect(url_for("display_table_page"))
        elif page_option == "Add Content":
            return redirect(url_for("add_content_page"))
        elif page_option == "Display Bar Chart":
            return redirect(url_for("display_bar_chart_page"))
    return render_template("page_selection.html")


# Display table page
@app.route("/display_table_page", methods=["GET", "POST"])
def display_table_page():
    return render_template("display_table.html", table=display_table())


# Add content page
@app.route("/add_content_page", methods=["GET", "POST"])
def add_content_page():
    if request.method == "POST":
        temp = list()
        global originalData, data
        for column in originalData.columns:
            print(column)
            temp.append(request.form[column])
            print(temp)
        df_extended = pd.DataFrame([temp], columns=originalData.columns)
        originalData = pd.concat([originalData, df_extended])
        originalData.to_csv("Books_Data_Clean.csv", index=False)
        originalData = pd.read_csv("Books_Data_Clean.csv")
        data = originalData.drop(
            columns=["Book Name", "Author", "Book_average_rating", "Book_ratings_count", "gross sales",
                     "publisher revenue", "sales rank"])
        global dt_classifier, rf_classifier, xgb_classifier, X_test, y_test
        dt_classifier, rf_classifier, xgb_classifier, X_test, y_test = train()
        return redirect(url_for("display_table_page"))
    return render_template("add_content.html", columns=columns)


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    return render_template("dashboard.html", columns=columns)


@app.route("/about", methods=["GET", "POST"])
def about():
    return render_template("about.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    return render_template("contact.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    global X_test
    if request.method == "POST":
        temp = [request.form[column] for column in X_test.columns]
        out = dt_classifier.predict([temp])[0]
        print(out)
        if out == 0:
            out = "LOW"
        elif out == 1:
            out = "MEDIUM"
        else:
            out = "HIGH"
        return render_template("predict.html", columns=X_test.columns, message="Predicted Book Sales is " + str(out))
    return render_template("predict.html", columns=X_test.columns)


# Display bar chart page
@app.route("/display_bar_chart_page", methods=["GET", "POST"])
def display_bar_chart_page():
    return render_template("display_chart.html", decision=display_table("Decision Tree"),
                           random=display_table("Random Forest"), xgb=display_table("XGBoost"))


if __name__ == "__main__":
    app.run(debug=True)
