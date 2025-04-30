import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib


df = pd.read_csv('./dataset.csv')  

print("Sample data:")
print(df.head())


X = df['Message']
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])


model.fit(X_train, y_train)
print("\nModel training complete.")


y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


def categorize_message(text):
    return model.predict([text])[0]


test_message = "I want to know the status of my order."
print(f"\nTest message: {test_message}")
print("Predicted category:", categorize_message(test_message))


joblib.dump(model, 'chat_categorizer_logreg.pkl')
print("\nModel saved as 'chat_categorizer_logreg.pkl'.")

while True:
    user_input = input("\nEnter a customer message (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    category = categorize_message(user_input)
    print(f"Predicted category: {category}")
