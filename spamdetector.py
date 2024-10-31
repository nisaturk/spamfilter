import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

file_path = 'spamorham.txt'

# Lists to hold spam and ham messages from the txt file
spam_messages = []
ham_messages = []

# Reading the file and separating the messages
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            stripped_line = line.strip()  # Removing leading/trailing whitespace
            if stripped_line.startswith("ham"):
                ham_messages.append(stripped_line[4:].strip())
            elif stripped_line.startswith("spam"):
                spam_messages.append(stripped_line[5:].strip())

    # Printing the count of spam & ham
    print(f"\nNumber of spam messages: {len(spam_messages)}")
    print(f"Number of ham messages: {len(ham_messages)}")

except Exception as e:
    print(f"Error: {e}")

# Create a DataFrame from the lists
data = pd.DataFrame({'label': ['ham'] * len(ham_messages) + ['spam'] * len(spam_messages),
                     'message': ham_messages + spam_messages})

# Function to clean the text
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

data['cleaned_message'] = data['message'].apply(preprocess_text)  # Cleaning

# Creating a "Bag of Words" model with n-grams
# Using n-grams for a better detection filter
vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english')  # Unigrams and bigrams
X = vectorizer.fit_transform(data['cleaned_message'])  # Converting the messages to feature vectors
y = data['label']  # Labels (ham or spam)

# Convert to an array if needed
X_array = X.toarray()
print("Feature vectors shape:", X_array.shape)

# Splitting the dataset into sets of training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=82) # Now using a 70-30 split

# Creating & training the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Function to predict if a message is spam or ham
def predict_message(message):
    cleaned_message = preprocess_text(message)
    features = vectorizer.transform([cleaned_message])  # Converting the message to a feature vector
    prediction = model.predict(features)
    return prediction[0]

# Testing the model manually, with some example texts:
    
#new_message = "Hey its me, the guy from biology101." #ham
#new_message = "Congratulations! You have been selected for a free trial of our amazing weight loss product. Sign up today to start your journey!" #spam
new_message = "Hey Isabel! Long time no see, we missed you! We put a 20 USD worth of coupon for you to use on our halloween collection! Offer ends on 31 October" #ham
print(f"The message is classified as: {predict_message(new_message)}")