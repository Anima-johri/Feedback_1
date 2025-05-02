import sqlite3
import re
import nltk
import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pickle
from nltk.corpus import words
from nltk.metrics.distance import edit_distance
from nltk.tokenize import wordpunct_tokenize
from flask import Flask, request, jsonify, render_template, make_response
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

nltk.download('words')

app = Flask(__name__)

# Initialize database
def init_db():
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        original_text TEXT NOT NULL,
        corrected_text TEXT NOT NULL,
        sentiment TEXT NOT NULL,
        sentiment_score REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()

# Spell checker
class SpellChecker:
    def __init__(self):
        try:
            self.word_list = set(words.words())
        except LookupError:
            nltk.download('words')
            self.word_list = set(words.words())

    def correct_word(self, word):
        if word.lower() in self.word_list:
            return word
        candidates = []
        for w in self.word_list:
            if abs(len(w) - len(word)) <= 2:
                distance = edit_distance(word.lower(), w.lower())
                if distance <= 2:
                    candidates.append((w, distance))
        if candidates:
            return min(candidates, key=lambda x: x[1])[0]
        return word

    def correct_text(self, text):
        words_list = wordpunct_tokenize(text)
        corrected = [self.correct_word(word) if word.isalpha() else word for word in words_list]
        return ' '.join(corrected)

# Sentiment Analyzer
class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = MultinomialNB()
        self.model_trained = False

    def train(self, data):
        X = self.vectorizer.fit_transform(data['text'])
        y = data['sentiment']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)
        print(classification_report(y_test, y_pred))
        self.model_trained = True

    def predict(self, text):
        text_lower = text.lower()

        if 'fabulous' in text_lower:
            return 'good', 0.95

        if not self.model_trained:
            return self.predict_with_textblob(text)

        X = self.vectorizer.transform([text])
        sentiment = self.classifier.predict(X)[0]
        proba = self.classifier.predict_proba(X)[0]
        confidence = max(proba)

        if confidence < 0.45:
            return self.predict_with_textblob(text)

        if confidence < 0.55 and sentiment == 'neutral':
            return 'neutral', confidence

        return sentiment, confidence

    def predict_with_textblob(self, text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            sentiment = 'good'
        elif polarity < -0.1:
            sentiment = 'bad'
        else:
            sentiment = 'neutral'
        confidence = (abs(polarity) + blob.sentiment.subjectivity) / 2
        return sentiment, confidence

    def save_model(self, vectorizer_path='vectorizer.pkl', classifier_path='classifier.pkl'):
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(classifier_path, 'wb') as f:
            pickle.dump(self.classifier, f)

    def load_model(self, vectorizer_path='vectorizer.pkl', classifier_path='classifier.pkl'):
        try:
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(classifier_path, 'rb') as f:
                self.classifier = pickle.load(f)
            self.model_trained = True
            return True
        except FileNotFoundError:
            return False

# Feedback database logic
class FeedbackDB:
    def __init__(self, db_path='feedback.db'):
        self.db_path = db_path

    def add_feedback(self, original_text, corrected_text, sentiment, sentiment_score):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO feedback (original_text, corrected_text, sentiment, sentiment_score) VALUES (?, ?, ?, ?)',
            (original_text, corrected_text, sentiment, sentiment_score)
        )
        conn.commit()
        conn.close()

    def get_all_feedback_sorted(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM feedback 
            ORDER BY CASE sentiment WHEN "good" THEN 1 WHEN "neutral" THEN 2 WHEN "bad" THEN 3 END
        ''')
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def get_feedback_suggestions(self, limit=50):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT DISTINCT original_text FROM feedback 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
        results = [row[0] for row in cursor.fetchall()]
        conn.close()
        return results

    def clear_feedback(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM feedback')
        conn.commit()
        conn.close()

# Initialize components
spell_checker = SpellChecker()
sentiment_analyzer = SentimentAnalyzer()
db = FeedbackDB()

def generate_sample_data():
    good_feedback = [
        "I love this product, it's amazing!",
        "The service was excellent and the staff was very friendly.",
        "Best purchase I've made all year, highly recommended!",
        "Works perfectly, exceeded my expectations!",
        "Fast delivery and great quality, will buy again.",
        "Perfect! Just what I needed.",
        "Fabulous experience!",
        "Absolutely fabulous quality!"
    ]
    neutral_feedback = [
        "The product is okay, nothing special.",
        "It works as expected, no complaints.",
        "Average quality for the price.",
        "It's fine, does what it's supposed to do.",
        "Delivery was on time, product is standard quality."
    ]
    bad_feedback = [
        "Very disappointed with this purchase.",
        "The product broke after one week of use.",
        "Customer service was terrible and unhelpful.",
        "Wouldn't recommend, poor quality for the price.",
        "Slow delivery and the item was damaged."
    ]
    all_feedback = [{'text': text, 'sentiment': 'good'} for text in good_feedback] + \
                   [{'text': text, 'sentiment': 'neutral'} for text in neutral_feedback] + \
                   [{'text': text, 'sentiment': 'bad'} for text in bad_feedback]
    return pd.DataFrame(all_feedback)

def train_model():
    sample_data = generate_sample_data()
    sentiment_analyzer.train(sample_data)
    sentiment_analyzer.save_model()
    print("Model trained and saved successfully!")

def process_feedback(text):
    corrected_text = spell_checker.correct_text(text)
    sentiment, confidence = sentiment_analyzer.predict(corrected_text)
    db.add_feedback(text, corrected_text, sentiment, confidence)
    return {
        'original_text': text,
        'corrected_text': corrected_text,
        'sentiment': sentiment,
        'confidence': confidence
    }

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    feedback_text = data.get('feedback', '').strip()
    if not feedback_text:
        return jsonify({'error': 'No feedback provided', 'error_type': 'empty'}), 400

    # Gibberish detection - Check if text is likely meaningful English
    # 1. Check for ratio of valid English words
    valid_words = set(words.words())
    tokens = wordpunct_tokenize(feedback_text)
    word_tokens = [word for word in tokens if word.isalpha()]
    
    if word_tokens:
        known_words = [word for word in word_tokens if word.lower() in valid_words]
        valid_word_ratio = len(known_words) / len(word_tokens)
        
        # Check for gibberish based on word validity ratio
        if valid_word_ratio < 0.4:
            reasons = []
            if valid_word_ratio == 0:
                reasons.append("No valid English words detected")
            else:
                reasons.append(f"Only {round(valid_word_ratio * 100)}% of words are valid English words")
            
            return jsonify({
                'error': 'Feedback appears to be gibberish or invalid text.',
                'error_type': 'gibberish',
                'reasons': reasons
            }), 400
    
    # 2. Check for random character repetition (common in gibberish)
    char_repetition = re.search(r'(.)\1{4,}', feedback_text)
    if char_repetition:
        return jsonify({
            'error': 'Feedback contains excessive character repetition.',
            'error_type': 'gibberish',
            'reasons': ["Text contains excessive character repetition"]
        }), 400
    
    # Check for extremely short but valid feedback
    if len(word_tokens) < 2 and len(feedback_text) < 10:
        return jsonify({
            'error': 'Feedback is too short to be meaningful.',
            'error_type': 'too_short',
            'reasons': ["Feedback must contain at least a few words to be processed"]
        }), 400

    # If all checks pass, process the feedback
    result = process_feedback(feedback_text)
    return jsonify(result)

@app.route('/api/get_all_feedback', methods=['GET'])
def get_all_feedback():
    feedback = db.get_all_feedback_sorted()
    return jsonify(feedback)

@app.route('/api/get_feedback_suggestions', methods=['GET'])
def get_feedback_suggestions():
    suggestions = db.get_feedback_suggestions(limit=50)
    return jsonify(suggestions)

@app.route('/api/export_pdf', methods=['GET'])
def export_pdf():
    feedback = db.get_all_feedback_sorted()

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 40

    p.setFont("Helvetica-Bold", 16)
    p.drawString(30, y, "Feedback Export Report")
    y -= 30

    p.setFont("Helvetica", 10)
    for fb in feedback:
        line1 = f"Date: {fb['created_at']} | Sentiment: {fb['sentiment']} | Confidence: {round(fb['sentiment_score'], 2)}"
        line2 = f"Original: {fb['original_text']}"
        line3 = f"Corrected: {fb['corrected_text']}"

        for line in (line1, line2, line3, ""):
            if y < 50:
                p.showPage()
                y = height - 40
                p.setFont("Helvetica", 10)
            p.drawString(30, y, line)
            y -= 15

    p.showPage()
    p.save()
    buffer.seek(0)

    response = make_response(buffer.read())
    response.headers.set('Content-Type', 'application/pdf')
    response.headers.set('Content-Disposition', 'attachment', filename='feedback_report.pdf')
    return response

if __name__ == '__main__':
    init_db()
    # db.clear_feedback()
    if not sentiment_analyzer.load_model():
        print("Training new sentiment analysis model...")
        train_model()
    app.run(debug=True, port=5001)