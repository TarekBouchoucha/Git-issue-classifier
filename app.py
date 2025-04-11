import pickle
import uuid
from flask import Flask, request, jsonify, send_from_directory, Response
from sqlalchemy import create_engine, Column, String, Table, MetaData
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from prometheus_client import Counter, Gauge, generate_latest, CollectorRegistry
from preprocessing import preprocess_text


metadata = MetaData()

issues_table = Table(
    'issues', metadata,
    Column('id', String, primary_key=True),
    Column('title', String),
    Column('body', String),
    Column('predicted_label', String),
    Column('corrected_label', String)
)

def create_app(engine, prometheus_registry=None):
    app = Flask(__name__)
    app.engine = engine

    with open('random_forest_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('tokenizer.pkl', 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    with open('label_encoder.pkl', 'rb') as label_encoder_file:
        label_encoder = pickle.load(label_encoder_file)


    metadata.create_all(engine)
    registry = prometheus_registry or CollectorRegistry()

    accuracy_metric = Gauge('accuracy', 'Accuracy of predictions', registry=registry)
    average_confidence_metric = Gauge('average_prediction_confidence', 'Average prediction confidence', registry=registry)
    predictions_per_category = Counter('predictions_per_category', 'Number of predictions per category', ['category'], registry=registry)
    correct_predictions_per_category = Counter('correct_predictions_per_category', 'Number of correct predictions per category', ['category'], registry=registry)
    incorrect_predictions_per_category = Counter('incorrect_predictions_per_category', 'Number of incorrect predictions per category', ['category'], registry=registry)

    global total_predictions, correct_predictions, total_confidence
    total_predictions = 0
    correct_predictions = 0
    total_confidence = 0.0

    def text_to_sequence(title, body):
        combined_text = (title + " " + body).lower()
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        preprocessed_text = preprocess_text(combined_text, stop_words, stemmer, lemmatizer)
        sequences = tokenizer.texts_to_sequences([preprocessed_text])
        padded_sequences = pad_sequences(sequences, padding='post', maxlen=200)
        return padded_sequences

    @app.route('/')
    def index():
        return send_from_directory(app.static_folder, 'index.html')

    @app.route('/metrics', methods=['GET'])
    def metrics():
        return Response(generate_latest(registry), mimetype="text/plain")

    @app.route('/api/predict', methods=['POST'])
    def predict_issue_type():
        global total_predictions, total_confidence
        data = request.get_json()
        title = data.get("title")
        body = data.get("body")
        issue_id = str(uuid.uuid4())

        processed_text = text_to_sequence(title, body)
        prediction = model.predict_proba(processed_text)
        predicted_label_index = prediction.argmax()
        predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
        confidence = prediction[0][predicted_label_index]

        total_predictions += 1
        total_confidence += confidence
        average_confidence_metric.set(total_confidence / total_predictions)
        predictions_per_category.labels(category=predicted_label).inc()

        with engine.connect() as conn:
            conn.execute(issues_table.insert().values(
                id=issue_id,
                title=title,
                body=body,
                predicted_label=predicted_label,
                corrected_label=None
            ))
            conn.commit()

        return jsonify({"id": issue_id, "label": predicted_label, "confidence": confidence}), 200

    @app.route('/api/correct', methods=['POST'])
    def correct_issue_type():
        global correct_predictions
        data = request.get_json()
        issue_id = data.get("id")
        corrected_label = data.get("label")

        with engine.connect() as conn:
            result = conn.execute(issues_table.select().where(issues_table.c.id == issue_id)).fetchone()

            if result:
                predicted_label = result.predicted_label
                conn.execute(issues_table.update().where(issues_table.c.id == issue_id).values(
                    corrected_label=corrected_label
                ))
                conn.commit()

                if predicted_label == corrected_label:
                    correct_predictions += 1
                    correct_predictions_per_category.labels(category=predicted_label).inc()
                else:
                    incorrect_predictions_per_category.labels(category=predicted_label).inc()

                if total_predictions > 0:
                    accuracy_metric.set(correct_predictions / total_predictions)
                else:
                    accuracy_metric.set(0)

                return jsonify({
                    "id": issue_id,
                    "corrected_label": corrected_label,
                    "predicted_label": predicted_label,
                }), 200
            else:
                return jsonify({"error": "Issue ID not found"}), 404


    return app

if __name__ == '__main__':
    app = create_app(create_engine('sqlite:///issues.db'))
    app.run(debug=True)
