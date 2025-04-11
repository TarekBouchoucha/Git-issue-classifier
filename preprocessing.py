import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import pickle

def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('stopwords')

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_text(text, stop_words, stemmer):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    #tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def preprocess_dataframe(df):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    #lemmatizer = WordNetLemmatizer()

    df['combined_text'] = df['issue_title']+ ' ' + df['issue_body']
    df['cleaned_issue_body'] = df['combined_text'].apply(
        lambda x: preprocess_text(x, stop_words, stemmer)
    )
    return df[['cleaned_issue_body', 'issue_label']]

def encode_labels(df, column_name):
    label_encoder = LabelEncoder()
    df['encoded_label'] = label_encoder.fit_transform(df[column_name])
    return df, label_encoder

def tokenize_and_pad_sequences(texts, max_vocab_size, maxlen=100):
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=maxlen)
    return tokenizer, padded_sequences

def save_objects(objects, filenames):
    for obj, filename in zip(objects, filenames):
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)

def main():
    #download_nltk_resources()
    
    #df1 = load_data('sample1.csv')
    #df2 = load_data('sample2.csv')
    #df = pd.concat([df1, df2], ignore_index=True)
    df = load_data('sample1.csv')

    
    df = df.dropna(subset=['issue_body', 'issue_title', 'issue_label'])
    df = df[(df['issue_body'].str.strip() != '') & (df['issue_title'].str.strip() != '') & (df['issue_label'].str.strip() != '')]
    
    df = preprocess_dataframe(df)
    

    df, label_encoder = encode_labels(df, 'issue_label')
    
    
    df = df.dropna()
    all_words = ' '.join(df['cleaned_issue_body']).split()
    vocab_size = len(set(all_words))
    max_vocab_size = min(vocab_size, 10000)
    
    tokenizer, padded_sequences = tokenize_and_pad_sequences(
        df['cleaned_issue_body'], max_vocab_size, maxlen=200
    )
    
    padded_sequences_df = pd.DataFrame(padded_sequences)

    df = df.reset_index(drop=True)
    padded_sequences_df = padded_sequences_df.reset_index(drop=True)

    combined_df = pd.concat([df['encoded_label'], padded_sequences_df], axis=1)
    combined_path = 'processed_data.csv'
    combined_df.to_csv(combined_path, index=False)

    save_objects(
        [tokenizer, label_encoder],
        ['tokenizer.pkl', 'label_encoder.pkl']
    )

if __name__ == "__main__":
    main()
