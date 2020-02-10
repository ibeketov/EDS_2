import sys
import pickle
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['wordnet', 'punkt', 'stopwords'])
import re
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    """
    Function taken from the run.py file provided in the workspace
    
    Arguments:
        database_filepath  -- path to the SQLite DB 
    
    Returns:
        X, y and a list of categories.
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponseData', engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    categories = y.columns.values
    return X, y, categories


def tokenize(text):
    """
    Function taken from the run.py file provided in the workspace
    
    Arguments:
        text {Str} -- input text
    
    Returns:
        [list] -- tokens
    """    
    # Normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize
    words = word_tokenize(text)
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    # Lemmatize
    lemmed_tokens = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    return lemmed_tokens


def build_model():
    """
    Builds the model using pipeline and gridserach, for optimizing
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(DecisionTreeClassifier()))])

    # Parameters for GridSearch
    parameters = {
        'clf__estimator__min_samples_split': [2, 4, 6],
        'clf__estimator__max_depth': [2, 3, 4]}

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=5, verbose=10)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Prints out simply a classification report
    """
    y_pred = model.predict(X_test)
    
    for i in range(len(category_names)):
        print("Label:", category_names[i], '\n')
        print(classification_report(Y_test.loc[:, category_names[i]], y_pred[:, i]))


def save_model(model, model_filepath):
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()