"""
Train and compare sentiment classifiers on the IMDB polarity review dataset.

Target: F1 of at least 0.85 on the held out test split.

Four models, each one a different point on the cost curve:
    Model 0  constant prediction, the floor every real model has to beat
    Model 1  TF-IDF over lightly normalized text, logistic regression
    Model 3  TF-IDF over spaCy lemmas, logistic regression
    Model 4  TF-IDF over spaCy lemmas, LightGBM

Run:
    python train_sentiment_models.py

The dataset path comes from the IMDB_REVIEWS_PATH environment variable and
defaults to the location used on the TripleTen platform. The spaCy model is
downloaded with: python -m spacy download en_core_web_sm
"""

import json
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA_PATH = os.environ.get('IMDB_REVIEWS_PATH', '/datasets/imdb_reviews.tsv')
FIG_DIR = 'figures'
RESULTS_DIR = 'results'
SEED = 12345

MY_REVIEWS = [
    'I did not like this movie at all. The plot made no sense and I checked my phone twice.',
    'Not bad. I went in expecting nothing and came out still thinking about it the next day.',
    'A masterpiece. Every frame is deliberate and the ending earns itself.',
    'Oh great, another two hours of a man staring sadly out of a window.',
    'The acting is wooden, the script is lazy, and somehow I still want to watch it again.',
    'Beautiful to look at and completely empty underneath.',
    'I wanted to hate this one and I could not. It is actually very good.',
    'Worst thing I have seen this year. Save your money and save your evening.',
]


def normalize(series):
    """Lowercase and drop everything that is not a letter or an apostrophe."""
    return (
        series.str.lower()
        .str.replace(r"[^a-z']", ' ', regex=True)
        .str.split()
        .str.join(' ')
    )


def evaluate_model(name, model, train_features, train_target, test_features, test_target):
    """Print the metric table, save the threshold / ROC / PRC panel, return the metrics."""
    scores = {}
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    table = pd.DataFrame(index=['Accuracy', 'F1', 'APS', 'ROC AUC'])

    for part, features, target in (
        ('train', train_features, train_target),
        ('test', test_features, test_target),
    ):
        pred_target = model.predict(features)
        pred_proba = model.predict_proba(features)[:, 1]

        table.loc['Accuracy', part] = metrics.accuracy_score(target, pred_target)
        table.loc['F1', part] = metrics.f1_score(target, pred_target)
        table.loc['APS', part] = metrics.average_precision_score(target, pred_proba)
        table.loc['ROC AUC', part] = metrics.roc_auc_score(target, pred_proba)

        scores[part] = {
            'accuracy': round(float(table.loc['Accuracy', part]), 4),
            'f1': round(float(table.loc['F1', part]), 4),
            'aps': round(float(table.loc['APS', part]), 4),
            'roc_auc': round(float(table.loc['ROC AUC', part]), 4),
        }

        thresholds = np.arange(0, 1.01, 0.05)
        f1_by_threshold = [metrics.f1_score(target, pred_proba >= t) for t in thresholds]
        axs[0].plot(thresholds, f1_by_threshold, label=f'{part}, max={max(f1_by_threshold):.2f}')

        fpr, tpr, _ = metrics.roc_curve(target, pred_proba)
        axs[1].plot(fpr, tpr, label=f'{part}, AUC={table.loc["ROC AUC", part]:.2f}')

        precision, recall, _ = metrics.precision_recall_curve(target, pred_proba)
        axs[2].plot(recall, precision, label=f'{part}, AP={table.loc["APS", part]:.2f}')

    axs[0].set_title('F1 by threshold')
    axs[0].set_xlabel('threshold')
    axs[0].set_ylabel('F1')
    axs[1].plot([0, 1], [0, 1], color='grey', linestyle='--')
    axs[1].set_title('ROC curve')
    axs[1].set_xlabel('false positive rate')
    axs[1].set_ylabel('true positive rate')
    axs[2].set_title('Precision recall curve')
    axs[2].set_xlabel('recall')
    axs[2].set_ylabel('precision')
    for ax in axs:
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'eval_{name}.png'), dpi=110)
    plt.close(fig)

    print(table.round(2))
    return scores


def lemmatize_corpus(nlp, texts, batch_size=256):
    """Lemmatize with nlp.pipe, which is several times faster than calling nlp per row."""
    return [' '.join(token.lemma_ for token in doc) for doc in nlp.pipe(texts, batch_size=batch_size)]


def score_my_reviews(label, probabilities, texts):
    print()
    print(f'--- {label} on my own reviews ---')
    for prob, text in zip(probabilities, texts):
        print(f'{prob:.2f}:  {text[:100]}')


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, sep='\t', dtype={'votes': 'Int64'})
    df['review_norm'] = normalize(df['review'])

    df_train = df.query('ds_part == "train"').copy()
    df_test = df.query('ds_part == "test"').copy()
    train_target = df_train['pos']
    test_target = df_test['pos']
    print('train:', df_train.shape, ' test:', df_test.shape)

    my_reviews = pd.DataFrame(MY_REVIEWS, columns=['review'])
    my_reviews['review_norm'] = normalize(my_reviews['review'])
    my_texts = my_reviews['review_norm'].tolist()

    results = {
        'dataset': DATA_PATH,
        'seed': SEED,
        'train_rows': int(df_train.shape[0]),
        'test_rows': int(df_test.shape[0]),
        'target_f1': 0.85,
        'models': {},
        'my_review_probabilities': {'reviews': MY_REVIEWS},
    }

    # Model 0: constant prediction.
    print('\n=== Model 0: constant ===')
    train_features_0 = np.zeros((len(df_train), 1))
    test_features_0 = np.zeros((len(df_test), 1))
    model_0 = DummyClassifier(strategy='most_frequent').fit(train_features_0, train_target)
    results['models']['model_0_constant'] = evaluate_model(
        'model_0_constant', model_0, train_features_0, train_target, test_features_0, test_target
    )

    # Model 1: TF-IDF over normalized text, logistic regression.
    print('\n=== Model 1: NLTK stopwords, TF-IDF, logistic regression ===')
    import nltk

    nltk.download('stopwords')
    stop_words = set(nltk.corpus.stopwords.words('english'))

    tfidf_1 = TfidfVectorizer(stop_words=stop_words)
    train_features_1 = tfidf_1.fit_transform(df_train['review_norm'])
    test_features_1 = tfidf_1.transform(df_test['review_norm'])
    print('train:', train_features_1.shape, ' test:', test_features_1.shape)
    results['models_features'] = {'model_1': list(train_features_1.shape)}

    model_1 = LogisticRegression(solver='liblinear', random_state=SEED)
    model_1.fit(train_features_1, train_target)
    results['models']['model_1_nltk_tfidf_lr'] = evaluate_model(
        'model_1_nltk_tfidf_lr', model_1, train_features_1, train_target, test_features_1, test_target
    )

    probs_1 = model_1.predict_proba(tfidf_1.transform(my_texts))[:, 1]
    score_my_reviews('Model 1', probs_1, my_texts)
    results['my_review_probabilities']['model_1'] = [round(float(p), 2) for p in probs_1]

    # Model 3: TF-IDF over spaCy lemmas, logistic regression.
    print('\n=== Model 3: spaCy lemmas, TF-IDF, logistic regression ===')
    import spacy

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    df_train['review_lemm'] = lemmatize_corpus(nlp, df_train['review_norm'].tolist())
    df_test['review_lemm'] = lemmatize_corpus(nlp, df_test['review_norm'].tolist())

    tfidf_3 = TfidfVectorizer(stop_words=stop_words)
    train_features_3 = tfidf_3.fit_transform(df_train['review_lemm'])
    test_features_3 = tfidf_3.transform(df_test['review_lemm'])
    print('train:', train_features_3.shape, ' test:', test_features_3.shape)
    results['models_features']['model_3'] = list(train_features_3.shape)

    model_3 = LogisticRegression(solver='liblinear', random_state=SEED)
    model_3.fit(train_features_3, train_target)
    results['models']['model_3_spacy_tfidf_lr'] = evaluate_model(
        'model_3_spacy_tfidf_lr', model_3, train_features_3, train_target, test_features_3, test_target
    )

    my_lemm = lemmatize_corpus(nlp, my_texts)
    probs_3 = model_3.predict_proba(tfidf_3.transform(my_lemm))[:, 1]
    score_my_reviews('Model 3', probs_3, my_texts)
    results['my_review_probabilities']['model_3'] = [round(float(p), 2) for p in probs_3]

    # Model 4: same features, gradient boosting.
    print('\n=== Model 4: spaCy lemmas, TF-IDF, LightGBM ===')
    from lightgbm import LGBMClassifier

    model_4 = LGBMClassifier(n_estimators=500, learning_rate=0.1, random_state=SEED, n_jobs=-1)
    model_4.fit(train_features_3, train_target)
    results['models']['model_4_spacy_tfidf_lgbm'] = evaluate_model(
        'model_4_spacy_tfidf_lgbm', model_4, train_features_3, train_target, test_features_3, test_target
    )

    probs_4 = model_4.predict_proba(tfidf_3.transform(my_lemm))[:, 1]
    score_my_reviews('Model 4', probs_4, my_texts)
    results['my_review_probabilities']['model_4'] = [round(float(p), 2) for p in probs_4]

    scored = pd.DataFrame(
        {
            'review': MY_REVIEWS,
            'model_1': np.round(probs_1, 2),
            'model_3': np.round(probs_3, 2),
            'model_4': np.round(probs_4, 2),
        }
    )
    scored.to_csv(os.path.join(RESULTS_DIR, 'custom_review_scores.csv'), index=False)

    best = max(
        results['models'].items(), key=lambda kv: kv[1]['test']['f1']
    )
    results['best_model'] = best[0]
    results['best_test_f1'] = best[1]['test']['f1']
    results['target_met'] = bool(best[1]['test']['f1'] >= 0.85)

    out = os.path.join(RESULTS_DIR, 'results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print()
    print('wrote', out)


if __name__ == '__main__':
    main()
