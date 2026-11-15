"""
Exploratory analysis of the IMDB polarity review dataset.

Answers the questions that decide how the sentiment models get built:
how much data there is, how it is split, whether the classes are balanced,
and whether the balance holds across years.

Run:
    python explore_reviews.py

The dataset path comes from the IMDB_REVIEWS_PATH environment variable and
defaults to the location used on the TripleTen platform.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_PATH = os.environ.get('IMDB_REVIEWS_PATH', '/datasets/imdb_reviews.tsv')
FIG_DIR = 'figures'
RESULTS_DIR = 'results'


def load_reviews():
    df = pd.read_csv(DATA_PATH, sep='\t', dtype={'votes': 'Int64'})
    print('rows, columns:', df.shape)
    print()
    print('missing values per column:')
    print(df.isna().sum()[lambda s: s > 0])
    print()
    print('fully duplicated review texts:', df['review'].duplicated().sum())
    return df


def movies_and_reviews_per_year(df):
    fig, axs = plt.subplots(2, 1, figsize=(16, 8))

    ax = axs[0]
    per_year = (
        df[['tconst', 'start_year']]
        .drop_duplicates()['start_year']
        .value_counts()
        .sort_index()
    )
    per_year = per_year.reindex(
        index=np.arange(per_year.index.min(), max(per_year.index.max(), 2021))
    ).fillna(0)
    per_year.plot(kind='bar', ax=ax)
    ax.set_title('Number of movies over years')
    ax.set_xlabel('year')
    ax.set_ylabel('movies')

    ax = axs[1]
    reviews_year = df['start_year'].value_counts().sort_index()
    reviews_year = reviews_year.reindex(
        index=np.arange(reviews_year.index.min(), max(reviews_year.index.max(), 2021))
    ).fillna(0)
    reviews_year.plot(kind='bar', ax=ax)
    ax.set_title('Number of reviews over years')
    ax.set_xlabel('year')
    ax.set_ylabel('reviews')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'movies_and_reviews_per_year.png'), dpi=110)
    plt.close(fig)


def reviews_per_movie(df):
    counts = df.groupby('tconst')['review'].count()

    fig, axs = plt.subplots(1, 2, figsize=(16, 5))

    ax = axs[0]
    counts.value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_title('Bar plot of reviews per movie')
    ax.set_xlabel('reviews')
    ax.set_ylabel('movies')

    ax = axs[1]
    counts.plot(kind='kde', ax=ax)
    ax.set_title('KDE of reviews per movie')
    ax.set_xlabel('reviews')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'reviews_per_movie.png'), dpi=110)
    plt.close(fig)
    return counts


def class_balance(df):
    print()
    print(df['ds_part'].value_counts())
    print()
    print(df['pos'].value_counts())
    print()
    print('class balance by split (count, share positive):')
    print(df.pivot_table(index='ds_part', values='pos', aggfunc=['count', 'mean']).round(3))
    print()
    print(df.pivot_table(index='ds_part', columns='pos', values='review', aggfunc='count'))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    df['ds_part'].value_counts().plot(kind='bar', ax=axs[0])
    axs[0].set_title('Reviews per split')
    axs[0].set_ylabel('reviews')

    df['pos'].value_counts().sort_index().plot(kind='bar', ax=axs[1])
    axs[1].set_title('Reviews per class')
    axs[1].set_xlabel('pos')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'class_balance.png'), dpi=110)
    plt.close(fig)


def polarity_by_year_and_split(df):
    fig, axs = plt.subplots(2, 2, figsize=(16, 8), gridspec_kw={'width_ratios': (2, 1)})

    for row, part in enumerate(['train', 'test']):
        sub = df.query('ds_part == @part')

        ax = axs[row][0]
        table = sub.pivot_table(
            index='start_year', columns='pos', values='review', aggfunc='count'
        ).fillna(0)
        table.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title(f'{part}: reviews per year by polarity')
        ax.set_xlabel('year')
        ax.set_ylabel('reviews')

        ax = axs[row][1]
        counts = sub.groupby(['tconst', 'pos'])['review'].count().unstack().fillna(0)
        for col in counts.columns:
            counts[col].plot(kind='kde', ax=ax, label=f'pos={col}')
        ax.legend()
        ax.set_title(f'{part}: reviews per movie by polarity')
        ax.set_xlabel('reviews')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'polarity_by_year_and_split.png'), dpi=110)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_reviews()
    movies_and_reviews_per_year(df)
    counts = reviews_per_movie(df)
    class_balance(df)
    polarity_by_year_and_split(df)

    summary = {
        'rows': int(df.shape[0]),
        'columns': int(df.shape[1]),
        'movies': int(df['tconst'].nunique()),
        'duplicate_review_texts': int(df['review'].duplicated().sum()),
        'missing_average_rating': int(df['average_rating'].isna().sum()),
        'missing_votes': int(df['votes'].isna().sum()),
        'year_min': int(df['start_year'].min()),
        'year_max': int(df['start_year'].max()),
        'reviews_per_movie_median': float(counts.median()),
        'reviews_per_movie_max': int(counts.max()),
        'split_counts': {k: int(v) for k, v in df['ds_part'].value_counts().items()},
        'share_positive_by_split': {
            k: round(float(v), 4)
            for k, v in df.groupby('ds_part')['pos'].mean().items()
        },
    }
    out = os.path.join(RESULTS_DIR, 'eda_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print()
    print('wrote', out)


if __name__ == '__main__':
    main()
