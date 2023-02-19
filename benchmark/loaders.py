from sklearn.datasets import fetch_20newsgroups, fetch_openml
import re
import pandas as pd
import os
import numpy as np


def loader_click_prediction(save_folder=".", seed=42):
    """
    Loads the click prediction dataset.
    See: https://www.openml.org/d/1219

    Uses a random 75% of the columns as features and combines them into a single
    column delimited by "|". The column index is used as a prefix to each token
    to avoid collisions.
    """
    
    filename = os.path.join(save_folder, f"click_prediction_{seed}.csv")
    if not os.path.exists(filename):
        data = fetch_openml(data_id = 1219, as_frame = True, parser = "auto", cache = True)
        df = data.frame

        feature_columns = [
            "impression",
            "url_hash",
            "ad_id",
            "advertiser_id",
            "depth",
            "position",
            "query_id",
            "keyword_id",
            "title_id",
            "description_id",
            "user_id",
        ]

        # Sample random 75% of columns to use as features
        np.random.seed(seed)
        np.random.shuffle(feature_columns)
        feature_columns = feature_columns[:int(len(feature_columns) * 0.75)]

        # Combine the feature columns into a single column delimited by "|" and with the
        # column index as a prefix to each token
        df["features"] = df[feature_columns].apply(lambda x: "|".join([f"{i}:{v}" for i, v in enumerate(x)]), axis=1)
        # Set the target column and convert to int
        df["target"] = df["click"].astype(int)

        # Write to csv
        df[["features", "target"]].to_csv(filename, index=False)
    else:
        df = pd.read_csv(filename)

    return df

def loader_newsgroup_binary(save_folder=".", word_grams=1, seed=42):
    """Returns the 20 Newsgroups dataset as a binary classification problem."""
    # check if csv file exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    file = os.path.join(save_folder, f"newsgroups_{word_grams}_{seed}.csv")

    if not os.path.exists(file):
        cats = [
            'alt.atheism',
            'comp.graphics',
            'comp.os.ms-windows.misc',
            'comp.sys.ibm.pc.hardware',
            'comp.sys.mac.hardware',
            'comp.windows.x',
            'misc.forsale',
            'rec.autos',
            'rec.motorcycles',
            'rec.sport.baseball',
            'rec.sport.hockey',
            'sci.crypt',
            'sci.electronics',
            'sci.med',
            'sci.space',
            'soc.religion.christian',
            'talk.politics.guns',
            'talk.politics.mideast',
            'talk.politics.misc',
            'talk.religion.misc'
        ]

        # Randomly select half of the categories as positive
        # and the other half as negative
        np.random.seed(seed)
        np.random.shuffle(cats)
        cats_pos = cats[:len(cats)//2]
        cats_neg = cats[len(cats)//2:]

        newsgroups = fetch_20newsgroups(remove=("headers", "footers", "quotes"), categories=cats_pos + cats_neg)
        data = newsgroups.data
        labels = [1 if newsgroups.target_names[i] in cats_pos else 0 for i in newsgroups.target]

        df = pd.DataFrame({"text": data, "target": labels})

        # Split the text column into "|"-delimited words
        # using sklearn
        token_pattern = re.compile(r"(?u)\b\w\w+\b")
        tokenizer = token_pattern.findall
        # Tokenize
        df["features"] = df["text"].apply(lambda x: tokenizer(x.replace("|", "?")))
        # Word ngram
        if word_grams != 1:
            df["features"] = df["features"].apply(lambda x: [" ".join(x[i:i+word_grams]) for i in range(len(x)-word_grams+1)])
        df["features"] = df["features"].apply(lambda x: np.str_("|".join(x)))

        df.to_csv(file, index=False)
    else:
        df = pd.read_csv(file)

    df["features"] = df["features"].apply(lambda x: np.str_(x))
    return df




if __name__ == "__main__":
    df = loader_click_prediction()

    print(df.head(100))

    exit()

    df = loader_newsgroup_binary(word_grams=1)
    print(df.head(100))

    # Fit a simple CountVectorizer model as a test
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(df["features"], df["target"], test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("vectorizer", CountVectorizer(tokenizer = lambda x: x.split('|'))),
        ("classifier", LogisticRegression())
    ])

    pipeline.fit(X_train, y_train)

    print(pipeline.score(X_test, y_test))


