from sklearn.datasets import fetch_20newsgroups
import re
import pandas as pd
import os
import numpy as np

def loader_newsgroup_binary(save_folder="."):
    """Returns the 20 Newsgroups dataset as a binary classification problem."""
    # check if csv file exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if not os.path.exists(os.path.join(save_folder,"newsgroups.csv")):
        cats_pos = ["alt.atheism", "sci.space"]
        cats_neg = ["comp.graphics", "comp.sys.ibm.pc.hardware"]

        newsgroups = fetch_20newsgroups(remove=("headers", "footers", "quotes"), categories=cats_pos + cats_neg)
        data = newsgroups.data
        labels = [1 if newsgroups.target_names[i] in cats_pos else 0 for i in newsgroups.target]

        df = pd.DataFrame({"text": data, "target": labels})

        # Split the text column into "|"-delimited words
        # using sklearn
        token_pattern = re.compile(r"(?u)\b\w\w+\b")
        tokenizer = token_pattern.findall
        df["features"] = df["text"].apply(lambda x: tokenizer(x.replace("|", "?")))
        df["features"] = df["features"].apply(lambda x: np.str_("|".join(x)))

        df.to_csv(os.path.join(save_folder,"newsgroups.csv"), index=False)
    else:
        df = pd.read_csv(os.path.join(save_folder,"newsgroups.csv"))

    df["features"] = df["features"].apply(lambda x: np.str_(x))
    return df


if __name__ == "__main__":
    df = loader_newsgroup_binary()
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


