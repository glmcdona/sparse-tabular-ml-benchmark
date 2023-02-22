from sklearn.datasets import fetch_20newsgroups, fetch_openml
import re
import pandas as pd
import os
import numpy as np



def loader_airlines(save_folder=".", seed=42):
    """
    Loads the airlines dataset.
    See: https://www.openml.org/d/1169
    """
    
    filename = os.path.join(save_folder, f"airlines_{seed}.csv")
    if not os.path.exists(filename):
        data = fetch_openml(data_id = 1169, as_frame = True, parser = "auto", cache = True)
        df = data.frame

        # Add some combined features
        df["FromTo"] = df["AirportFrom"].astype(str) + df["AirportTo"].astype(str)
        df["AirlineFlight"] = df["Airline"].astype(str) + df["Flight"].astype(str)
        df["AirlineFlightFromTo"] = df["Airline"].astype(str) + df["Flight"].astype(str) + df["AirportFrom"].astype(str) + df["AirportTo"].astype(str)
        df["AirlineFlightFromToTime"] = df["Airline"].astype(str) + df["Flight"].astype(str) + df["AirportFrom"].astype(str) + df["AirportTo"].astype(str) + df["Time"].astype(str)
        df["AirlineFlightDayOfWeek"] = df["Airline"].astype(str) + df["Flight"].astype(str) + df["DayOfWeek"].astype(str)
        df["AirlineFlightDayOfWeekTime"] = df["Airline"].astype(str) + df["Flight"].astype(str) + df["DayOfWeek"].astype(str) + df["Time"].astype(str)

        feature_columns = [
            "Airline",
            "Flight",
            "AirportFrom",
            "AirportTo",
            "DayOfWeek",
            "Time",
            "Length",
            "FromTo",
        ]

        # Combine the feature columns into a single column delimited by "|" and with the
        # column index as a prefix to each token
        df["features"] = df[feature_columns].apply(lambda x: "|".join([f"{i}:{v}" for i, v in enumerate(x)]), axis=1)
        
        # Set the target column and convert to int
        df["target"] = df["Delay"].astype(int)
        
        # Write to csv
        df[["features", "target"]].to_csv(filename, index=False)
    else:
        df = pd.read_csv(filename)
    
    return df



def loader_safe_driver(save_folder=".", seed=42):
    """
    'Porto Seguros Safe Driver Prediction' Kaggle challenge
    [https://www.kaggle.com/c/porto-seguro-safe-driver-prediction]
    See: https://www.openml.org/d/42742
    """
    seed = 42
    filename = os.path.join(save_folder, f"safe_driver_{seed}.csv")
    if not os.path.exists(filename):
        data = fetch_openml(data_id = 42742, as_frame = True, parser = "auto", cache = True)
        df = data.frame

        # Filter to only categorical columns
        df = df.select_dtypes(include=["category"])

        # Create multiple categorical features:
        #  1. 10 hashes from random 25% of categorical columns combined
        #  2. 10 hashes from random 40% of categorical columns combined
        #  3. 4 hashes from random 60% of categorical columns combined
        #  4. 1 hash from all categorical columns combined
        np.random.seed(seed)
        columns = list(df.columns)
        columns.remove("target")  

        max_hash_value = 100_000_000

        for i in range(10):
            np.random.shuffle(columns)
            df[f"features_{i}"] = df[columns[:int(len(columns) * 0.25)]].apply(lambda x: hash(tuple(x)) % max_hash_value, axis=1)
        for i in range(10):
            np.random.shuffle(columns)
            df[f"features_{i + 10}"] = df[columns[:int(len(columns) * 0.4)]].apply(lambda x: hash(tuple(x)) % max_hash_value, axis=1)
        for i in range(1):
            np.random.shuffle(columns)
            df[f"features_{i + 20}"] = df[columns[:int(len(columns) * 0.60)]].apply(lambda x: hash(tuple(x)) % max_hash_value, axis=1)
        df["features_24"] = df.apply(lambda x: hash(tuple(x)) % max_hash_value, axis=1)
        
        # Combine all categorical features into one value separated by "|"
        columns.extend( [f for f in df.columns if f.startswith("features_")] )
        df["features"] = df[columns].apply(lambda x: "|".join([f"{i}:{v}" for i, v in enumerate(x)]), axis=1)

        df["target"] = df["target"].astype(int)

        # Print column names with counts of unique values
        #for col in df.columns:
        #    print(f"{col}: {df[col].nunique()}")

        # Write to csv
        df[["features", "target"]].to_csv(filename, index=False)
    else:
        df = pd.read_csv(filename)

    return df



def loader_census_income(save_folder=".", seed=42):
    """
    https://www.openml.org/d/42750
    This version has feature names based on
    https://www2.1010data.com/documentationcenter/beta/Tutorials/MachineLearningExamples/CensusIncomeDataSet.html
    """
    filename = os.path.join(save_folder, f"census_income_{seed}.csv")
    if not os.path.exists(filename):
        data = fetch_openml(data_id = 42750, as_frame = True, parser = "auto", cache = True)
        df = data.frame

        # Drop numeric numbers that don't make sense as categorical features
        df.drop(columns=["unknown","wage_per_hour","stock_dividends","capital_gains","capital_losses"], inplace=True)

        # Create multiple categorical features:
        #  1. 10 hashes from random 25% of categorical columns combined
        #  2. 10 hashes from random 40% of categorical columns combined
        np.random.seed(seed)
        columns = list(df.columns)
        columns.remove("income_50k")  

        max_hash_value = 100_000_000

        for i in range(10):
            np.random.shuffle(columns)
            df[f"features_{i}"] = df[columns[:int(len(columns) * 0.25)]].apply(lambda x: hash(tuple(x)) % max_hash_value, axis=1)
        for i in range(10):
            np.random.shuffle(columns)
            df[f"features_{i + 10}"] = df[columns[:int(len(columns) * 0.4)]].apply(lambda x: hash(tuple(x)) % max_hash_value, axis=1)
        
        # Combine all categorical features into one value separated by "|"
        columns.extend( [f for f in df.columns if f.startswith("features_")] )
        df["features"] = df[columns].apply(lambda x: "|".join([f"{i}:{v}" for i, v in enumerate(x)]), axis=1)

        # Print column names with counts of unique values
        #for col in df.columns:
        #    print(f"{col}: {df[col].nunique()}")

        # Convert target column to 0/1. ("' 50000+.'" -> 1, "' - 50000.'" -> 0)
        df["target"] = df["income_50k"].apply(lambda x: 1 if x == "' 50000+.'" else 0)

        # Write to csv
        df[["features", "target"]].to_csv(filename, index=False)
    else:
        df = pd.read_csv(filename)

    return df

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
    #df = loader_airlines()
    df = loader_census_income()

    print(df.head(100))

    # Print column names with types
    print(df.dtypes)

    #print(compute_dataset_properties(df))

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


