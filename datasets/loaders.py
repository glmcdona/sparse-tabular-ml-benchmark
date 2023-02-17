import re
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import os

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

        df = pd.DataFrame({"text": data, "label": labels})

        # Split the text column into "|"-delimited words
        # using sklearn
        token_pattern = re.compile(r"(?u)\\b\\w\\w+\\b")
        tokenizer = token_pattern.findall
        #TODO: CONTINUE HERE
        df["features"] = df["text"].apply(tokenizer)

        df.to_csv(os.path.join(save_folder,"newsgroups.csv"), index=False)
    else:
        df = pd.read_csv(os.path.join(save_folder,"newsgroups.csv"))

    return df
    

