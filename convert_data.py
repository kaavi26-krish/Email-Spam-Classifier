import pandas as pd

data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

data.to_csv("spam.csv", index=False)
print("Dataset converted successfully!")