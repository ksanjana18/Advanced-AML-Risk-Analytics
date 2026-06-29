import pandas as pd

def filter_transactions(data, bank, amount):

    filtered = data[(data["Bank"] == bank) & (data["Amount"] == amount)]

    return filtered


def nearby_transactions(data, amount, range_value=5000):

    lower = amount - range_value
    upper = amount + range_value

    nearby = data[(data["Amount"] >= lower) & (data["Amount"] <= upper)]

    return nearby