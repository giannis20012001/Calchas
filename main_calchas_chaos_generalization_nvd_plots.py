import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================================================================================
# Functions space
# ======================================================================================================================
def save_data(data_list, file_name):
    with open("data/chaos_data/" + file_name + ".dat", "wb") as fp:
        pickle.dump(data_list, fp)


# noinspection PyBroadException
def load_data(file_name):
    try:
        with open("data/chaos_data/" + file_name + ".dat", "rb") as fp:
            data_list = pickle.load(fp)
    except:
        data_list = []

    return data_list


# ======================================================================================================================
# Main function
# ======================================================================================================================
# noinspection PyTypeChecker
def main():
    print("Welcome to Calchas chaos generalization plot creation...")
    df_list = load_data("lyapunov_lles_cs_df_list")

    print()


if __name__ == "__main__":
    main()
