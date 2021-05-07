import pickle
import nolds
import numpy as np
import pandas as pd
import seaborn as sns
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
        with open("../data/chaos_data/" + file_name + ".dat", "rb") as fp:
            data_list = pickle.load(fp)
    except:
        data_list = []

    return data_list


# ======================================================================================================================
# Main function
# ======================================================================================================================
# noinspection PyTypeChecker
def main():
    # df_he_cs_list = load_data("hurst_exponent_cs_df_list")
    # df_he_os_list = load_data("hurst_exponent_os_df_list")
    #
    # df_he_concat_cs_list = pd.concat(df_he_cs_list, axis=1)
    # df_he_concat_cs_list.columns = ["%d" % i for i, _ in enumerate(df_he_concat_cs_list.columns)]
    # ranges = [0, 0.20, 0.60, 1]
    # df_he_concat_cs_list.groupby(pd.cut(df_he_concat_cs_list.a, ranges)).count()
    #
    # df_he_concat_os_list = pd.concat(df_he_os_list, axis=1)
    # df_he_concat_os_list.columns = ["%d" % i for i, _ in enumerate(df_he_concat_os_list.columns)]
    lm = nolds.logistic_map(0.1, 1000, r=4)
    x = np.fromiter(lm, dtype="float32")
    print(nolds.sampen(x,  emb_dim=3, tolerance=None))
    print(nolds.sampen(x, emb_dim=5, tolerance=None))
    print(nolds.sampen(x, emb_dim=7, tolerance=None))


if __name__ == "__main__":
    main()