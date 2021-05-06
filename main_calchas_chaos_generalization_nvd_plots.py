import pickle
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
        with open("data/chaos_data/" + file_name + ".dat", "rb") as fp:
            data_list = pickle.load(fp)
    except:
        data_list = []

    return data_list


def create_heatmap():
    my_colors = [(0.2, 0.3, 0.3), (0.4, 0.5, 0.4), (0.1, 0.7, 0), (0.1, 0.7, 0)]
    df_lp_cs_list = load_data("lyapunov_lles_cs_df_list")
    df_lp_os_list = load_data("lyapunov_lles_os_df_list")

    fig, ax = plt.subplots(figsize=(20, 15))
    df_lp_concat_cs_list = pd.concat(df_lp_cs_list, axis=1)
    df_lp_concat_cs_list.columns = ["%d" % i for i, _ in enumerate(df_lp_concat_cs_list.columns)]
    sns.heatmap(df_lp_concat_cs_list, ax=ax, cmap=my_colors, linewidths=.5)  # cmap = "YlGnBu" or cmap="RdYlGn"
    fig.savefig('data/chaos_data/df_lp_concat_cs_list_heatmap.pdf', dpi=300)

    fig, ax = plt.subplots(figsize=(20, 15))
    df_lp_concat_os_list = pd.concat(df_lp_os_list, axis=1)
    df_lp_concat_os_list.columns = ["%d" % i for i, _ in enumerate(df_lp_concat_os_list.columns)]
    sns.heatmap(df_lp_concat_os_list, ax=ax, cmap=my_colors, linewidths=.5)  # cmap = "YlGnBu" or cmap="RdYlGn"
    plt.show()
    fig.savefig('data/chaos_data/df_lp_concat_os_list_heatmap.pdf', dpi=300)


def calculate_lle_stats():
    df_lp_cs_list = load_data("lyapunov_lles_cs_df_list")
    df_lp_os_list = load_data("lyapunov_lles_os_df_list")
    df_lp_concat_cs_list = pd.concat(df_lp_cs_list, axis=1)
    df_lp_concat_cs_list.columns = ["%d" % i for i, _ in enumerate(df_lp_concat_cs_list.columns)]
    df_lp_concat_os_list = pd.concat(df_lp_os_list, axis=1)
    df_lp_concat_os_list.columns = ["%d" % i for i, _ in enumerate(df_lp_concat_os_list.columns)]
    non_negative_values_cs = df_lp_concat_cs_list.where(df_lp_concat_cs_list > 0.0).count().sum()
    non_negative_values_os = df_lp_concat_os_list.where(df_lp_concat_os_list > 0.0).count().sum()
    print("Total values for cs: " + str(500 * 144))
    print("Non negative values for cs: " + str(non_negative_values_cs))
    print("Percentage: " + str((non_negative_values_cs * 100) / 72000))
    print("Total values for os: " + str(500 * 144))
    print("Non negative values for os: " + str(non_negative_values_os))
    print("Percentage: " + str((non_negative_values_os * 100) / 72000))


def calculate_he_ranges_stats():
    df_he_cs_list = load_data("hurst_exponent_cs_df_list")
    df_he_os_list = load_data("hurst_exponent_os_df_list")
    df_he_concat_cs_list = pd.concat(df_he_cs_list)
    df_he_concat_cs_list.reset_index(inplace=True, drop=True)
    df_he_concat_cs_list = df_he_concat_cs_list.rename(columns={'value': 'a'})
    df_he_concat_os_list = pd.concat(df_he_os_list)
    df_he_concat_os_list.reset_index(inplace=True, drop=True)
    df_he_concat_os_list = df_he_concat_os_list.rename(columns={'value': 'a'})
    ranges = [0, 0.20, 0.60, 1]
    range_vals_cs = df_he_concat_cs_list.groupby(pd.cut(df_he_concat_cs_list.a, ranges)).count()
    range_vals_os = df_he_concat_os_list.groupby(pd.cut(df_he_concat_os_list.a, ranges)).count()
    print("HE ranges stats for cs: ")
    print(range_vals_cs)
    print("HE ranges stats for os: ")
    print(range_vals_os)


def calculate_se_histogram():
    df_se_cs_list = load_data("sample_entropy_cs_df_list")
    df_se_os_list = load_data("sample_entropy_os_df_list")

    df_se_concat_cs_list = pd.concat(df_se_cs_list)
    df_se_concat_cs_list.reset_index(inplace=True, drop=True)
    df_se_concat_cs_list = df_se_concat_cs_list.rename(columns={'value': 'a'})
    df_se_concat_cs_list = df_se_concat_cs_list.replace([np.inf, -np.inf], np.nan)

    df_se_concat_os_list = pd.concat(df_se_os_list)
    df_se_concat_os_list.reset_index(inplace=True, drop=True)
    df_se_concat_os_list = df_se_concat_os_list.rename(columns={'value': 'a'})
    df_se_concat_os_list = df_se_concat_os_list.replace([np.inf, -np.inf], np.nan)

    df_se_concat_cs_list.hist()
    plt.xlabel('Sample entropy bin values')
    plt.ylabel('Frequency')
    plt.title('Sample entropy Histogram for closed source systems')
    plt.savefig('data/chaos_data/df_se_concat_cs_list_hist.pdf', dpi=300)
    plt.show()

    df_se_concat_os_list.hist()
    plt.xlabel('Sample entropy bin values')
    plt.ylabel('Frequency')
    plt.title('Sample entropy Histogram for open source systems')
    plt.savefig('data/chaos_data/df_se_concat_os_list_hist.pdf', dpi=300)
    plt.show()


def df_count_freq(df, thresh=0):
    # count how many are greater than a threshold `thresh` per row
    c = df.gt(thresh).sum(1)

    # find where `counts` are > `0` for both dataframes
    # conveniently dropped into one dataframe so we can do
    # this nifty `groupby` trick
    mask = c.gt(0).groupby(level=[1, 2]).transform('all')
    #                                    \-------/
    #                         This is key to broadcasting over
    #                         original index rather than collapsing
    #                         over the index levels we grouped by

    #     create a new column named `counts`
    #         /------------\
    return df.assign(counts=c)[mask]
    #                         \--/
    #                    filter with boolean mask


def df_lle_positive_count():
    df_lp_cs_list = load_data("lyapunov_lles_cs_df_list")
    df_lp_os_list = load_data("lyapunov_lles_os_df_list")

    df_lp_concat_cs_list = pd.concat(df_lp_cs_list, axis=1)
    df_lp_concat_cs_list.columns = ["%d" % i for i, _ in enumerate(df_lp_concat_cs_list.columns)]

    df_lp_concat_os_list = pd.concat(df_lp_os_list, axis=1)
    df_lp_concat_os_list.columns = ["%d" % i for i, _ in enumerate(df_lp_concat_os_list.columns)]

    df_count_freq_cs = df_count_freq(df_lp_concat_cs_list)
    df_count_freq_cs = df_count_freq_cs.sort_values('counts', ascending=False)
    final_df_count_freq_cs = df_count_freq_cs.filter(['counts'], axis=1)

    df_count_freq_os = df_count_freq(df_lp_concat_os_list)
    df_count_freq_os = df_count_freq_os.sort_values('counts', ascending=False)
    final_df_count_freq_os = df_count_freq_os.filter(['counts'], axis=1)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(final_df_count_freq_cs)
        print(final_df_count_freq_os)


# ======================================================================================================================
# Main function
# ======================================================================================================================
# noinspection PyTypeChecker
def main():
    print("Welcome to Calchas chaos generalization plot creation...")
    # Generate heatmap
    # create_heatmap()

    # LLEs
    # calculate_lle_stats()

    # Hurst exponent
    # calculate_he_ranges_stats()

    # Sample entropy
    # calculate_se_histogram()

    # Display dimensions for positive LLEs count for all systems
    df_lle_positive_count()


if __name__ == "__main__":
    main()
