import time
import nolds
import random
import pickle
import sqlite3
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from numpy import nan
# from p_tqdm import p_map
from datetime import datetime
from pandas import concat, DataFrame
from joblib import Parallel, delayed
from sklearn.impute import KNNImputer


# ======================================================================================================================
# Functions space
# ======================================================================================================================
# transform list into supervised learning format
# noinspection Duplicates
def series_to_supervised(data, n_in=1, n_out=1):
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = concat(cols, axis=1)

    return agg.values


# noinspection DuplicatedCode,PyUnusedLocal
def calculate_fft(sig, dt=None, plot=True):
    # here it's assumes analytic signal (real signal...) - so only half of the axis is required
    if dt is None:
        dt = 1
        t = np.arange(0, sig.shape[-1])
        x_label = 'samples'
    else:
        t = np.arange(0, sig.shape[-1]) * dt
        x_label = 'freq [Hz]'

    if sig.shape[0] % 2 != 0:
        # warnings.warn("signal preferred to be even in size, autoFixing it...")
        t = t[0:-1]
        sig = sig[0:-1]

    sig_fft = np.fft.fft(sig) / t.shape[0]  # divided by size t for coherent magnitude
    freq = np.fft.fftfreq(t.shape[0], d=dt)

    np_fft = np.fft.fft(sig)
    n_samples = sig.size
    amplitudes = 2 / n_samples * np.abs(np_fft)
    amplitudes = np.abs(amplitudes)

    # plot analytic signal - right half of freq axis needed only...
    first_neg_ind = np.argmax(freq < 0)
    freq_axis_pos = freq[0:first_neg_ind]
    sig_fft_pos = 2 * sig_fft[0:first_neg_ind]  # *2 because of magnitude of analytic signal

    first_ten = 0
    for x in np.nditer(np.abs(sig_fft_pos[1:11])):
        first_ten = first_ten + x

    last_ten = 0
    for x in np.nditer(np.abs(sig_fft_pos[-10:])):
        last_ten = last_ten + x

    # first_ten = 0
    # for x in np.nditer(amplitudes[1:11]):
    #     first_ten = first_ten + x
    #
    # last_ten = 0
    # for x in np.nditer(amplitudes[-10:]):
    #     last_ten = last_ten + x

    # if plot:
    #     plt.figure()
    #     plt.plot(freq_axis_pos, np.abs(sig_fft_pos))
    #     plt.xlabel(x_label)
    #     plt.ylabel('mag')
    #     plt.title('Analytic FFT plot')
    #     plt.show()

    return last_ten / first_ten


def get_fft_ratio(ts):
    # Check for fft ratio
    D = ts.size
    time_interval = ts.index[-1] - ts.index[1]
    # fs is sampling frequency
    fs = float(D / time_interval.days)
    result = np.abs(calculate_fft(ts.values, dt=fs))

    return result


# noinspection Duplicates
def fill_missing_values_knn(ts):
    # temp = ts.values.reshape(-1, 1)
    xtrans = series_to_supervised(ts.values, n_in=3)
    # Define imputer
    imputer = KNNImputer(n_neighbors=5)
    # Fit on the dataset
    imputer.fit(xtrans)
    # Transform the dataset
    x_filled_knn = imputer.transform(xtrans)
    # Find total column number for ndarray
    columns = len(x_filled_knn[0])
    # Get last column
    ts.iloc[:] = x_filled_knn[:, (columns - 1)]

    return ts


# noinspection Duplicates
def check_missing_value_percentage(ts, missing_values):
    result = get_fft_ratio(ts)
    missing_values_percentage_threshold = 0
    if result < 0.5:
        missing_values_percentage_threshold = 20
    elif result >= 0.5:
        missing_values_percentage_threshold = 10

    mv_flag = False
    start_year = 0
    end_year = 0
    for index_val, series_val in missing_values['Percentage_of_missing_values'].iteritems():
        if series_val <= missing_values_percentage_threshold:
            mv_flag = True
            start_year = missing_values['Start_Year_Range'][index_val]
            end_year = missing_values['Stop_Year_Range'][index_val]
            break

    return mv_flag, missing_values_percentage_threshold, start_year, end_year


# noinspection Duplicates
def calculate_missing_values_dataframe(ts_wmv, years_list, random_system_name, time_granularity_val):
    missing_values_df = DataFrame(index=range(len(years_list)), columns=['System_Examined',
                                                                         'Start_Year_Range',
                                                                         'Stop_Year_Range',
                                                                         'Total_num_of_Values',
                                                                         'Non_Zero_values',
                                                                         'Percentage_of_missing_values',
                                                                         'time_granularity'])
    missing_values_df['System_Examined'] = random_system_name
    missing_values_df['Start_Year_Range'] = years_list
    # missing_values_df['Start_Year_Range'] = pd.to_datetime(missing_values_df['Start_Year_Range'])
    missing_values_df['Stop_Year_Range'] = years_list[-1]
    # missing_values_df['Stop_Year_Range'] = pd.to_datetime(missing_values_df['Stop_Year_Range'])
    missing_values_df['time_granularity'] = time_granularity_val

    # Count total number of values (zeros included)
    for i in range(len(years_list)):
        count = 0
        for index_val, series_val in ts_wmv.iteritems():
            if int(missing_values_df['Start_Year_Range'][i]) <= index_val.year <= \
                    int(missing_values_df['Stop_Year_Range'][i]):
                count = count + 1
        missing_values_df.iloc[i, missing_values_df.columns.get_loc('Total_num_of_Values')] = count

    # Count total number of none zero values
    for i in range(len(years_list)):
        count = 0
        for index_val, series_val in ts_wmv.iteritems():
            if (int(missing_values_df['Start_Year_Range'][i]) <= index_val.year <=
                int(missing_values_df['Stop_Year_Range'][i])) and (series_val > 0):
                count = count + 1
        missing_values_df.iloc[i, missing_values_df.columns.get_loc('Non_Zero_values')] = count

    # Calculate the data missing values percentage
    for i in range(len(years_list)):
        missing_values_df.iloc[i, missing_values_df.columns.get_loc('Percentage_of_missing_values')] = \
            ((missing_values_df['Total_num_of_Values'][i] - missing_values_df['Non_Zero_values'][i]) /
             missing_values_df['Total_num_of_Values'][i]) * 100

    return missing_values_df


# noinspection Duplicates
def check_eligible_range_for_day(initial_df, random_system_name):
    # create series from dataframe
    ts = pd.Series(initial_df['score'].values, index=initial_df['published_datetime'])
    ts.index = pd.DatetimeIndex(ts.index)
    # Create time series with all missing date values and fill them with zeros
    idx = pd.date_range(initial_df['published_datetime'].min(), initial_df['published_datetime'].max())
    # Create day time interval
    ts_wmv_day = ts.reindex(idx, fill_value=0)
    # Create missing values dataframe
    years_list = ts_wmv_day.index.strftime("%Y").drop_duplicates().tolist()
    missing_values = calculate_missing_values_dataframe(ts_wmv_day, years_list, random_system_name, "day")
    # Check for full time range
    eligible_range_for_day, missing_values_percentage_threshold, start_year, end_year = \
        check_missing_value_percentage(ts, missing_values)

    if eligible_range_for_day:
        for year in years_list:
            if year < start_year:
                ts_wmv_day = ts_wmv_day.drop(ts_wmv_day.index[ts_wmv_day.index.year.isin([int(year)])])
            elif year >= start_year:
                break
        # Series within the selected time range, with 0 replaced by NaN
        ts_wmv_day = ts_wmv_day.replace({0: nan})
        # ts_wmv_day = fill_missing_values_pca(ts_wmv_day)
        ts_wmv_day = fill_missing_values_knn(ts_wmv_day)

        return True, ts_wmv_day

    return False, None


# noinspection Duplicates
def check_eligible_range_for_week(initial_df, random_system_name):
    # create series from dataframe
    ts = pd.Series(initial_df['score'].values, index=initial_df['published_datetime'])
    ts.index = pd.DatetimeIndex(ts.index)
    # Create all missing date values and fill them with zero
    idx = pd.date_range(initial_df['published_datetime'].min(), initial_df['published_datetime'].max())
    ts_wmv_week = ts.reindex(idx, fill_value=0)
    # Create week time interval
    ts_wmv_week = ts_wmv_week.resample('W-MON', label='left', closed='left').max()
    # Create missing values dataframe
    years_list = ts_wmv_week.index.strftime("%Y").drop_duplicates().tolist()
    missing_values = calculate_missing_values_dataframe(ts_wmv_week, years_list, random_system_name, "week")
    # Check for full time range
    eligible_range_for_week, missing_values_percentage_threshold, start_year, end_year = \
        check_missing_value_percentage(ts, missing_values)

    if eligible_range_for_week:
        for year in years_list:
            if year < start_year:
                ts_wmv_week = ts_wmv_week.drop(ts_wmv_week.index[ts_wmv_week.index.year.isin([int(year)])])
            elif year >= start_year:
                break
        # Series within the selected time range, with 0 replaced by NaN
        ts_wmv_week = ts_wmv_week.replace({0: nan})
        # ts_wmv_week = fill_missing_values_pca(ts_wmv_week)
        ts_wmv_week = fill_missing_values_knn(ts_wmv_week)

        return True, ts_wmv_week

    return False, None


# noinspection Duplicates
def check_eligible_range_for_month(initial_df, random_system_name):
    # create series from dataframe
    ts = pd.Series(initial_df['score'].values, index=initial_df['published_datetime'])
    ts.index = pd.DatetimeIndex(ts.index)
    # Create all missing date values and fill them with zero
    idx = pd.date_range(initial_df['published_datetime'].min(), initial_df['published_datetime'].max())
    ts_wmv_month = ts.reindex(idx, fill_value=0)
    # Create week time interval
    ts_wmv_month = ts_wmv_month.resample('M-MON', label='left', closed='left').max()
    # Create missing values dataframe
    years_list = ts_wmv_month.index.strftime("%Y").drop_duplicates().tolist()
    missing_values = calculate_missing_values_dataframe(ts_wmv_month, years_list, random_system_name, "month")
    # Check for full time range
    eligible_range_for_month, missing_values_percentage_threshold, start_year, end_year = \
        check_missing_value_percentage(ts, missing_values)

    if eligible_range_for_month:
        for year in years_list:
            if year < start_year:
                ts_wmv_month = ts_wmv_month.drop(ts_wmv_month.index[ts_wmv_month.index.year.isin([int(year)])])
            elif year >= start_year:
                break
        # Series within the selected time range, with 0 replaced by NaN
        ts_wmv_month = ts_wmv_month.replace({0: nan})
        # ts_wmv_month = fill_missing_values_pca(ts_wmv_month)
        ts_wmv_month = fill_missing_values_knn(ts_wmv_month)

        return True, ts_wmv_month

    return False, None


def create_random_cs_system_dataset(idx, seed):
    print("Start creation of random closed source system " + str(idx + 1) + "...")

    # Create a SQL connection to SQLite database
    con = sqlite3.connect("/home/lumi/Dropbox/unipi/paper_NVD_forcasting/sqlight_db/nvd_nist.db")
    # prepare the lists of sequences
    closed_source_os_sequence = ['%microsoft%windows%']
    volatile_services_sequence = ['%iptables%', '%ntp%', '%fail2ban%', '%mysql%', '%mongodb%', '%postgres%',
                                  '%apache2%', '%rabbitmq%', '%activemq%', '%kafka%', '%jboss%', '%memcached%',
                                  '%redis%', '%gitlab%', '%elastic:%', '%haproxy%', '%traefik%', '%seesaw%',
                                  '%neutrino%', '%squid%', '%clamav%', '%ufw%', '%ldap%', '%zimbra%', '%spamassassin%',
                                  '%microsoft%active%directory%', '%.net%framework%', '%microsoft%iis%',
                                  '%microsoft%sql%server%', '%microsoft%exchange%', '%spamassassin%', '%mcafee%',
                                  '%kaspersky%', '%windows%defender%', '%avira%', '%bitdefender%', '%comodo%',
                                  '%avast%', '%f%secure%', '%sophos%']

    # Make random closed source system datasets
    # Select a subset without replacement
    random.seed(seed)
    subset_volatile_services_sequence = random.sample(volatile_services_sequence, 7)
    # Build the query string
    query = ("SELECT date(published_datetime) as published_datetime, score " +
             "FROM cve_items " +
             "WHERE LOWER(vulnerable_software_list) LIKE '" + closed_source_os_sequence[0] + "' " +
             "OR LOWER(vulnerable_software_list) LIKE '" + subset_volatile_services_sequence[0] + "' " +
             "OR LOWER(vulnerable_software_list) LIKE '" + subset_volatile_services_sequence[1] + "' " +
             "OR LOWER(vulnerable_software_list) LIKE '" + subset_volatile_services_sequence[2] + "' " +
             "OR LOWER(vulnerable_software_list) LIKE '" + subset_volatile_services_sequence[3] + "' " +
             "OR LOWER(vulnerable_software_list) LIKE '" + subset_volatile_services_sequence[4] + "' " +
             "OR LOWER(vulnerable_software_list) LIKE '" + subset_volatile_services_sequence[5] + "' " +
             "OR LOWER(vulnerable_software_list) LIKE '" + subset_volatile_services_sequence[6] + "' " +
             "ORDER BY published_datetime")
    # Read sqlite query results into a pandas DataFrame
    initial_df = pd.read_sql_query(query, con)
    # Performing initial value reduction
    initial_df = initial_df.loc[initial_df.groupby('published_datetime')['score'].idxmax()]
    initial_df.published_datetime = pd.to_datetime(initial_df.published_datetime)
    # Check if duplicate values exist in published_datetime column and keep the maximum value
    if initial_df.published_datetime.duplicated().any():
        # Remove duplicate values for the same date
        initial_df = initial_df.groupby('published_datetime').max()
        initial_df = initial_df.reset_index()

    # First check for day
    status_true, pd_ts = check_eligible_range_for_day(initial_df, ("random_system_" + str(idx + 1)))
    if status_true:
        return pd_ts

    # If day fails then check for week
    status_true, pd_ts = check_eligible_range_for_week(initial_df, ("random_system_" + str(idx + 1)))
    if status_true:
        return pd_ts

    # If week fails check for month
    status_true, pd_ts = check_eligible_range_for_month(initial_df, ("random_system_" + str(idx + 1)))
    if status_true:
        return pd_ts

    # Close connection when done
    con.close()

    return None


def create_random_os_system_dataset(idx, seed):
    print("Start creation of random open source system " + str(idx + 1) + "...")

    # Create a SQL connection to SQLite database
    con = sqlite3.connect("/home/lumi/Dropbox/unipi/paper_NVD_forcasting/sqlight_db/nvd_nist.db")
    # prepare the lists of sequences
    open_source_os_sequence = ['%ubuntu%', '%debian%', '%redhat%', '%centos%', '%fedora%']
    open_source_permanent_services_sequence = ['%linux_kernel:%']
    volatile_services_sequence = ['%iptables%', '%ntp%', '%fail2ban%', '%mysql%', '%mongodb%', '%postgres%',
                                  '%apache2%', '%rabbitmq%', '%activemq%', '%kafka%', '%jboss%', '%memcached%',
                                  '%redis%', '%gitlab%', '%elastic:%', '%haproxy%', '%traefik%', '%seesaw%',
                                  '%neutrino%', '%squid%', '%clamav%', '%ufw%', '%ldap%', '%zimbra%', '%spamassassin%',
                                  '%microsoft%active%directory%', '%.net%framework%', '%microsoft%iis%',
                                  '%microsoft%sql%server%', '%microsoft%exchange%', '%spamassassin%', '%mcafee%',
                                  '%kaspersky%', '%windows%defender%', '%avira%', '%bitdefender%', '%comodo%',
                                  '%avast%', '%f%secure%', '%sophos%']

    # Make random closed source system datasets
    # Select a subset without replacement
    random.seed(seed)
    subset_open_source_os_sequence = random.sample(open_source_os_sequence, 1)
    subset_volatile_services_sequence = random.sample(volatile_services_sequence, 6)
    # Build the query string
    query = ("SELECT date(published_datetime) as published_datetime, score " +
             "FROM cve_items " +
             "WHERE LOWER(vulnerable_software_list) LIKE '" + subset_open_source_os_sequence[0] + "' " +
             "OR LOWER(vulnerable_software_list) LIKE '" + open_source_permanent_services_sequence[0] + "' " +
             "OR LOWER(vulnerable_software_list) LIKE '" + subset_volatile_services_sequence[0] + "' " +
             "OR LOWER(vulnerable_software_list) LIKE '" + subset_volatile_services_sequence[1] + "' " +
             "OR LOWER(vulnerable_software_list) LIKE '" + subset_volatile_services_sequence[2] + "' " +
             "OR LOWER(vulnerable_software_list) LIKE '" + subset_volatile_services_sequence[3] + "' " +
             "OR LOWER(vulnerable_software_list) LIKE '" + subset_volatile_services_sequence[4] + "' " +
             "OR LOWER(vulnerable_software_list) LIKE '" + subset_volatile_services_sequence[5] + "' " +
             "ORDER BY published_datetime")
    # Read sqlite query results into a pandas DataFrame
    initial_df = pd.read_sql_query(query, con)
    # Performing initial value reduction
    initial_df = initial_df.loc[initial_df.groupby('published_datetime')['score'].idxmax()]
    initial_df.published_datetime = pd.to_datetime(initial_df.published_datetime)
    # Check if duplicate values exist in published_datetime column and keep the maximum value
    if initial_df.published_datetime.duplicated().any():
        # Remove duplicate values for the same date
        initial_df = initial_df.groupby('published_datetime').max()
        initial_df = initial_df.reset_index()

    # First check for day
    status_true, pd_ts = check_eligible_range_for_day(initial_df, ("random_system_" + str(idx + 1)))
    if status_true:
        return pd_ts

    # If day fails then check for week
    status_true, pd_ts = check_eligible_range_for_week(initial_df, ("random_system_" + str(idx + 1)))
    if status_true:
        return pd_ts

    # If week fails check for month
    status_true, pd_ts = check_eligible_range_for_month(initial_df, ("random_system_" + str(idx + 1)))
    if status_true:
        return pd_ts

    # Close connection when done
    con.close()

    return None


def perform_random_system_lles_calc(idx, system_type, system):
    print("Perform lyapunov exponent calculations of random " + system_type + " system " + str(idx + 1) + "...")
    # load results dataframe
    lyapunov_exps_df = pd.read_csv('data/chaos_data/base_data_lle.csv')
    lyapunov_exps_df = lyapunov_exps_df.set_index(['trajectory_len', 'emb_dim', 'min_neighbors', 'lag']).sort_index()
    # load data
    x = system.values
    x = x.astype('float32')
    # x = np.fromiter(cs_system.values, dtype="float32")

    # Perform the calculation and put it on the above created dataframe
    for idx_vals, values in lyapunov_exps_df['value'].iteritems():
        lyapunov_exps_df.at[(int(idx_vals[0]), int(idx_vals[1]), int(idx_vals[2]), int(idx_vals[3])), 'value'] = \
            nolds.lyap_r(x, emb_dim=int(idx_vals[1]), lag=int(idx_vals[3]), min_tsep=None,
                         min_neighbors=int(idx_vals[2]), trajectory_len=int(idx_vals[0]))

    return lyapunov_exps_df


def perform_random_system_he_calc(idx, system_type, system):
    print("Perform lyapunov exponent calculations of random " + system_type + " system " + str(idx + 1) + "...")
    # load results dataframe
    hurst_exp_df = pd.read_csv('data/chaos_data/base_data_he.csv')
    hurst_exp_df = hurst_exp_df.set_index('index')
    # load data
    x = system.values
    x = x.astype('float32')
    # x = np.fromiter(series.values, dtype="float32")

    # Perform the calculation and put it on the above created dataframe
    for row_index, row_val in hurst_exp_df.iterrows():
        hurst_exp_df.at[(int(row_index)), 'value'] = nolds.hurst_rs(x)

    return hurst_exp_df


def perform_random_system_se_calc(idx, system_type, system):
    print("Perform sample entropy calculations of random " + system_type + " system " + str(idx + 1) + "...")
    # load results dataframe
    sample_entropy_df = pd.read_csv('data/chaos_data/base_data_se.csv')
    sample_entropy_df = sample_entropy_df.set_index('emb_dim')
    print()
    # load data
    x = system.values
    x = x.astype('float32')
    # x = np.fromiter(series.values, dtype="float32")

    # Perform the calculation and put it on the above created dataframe
    for row_index, row_val in sample_entropy_df.iterrows():
        sample_entropy_df.at[(int(row_index)), 'value'] = nolds.sampen(x, emb_dim=row_index, tolerance=None)

    return sample_entropy_df


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
    print("Welcome to Calchas chaos generalization...")

    num_cores = multiprocessing.cpu_count()
    seed_table = [0] * 500  # 500
    for x in range(len(seed_table)):
        time.sleep(0.5)
        seed_table[x] = datetime.now()

    cs_systems_list = Parallel(n_jobs=num_cores) \
        (delayed(create_random_cs_system_dataset)(idx, seed) for idx, seed in enumerate(tqdm(seed_table)))
    save_data(cs_systems_list, "cs_systems_list")

    seed_table = [0] * 500  # 500
    for x in range(len(seed_table)):
        time.sleep(0.5)
        seed_table[x] = datetime.now()

    os_systems_list = Parallel(n_jobs=num_cores) \
        (delayed(create_random_os_system_dataset)(idx, seed) for idx, seed in enumerate(tqdm(seed_table)))
    save_data(os_systems_list, "os_systems_list")

    # Perform lyapunov exponent calculations
    lyapunov_lles_cs_df_list = Parallel(n_jobs=num_cores)(delayed(perform_random_system_lles_calc)
                                                          (idx, "closed source", cs_system) for idx, cs_system
                                                          in enumerate(tqdm(cs_systems_list)))
    save_data(lyapunov_lles_cs_df_list, "lyapunov_lles_cs_df_list")
    lyapunov_lles_os_df_list = Parallel(n_jobs=num_cores)(delayed(perform_random_system_lles_calc)
                                                          (idx, "open source", os_system) for idx, os_system
                                                          in enumerate(tqdm(os_systems_list)))
    save_data(lyapunov_lles_os_df_list, "lyapunov_lles_os_df_list")

    # Perform hurst exponent calculations
    hurst_exponent_cs_df_list = Parallel(n_jobs=num_cores)(delayed(perform_random_system_he_calc)
                                                           (idx, "closed source", cs_system) for idx, cs_system
                                                           in enumerate(tqdm(cs_systems_list)))
    save_data(hurst_exponent_cs_df_list, "hurst_exponent_cs_df_list")
    hurst_exponent_os_df_list = Parallel(n_jobs=num_cores)(delayed(perform_random_system_he_calc)
                                                           (idx, "open source", os_system) for idx, os_system
                                                           in enumerate(tqdm(os_systems_list)))
    save_data(hurst_exponent_os_df_list, "hurst_exponent_os_df_list")

    # Perform sample entropy calculations
    perform_random_system_se_calc(0, "closed source", cs_systems_list[0])
    sample_entropy_cs_df_list = Parallel(n_jobs=num_cores)(delayed(perform_random_system_se_calc)
                                                           (idx, "closed source", cs_system) for idx, cs_system
                                                           in enumerate(tqdm(cs_systems_list)))
    save_data(sample_entropy_cs_df_list, "sample_entropy_cs_df_list")
    sample_entropy_os_df_list = Parallel(n_jobs=num_cores)(delayed(perform_random_system_se_calc)
                                                           (idx, "open source", os_system) for idx, os_system
                                                           in enumerate(tqdm(os_systems_list)))
    save_data(sample_entropy_os_df_list, "sample_entropy_os_df_list")

    print()

if __name__ == "__main__":
    main()
