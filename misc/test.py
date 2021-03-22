import pandas as pd

# load data
lyapunov_df = pd.read_csv('../data/chaos_data/base_data_lles.csv')
lyapunov_df = lyapunov_df.set_index(['trajectory_len', 'emb_dim', 'min_neighbors', 'lag']).sort_index()
lyapunov_df.at[(6, 3, 2, 1), 'value'] = 45

# with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None,
#                        'display.max_colwidth', -1):
#     print(lyapunov_df)

for user_id, val in lyapunov_df['value'].iteritems():
    print (user_id[0])
    print(user_id[1])
    print(user_id[2])
    print(user_id[3])
    print (val)
