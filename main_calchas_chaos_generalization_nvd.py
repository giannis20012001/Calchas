import sqlite3
import pandas as pd
from p_tqdm import p_map


# ======================================================================================================================
# Functions space
# ======================================================================================================================
def create_random_systems():
    print("Star creation of random systems...")

    # Create a SQL connection to SQLite database that holds final tables
    con = sqlite3.connect("/home/lumi/Dropbox/unipi/paper_NVD_forcasting/sqlight_db/nvd_nist.db")
    # Read sqlite query results into a pandas DataFrame
    query = "SELECT published_datetime, score from " + table_name
    query = "INSERT INTO microsoft_application_server(cve_id, published_datetime, score, " \
            "vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "where LOWER(vlist) LIKE '%mcafee%' " \
            "or LOWER(vlist) LIKE '%microsoft%windows%' " \
            "or LOWER(vlist) LIKE '%microsoft%active%directory%' " \
            "or LOWER(vlist) LIKE '%.net%framework%' " \
            "or LOWER(vlist) LIKE '%microsoft%iis%' " \
            "ORDER BY pdate"

    initial_df = pd.read_sql_query(query, con)
    # Close connection when done
    con.close()

    # Value reduction
    print("Performing value reduction in final tables...")
    # microsoft_application_server_final
    query = "INSERT INTO microsoft_application_server_final(cve_id, published_datetime, score, " \
            "vulnerable_software_list) " \
            "SELECT tt.* " \
            "FROM microsoft_application_server AS tt " \
            "INNER JOIN " \
            "(SELECT published_datetime, MAX(score) as MaxScore " \
            "FROM microsoft_application_server " \
            "GROUP BY published_datetime) AS groupedtt " \
            "ON  tt.score = groupedtt.MaxScore " \
            "AND tt.published_datetime = groupedtt.published_datetime;"


# ======================================================================================================================
# Main function
# ======================================================================================================================
def main():
    print("Welcome to Calchas chaos generalization...")
    p_map(create_random_systems(),)

if __name__ == "__main__":
    main()
