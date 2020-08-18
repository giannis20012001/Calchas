# Modules import
import os
import gzip
import json
import shutil
import requests
import datetime
from glob import glob
from os import listdir
from db_models import *
from os.path import isfile, join
from sqlalchemy.sql import text
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# ======================================================================================================================
# Functions space
# ======================================================================================================================
def download(url: str, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))


def extract_all_files_to_dir(source_dir: str, dest_dir: str):
    for src_name in glob(os.path.join(source_dir, '*.gz')):
        base_path = os.path.basename(src_name)
        dest_name = os.path.join(dest_dir, base_path[:-3])
        with gzip.open(src_name, 'rb') as infile:
            with open(dest_name, 'wb') as outfile:
                for line in infile:
                    outfile.write(line)

        print('Added: ' + dest_name)


def extract_file_to_dir(source_file: str, dest_file: str):
    with gzip.open(source_file, 'rb') as f_in:
        with open(dest_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    print('Added: ' + dest_file)


def decoded_json_data_from_file(cve_items_dict):
    print("Decoded JSON Data From File")
    for cve_item_dict in cve_items_dict:
        for key, value in cve_item_dict.items():
            print(key, ":", value)

    print("Done reading json file")


def sqlalchemy_engine_start():
    engine = create_engine('sqlite:////home/lumi/Dropbox/unipi/paper_NVD_forcasting/sqlight_db/nvd_nist.db',
                           echo=True)
    Session = sessionmaker(bind=engine)
    session = Session()
    base.Base.metadata.create_all(engine, checkfirst=True)

    return engine, session


def get_cve_items_from_feeds(cve_feeds_dict):
    print("Create CVE_Items dictionary...")
    cve_items_dict = []
    for cve_feed_dict in cve_feeds_dict:
        for key, value in cve_feed_dict.items():
            if key == "CVE_Items":
                cve_items_dict.append(value)

    return cve_items_dict


def check_keys_exists(element, *keys):
    """
    Check if *keys (nested) exists in `element` (dict).
    """
    if not isinstance(element, dict):
        raise AttributeError('keys_exists() expects dict as first argument.')
    if len(keys) == 0:
        raise AttributeError('keys_exists() expects at least two arguments, one given.')

    _element = element
    for key in keys:
        try:
            if isinstance(_element, dict):
                _element = _element[key]
            elif isinstance(_element, list):
                for __element in _element:
                    _element = __element[key]
        except KeyError:
            return False
    return True


def create_tables():
    print("Creating basic & sub tables in sqlite...")
    engine, session = sqlalchemy_engine_start()
    print("Available tables before creation: ", engine.table_names())
    # Create tables
    cve_item_model.CveItem().__table__.create(engine, checkfirst=True)
    openstack_compute_server_model.OpenstackComputeServer().__table__.create(engine, checkfirst=True)
    openstack_controller_server_model.OpenstackControllerServer().__table__.create(engine, checkfirst=True)
    ubuntu_application_server_model.UbuntuApplicationServer().__table__.create(engine, checkfirst=True)
    ubuntu_database_server_model.UbuntuDatabaseServer().__table__.create(engine, checkfirst=True)
    ubuntu_mail_server_model.UbuntuMailServer().__table__.create(engine, checkfirst=True)
    microsoft_application_server_model.MicrosoftApplicationServer().__table__.create(engine, checkfirst=True)
    microsoft_database_server_model.MicrosoftDatabaseServer().__table__.create(engine, checkfirst=True)
    microsoft_mail_server_model.MicrosoftMailServer().__table__.create(engine, checkfirst=True)
    active_directory_model.ActiveDirectory().__table__.create(engine, checkfirst=True)
    apache_model.Apache().__table__.create(engine, checkfirst=True)
    clamav_model.Clamav().__table__.create(engine, checkfirst=True)
    dot_Net_model.DotNet().__table__.create(engine, checkfirst=True)
    fail2ban_model.Fail2Ban().__table__.create(engine, checkfirst=True)
    iis_model.Iis().__table__.create(engine, checkfirst=True)
    iptables_model.IpTables().__table__.create(engine, checkfirst=True)
    java_model.Java().__table__.create(engine, checkfirst=True)
    jboss_model.Jboss().__table__.create(engine, checkfirst=True)
    ldap_model.Ldap().__table__.create(engine, checkfirst=True)
    linux_kernel_model.LinuxKernel().__table__.create(engine, checkfirst=True)
    mcafee_model.Mcafee().__table__.create(engine, checkfirst=True)
    memcached_model.MemCached().__table__.create(engine, checkfirst=True)
    microsoft_exchange_model.MicrosoftExchange().__table__.create(engine, checkfirst=True)
    microsoft_sql_server_model.MicrosoftSqlServer().__table__.create(engine, checkfirst=True)
    mongodb_model.MongoDb().__table__.create(engine, checkfirst=True)
    mysql_model.MySql().__table__.create(engine, checkfirst=True)
    ntp_model.Ntp().__table__.create(engine, checkfirst=True)
    openstack_ceilometer_model.OpenstackCeilometer().__table__.create(engine, checkfirst=True)
    openstack_glance_model.OpenstackGlance().__table__.create(engine, checkfirst=True)
    openstack_horizon_model.OpenstackHorizon().__table__.create(engine, checkfirst=True)
    openstack_keystone_model.OpenstackKeystone().__table__.create(engine, checkfirst=True)
    openstack_neutron_model.OpenstackNeutron().__table__.create(engine, checkfirst=True)
    openstack_nova_model.OpenstackNova().__table__.create(engine, checkfirst=True)
    postgres_model.Postgres().__table__.create(engine, checkfirst=True)
    rabbitmq_model.RabbitMq().__table__.create(engine, checkfirst=True)
    spamassassin_model.SpamAssassin().__table__.create(engine, checkfirst=True)
    ubuntu_os_model.UbuntuOs().__table__.create(engine, checkfirst=True)
    ufw_model.Ufw().__table__.create(engine, checkfirst=True)
    windows_os_model.WindowsOs().__table__.create(engine, checkfirst=True)
    zimbra_model.Zimbra().__table__.create(engine, checkfirst=True)
    # Check things upon finishing
    print("Available tables after creation: ", engine.table_names())


def fill_data_to_main_tables(cve_items_dict):
    # TODO: Add check if tables are full or not.
    print("Filling data to main tables...")
    engine, session = sqlalchemy_engine_start()

    while True:
        try:
            choice = int(input("Enter 1 fot filling table cve_items...\n"
                               "Enter 2 fot filling table microsoft_application_server...\n"
                               "Enter 3 fot filling table microsoft_database_server...\n"
                               "Enter 4 fot filling table microsoft_mail_server...\n"
                               "Enter 5 fot filling table openstack_compute_server...\n"
                               "Enter 6 fot filling table openstack_controller_server...\n"
                               "Enter 7 fot filling table ubuntu_application_server...\n"
                               "Enter 8 fot filling table ubuntu_database_server...\n"
                               "Enter 9 fot filling table ubuntu_mail_server...\n"
                               "Enter -1 to exit third step subroutine execution...\n"))
        except ValueError:
            print("You entered a wrong choice...\n\n")
        else:
            if choice == 1:
                # Fill data to table cve_items
                counter = 0
                cve_items = []
                print("Filling data in cve_items table...")
                for cve_item_dict in cve_items_dict:
                    for cve in cve_item_dict:
                        # Get multiple description values from dictionary
                        summary_list = ""
                        for internal_data_item in cve['cve']['description']['description_data']:
                            if not summary_list:
                                summary_list = summary_list + str(internal_data_item['value'])
                            else:
                                summary_list = summary_list + "," + str(internal_data_item['value'])
                        # Get multiple URLs from dictionary
                        urls_list = ""
                        for internal_data_item in cve['cve']['references']['reference_data']:
                            if not urls_list:
                                urls_list = urls_list + str(internal_data_item['url'])
                            else:
                                urls_list = urls_list + "," + str(internal_data_item['url'])
                        # Get vulnerable software from dictionary
                        vulnerable_software_list = ""
                        if check_keys_exists(cve, "configurations", "nodes", "children"):
                            for internal_data_item in cve['configurations']['nodes']:
                                for _internal_data_item in internal_data_item['children']:
                                    for __internal_data_item in _internal_data_item['cpe_match']:
                                        if not vulnerable_software_list:
                                            vulnerable_software_list = vulnerable_software_list \
                                                                       + str(__internal_data_item['cpe23Uri'])
                                        else:
                                            vulnerable_software_list = vulnerable_software_list + "," \
                                                                       + str(__internal_data_item['cpe23Uri'])
                        elif check_keys_exists(cve, "configurations", "nodes", "cpe_match"):
                            for internal_data_item in cve['configurations']['nodes']:
                                for _internal_data_item in internal_data_item['cpe_match']:
                                    if not vulnerable_software_list:
                                        vulnerable_software_list = vulnerable_software_list \
                                                                   + str(_internal_data_item['cpe23Uri'])
                                    else:
                                        vulnerable_software_list = vulnerable_software_list + "," \
                                                                   + str(_internal_data_item['cpe23Uri'])
                        # Get Time as Datetime object
                        try:
                            timestamp = datetime.datetime.strptime(str(cve.get('publishedDate')), "%Y-%m-%dT%H:%M:%SZ")
                        except ValueError:
                            timestamp = datetime.datetime.strptime(str(cve.get('publishedDate')), "%Y-%m-%dT%H:%MZ")

                        cve_item = CveItem(
                            cve_id=cve['cve'].get('CVE_data_meta', {}).get('ID'),
                            published_datetime=timestamp,
                            score=cve['impact'].get('baseMetricV2', {}).get('cvssV2', {}).get('baseScore'),
                            access_vector=cve['impact'].get('baseMetricV2', {}).get('cvssV2', {}).get('accessVector'),
                            access_complexity=cve['impact'].get('baseMetricV2', {}).get('cvssV2', {}).get(
                                'accessComplexity'),
                            authentication=cve['impact'].get('baseMetricV2', {}).get('cvssV2', {})
                                .get('authentication'),
                            availability_impact=cve['impact'].get('baseMetricV2', {}).get('cvssV2', {}).get(
                                'availabilityImpact'),
                            confidentiality_impact=cve['impact'].get('baseMetricV2', {}).get('cvssV2', {})
                                .get('confidentialityImpact'),
                            integrity_impact=cve['impact'].get('baseMetricV2', {}).get('cvssV2', {})
                                .get('integrityImpact'),
                            last_modified_datetime=cve.get('lastModifiedDate'),
                            urls=urls_list,
                            summary=summary_list,
                            vulnerable_software_list=vulnerable_software_list
                        )

                        cve_items.append(cve_item)
                        counter += 1

                # Add data to table
                print("Total item filed: " + str(counter))
                session.add_all(cve_items)
                session.commit()

            elif choice == 2:
                # Fill data to table microsoft_application_server
                print("Filling data in microsoft_application_server table...")
                query = "INSERT INTO microsoft_application_server " \
                        "SELECT cve_id, score,date(published_datetime) AS pdate, vulnerable_software_list AS vlist " \
                        "FROM cve_items " \
                        "where LOWER(vulnerable_software_list) LIKE '%mcafee%' " \
                        "or LOWER(vulnerable_software_list) LIKE '%microsoft%windows%' " \
                        "or LOWER(vulnerable_software_list) LIKE '%microsoft%active%directory%' " \
                        "or LOWER(vulnerable_software_list) LIKE '%.net%framework%' " \
                        "or LOWER(vulnerable_software_list) LIKE '%microsoft%iis%' " \
                        "ORDER BY pdate"
                connection = engine.connect()
                connection.execute(query)
            elif choice == 3:
                # Fill data to table microsoft_database_server
                print(choice)
            elif choice == 4:
                # Fill data to table microsoft_mail_server
                print(choice)
            elif choice == 5:
                # Fill data to table openstack_compute_server
                print(choice)
            elif choice == 6:
                # Fill data to table openstack_controller_server
                print(choice)
            elif choice == 7:
                # Fill data to table ubuntu_application_server
                print(choice)
            elif choice == 8:
                # Fill data to table ubuntu_database_server
                print(choice)
            elif choice == 9:
                # Fill data to table ubuntu_mail_server
                print(choice)
            elif choice == -1:
                print("Exiting current third step subroutine execution...\n\n")
                break
            else:
                print("You entered a wrong choice...\n\n")

    print("Data entry to main tables has finished...\n\n")


def fill_data_to_sub_tables(cve_items_dict):
    print("Fill data to sub tables...")
    # Fill data to table active_directory
    # Fill data to table apache
    # Fill data to table clamav
    # Fill data to table dot_Net
    # Fill data to table fail2ban
    # Fill data to table iis
    # Fill data to table iptables
    # Fill data to table java
    # Fill data to table jboss
    # Fill data to table ldap
    # Fill data to table linux_kernel
    # Fill data to table mcafee
    # Fill data to table memcached
    # Fill data to table microsoft_exchange
    # Fill data to table microsoft_sql_server
    # Fill data to table mongodb
    # Fill data to table mysql
    # Fill data to table ntp
    # Fill data to table openstack_ceilometer
    # Fill data to table openstack_glance
    # Fill data to table openstack_horizon
    # Fill data to table openstack_keystone
    # Fill data to table openstack_neutron
    # Fill data to table openstack_nova
    # Fill data to table postgres
    # Fill data to table rabbitmq
    # Fill data to table spamassassin
    # Fill data to table ubuntu_os
    # Fill data to table ufw
    # Fill data to table windows_os
    # Fill data to table zimbra


# ======================================================================================================================
# Main function
# ======================================================================================================================
def main():
    print("Welcome to Calchas... Please choose step...")
    while True:
        try:
            choice = int(input("Enter 1 to to run first step --> Download raw data from NVD...\n"
                               "Enter 2 to to run second step --> "
                               "Extract data from downloaded NVD .gz files & create json files...\n"
                               "Enter 3 to to run third step  --> "
                               "Extract CVE_Items[] from json files, perform feature extraction "
                               "(dimensionality reduction) & save them to sqlite...\n"
                               "Enter 4 to run fourth step ---> ...\n"
                               "Enter -1 to to stop program execution...\n"))
        except ValueError:
            print("You entered a wrong choice...\n\n")
        else:
            if choice == 1:
                # First step.
                # Create folder if it does not exist.
                if not os.path.exists('data/nvd_raw_data'):
                    os.makedirs('data/nvd_raw_data')
                # Download files from `url` raw data & save it in a temporary directory.
                print("Started first step! Downloading raw data...")
                year = 2002
                while year <= 2020:
                    # Path to it in the `file_name` variable:
                    file_name = "nvdcve-1.1-" + str(year) + ".json.gz"
                    url = "https://nvd.nist.gov/feeds/json/cve/1.1/" + file_name
                    # Download the file from `url` and save it locally under `file_name` if it does not exist:
                    if not os.path.isfile('data/nvd_raw_data/' + file_name):
                        download(url, dest_folder='data/nvd_raw_data/')
                    year = year + 1

                print("First step has been finished...\n\n")
            elif choice == 2:
                # Second step
                # Create folder if it does not exist
                if not os.path.exists('data/nvd_json_data'):
                    os.makedirs('data/nvd_json_data')
                # Extract data from downloaded .gz files & create json files
                print("Started second step! Extract data from downloaded .gz files...")
                file_names = [f for f in listdir('data/nvd_json_data') if isfile(join('data/nvd_json_data', f))]
                file_names.sort()
                if not file_names:
                    print("It seems the directory is empty, so the all the .gz files will be called...")
                    extract_all_files_to_dir('data/nvd_raw_data/', 'data/nvd_json_data/')
                else:
                    print("It seems the directory is partially/not empty, so only the missing .gz files "
                          "will be called...")
                    expected_file_names = [f for f in listdir('data/nvd_raw_data') if
                                           isfile(join('data/nvd_raw_data', f))]
                    expected_file_names = [s.strip('.gz') for s in expected_file_names]
                    for file in expected_file_names:
                        if not os.path.isfile('data/nvd_json_data/' + file):
                            extract_file_to_dir('data/nvd_raw_data/' + file + '.gz', 'data/nvd_json_data/' + file)

                print("Second step has been finished...\n\n")
            elif choice == 3:
                # Third step extract CVE_Items[] from json files,
                # perform feature extraction (dimensionality reduction) & save them to sqlite
                print("Started third step! Extract CVE_Items[] from the created json files & save them to sqlite...")
                cve_feeds_dict = []
                for file_name in glob('data/nvd_json_data/*.json'):
                    print("Converting JSON encoded data into Python dictionary...")
                    print("Started Reading JSON file: " + file_name + "...")
                    with open(file_name, "r") as read_file:
                        cve_feeds_dict.append(json.load(read_file))

                while True:
                    try:
                        _choice = int(input("Enter 1 to create tables...\n"
                                            "Enter 2 to fill data to main tables...\n"
                                            "Enter 3 to fill data to sub tables...\n"
                                            "Enter -1 to exit third step subroutine execution...\n"))
                    except ValueError:
                        print("You entered a wrong choice...\n\n")
                    else:
                        # Extract CVE_Items from each CVE_feed
                        cve_items_dict = get_cve_items_from_feeds(cve_feeds_dict)
                        if _choice == 1:
                            # Create initial NVD table & the various sub tables
                            create_tables()
                        elif _choice == 2:
                            # Fill data to NVD table
                            fill_data_to_main_tables(cve_items_dict)
                        elif _choice == 3:
                            # Then fill data to the rest of the sub tables
                            fill_data_to_sub_tables(cve_items_dict)
                        elif _choice == -1:
                            print("Exiting current third step subroutine execution...\n\n")
                            break
                        else:
                            print("You entered a wrong choice...\n\n")

                print("Third step has been finished...\n\n")
            elif choice == 4:
                print("Started fourth step!...")
                print("Fourth step has been finished...\n\n")
            elif choice == -1:
                print("Thank you for using Calchas! Now exiting program...\n\n")
                break
            else:
                print("You entered a wrong choice...\n\n")


if __name__ == "__main__":
    main()
