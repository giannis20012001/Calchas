# Modules import
import os
import gzip
import json
import shutil
import sqlite3
import requests
import datetime
import warnings
import numpy as np
import pandas as pd
# noinspection PyUnresolvedReferences
import matplotlib.pyplot as plt
from glob import glob
from os import listdir
from db_models import *
from numpy import nan, isnan
from pandas import DataFrame
from os.path import isfile, join
from sqlalchemy import create_engine
from sklearn.impute import KNNImputer
from sqlalchemy.orm import sessionmaker
from forecast_models import common_steps
from forecast_models.multilayer_perceptron_models import mlp_modeling
from forecast_models.convolutional_neural_network_models import cnn_modeling
from forecast_models.arma_models import arma_modeling_steps, arma_validation_steps
from forecast_models.arima import arima_modeling_steps, arima_validation_steps
from forecast_models.recurrent_neural_network_models import lstm_modeling, cnn_lstm_modeling, conv_lstm_modeling


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
    engine = create_engine("sqlite:////home/lumi/Dropbox/unipi/paper_NVD_forcasting/sqlight_db/nvd_nist.db",
                           echo=True)
    Session = sessionmaker(bind=engine)
    session = Session()
    base.Base.metadata.create_all(engine, checkfirst=True)

    return engine, session


def get_cve_items_from_feeds(cve_feeds_dict):
    print("Creating CVE_Items dictionary...")
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


# noinspection PyUnresolvedReferences
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
            choice = int(input("Enter 1 for filling table cve_items...\n"
                               "Enter 2 for filling table microsoft_application_server...\n"
                               "Enter 3 for filling table microsoft_database_server...\n"
                               "Enter 4 for filling table microsoft_mail_server...\n"
                               "Enter 5 for filling table openstack_compute_server...\n"
                               "Enter 6 for filling table openstack_controller_server...\n"
                               "Enter 7 for filling table ubuntu_application_server...\n"
                               "Enter 8 for filling table ubuntu_database_server...\n"
                               "Enter 9 for filling table ubuntu_mail_server...\n"
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
                connection = engine.connect()
                connection.execute(query)
            elif choice == 3:
                # Fill data to table microsoft_database_server
                print("Filling data in microsoft_database_server table...")
                query = "INSERT INTO microsoft_database_server(cve_id, published_datetime, score, " \
                        "vulnerable_software_list) " \
                        "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
                        "FROM cve_items " \
                        "where LOWER(vlist) LIKE '%mcafee%' " \
                        "or LOWER(vlist) LIKE '%microsoft%windows%' " \
                        "or LOWER(vlist) LIKE '%microsoft%active%directory%' " \
                        "or LOWER(vlist) LIKE '%microsoft%sql%server%' " \
                        "ORDER BY pdate;"
                connection = engine.connect()
                connection.execute(query)
            elif choice == 4:
                # Fill data to table microsoft_mail_server
                print("Filling data in microsoft_mail_server table...")
                query = "INSERT INTO microsoft_mail_server(cve_id, published_datetime, score, " \
                        "vulnerable_software_list) " \
                        "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
                        "FROM cve_items " \
                        "where LOWER(vlist) LIKE '%mcafee%' " \
                        "or LOWER(vlist) LIKE '%microsoft%windows%' " \
                        "or LOWER(vlist) LIKE '%microsoft%active%directory%' " \
                        "or LOWER(vlist) LIKE '%microsoft%exchange%' " \
                        "or LOWER(vlist) LIKE '%spamassassin%' " \
                        "ORDER BY pdate; "
                connection = engine.connect()
                connection.execute(query)
            elif choice == 5:
                # Fill data to table openstack_compute_server
                print("Filling data in openstack_compute_server table...")
                query = "INSERT INTO openstack_compute_server(cve_id, published_datetime, score, " \
                        "vulnerable_software_list) " \
                        "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
                        "FROM cve_items " \
                        "WHERE LOWER(vlist) LIKE '%linux_kernel:4%' " \
                        "or LOWER(vlist) LIKE '%linux_kernel:5%' " \
                        "or LOWER(vlist) LIKE '%iptables%' " \
                        "or LOWER(vlist) LIKE '%ubuntu%' " \
                        "or LOWER(vlist) LIKE '%ntp%' " \
                        "or LOWER(vlist) LIKE '%fail2ban%' " \
                        "or LOWER(vlist) LIKE '%openstack%nova%' " \
                        "or LOWER(vlist) LIKE '%openstack%neutron%' " \
                        "or LOWER(vlist) LIKE '%openstack%ceilometer%' " \
                        "ORDER BY pdate;"
                connection = engine.connect()
                connection.execute(query)
            elif choice == 6:
                # Fill data to table openstack_controller_server
                print("Filling data in openstack_controller_server table...")
                query = "INSERT INTO openstack_controller_server(cve_id, published_datetime, score, " \
                        "vulnerable_software_list) " \
                        "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
                        "FROM cve_items " \
                        "where LOWER(vlist) LIKE '%linux_kernel:4%' " \
                        "or LOWER(vlist) LIKE '%linux_kernel:5%' " \
                        "or LOWER(vlist) LIKE '%mysql:5.7%' " \
                        "or LOWER(vlist) LIKE '%mysql:8%' " \
                        "or LOWER(vlist) LIKE '%iptables%' " \
                        "or LOWER(vlist) LIKE '%apache2%' " \
                        "or LOWER(vlist) LIKE '%memcached%' " \
                        "or LOWER(vlist) LIKE '%mongodb%' " \
                        "or LOWER(vlist) LIKE '%ntp%' " \
                        "or LOWER(vlist) LIKE '%rabbitmq%' " \
                        "or LOWER(vlist) LIKE '%fail2ban%' " \
                        "or LOWER(vlist) LIKE '%ubuntu%' " \
                        "or LOWER(vlist) LIKE '%openstack%keystone%' " \
                        "or LOWER(vlist) LIKE '%openstack%glance%' " \
                        "or LOWER(vlist) LIKE '%openstack%nova%' " \
                        "or LOWER(vlist) LIKE '%openstack%neutron%' " \
                        "or LOWER(vlist) LIKE '%openstack%ceilometer%' " \
                        "or LOWER(vlist) LIKE '%openstack%horizon%' " \
                        "ORDER BY pdate;"
                connection = engine.connect()
                connection.execute(query)
            elif choice == 7:
                # Fill data to table ubuntu_application_server
                print("Filling data in ubuntu_application_server table...")
                query = "INSERT INTO ubuntu_application_server(cve_id, published_datetime, score, " \
                        "vulnerable_software_list) " \
                        "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
                        "FROM cve_items " \
                        "where LOWER(vlist) LIKE '%linux_kernel%' " \
                        "or LOWER(vlist) LIKE '%iptables%' " \
                        "or LOWER(vlist) LIKE '%apache2%' " \
                        "or LOWER(vlist) LIKE '%ntp%' " \
                        "or LOWER(vlist) LIKE '%fail2ban%' " \
                        "or LOWER(vlist) LIKE '%ubuntu%' " \
                        "or LOWER(vlist) LIKE '%jboss%' " \
                        "or LOWER(vlist) LIKE '%clamav%' " \
                        "or LOWER(vlist) LIKE '%ufw%' " \
                        "or LOWER(vlist) LIKE '%ldap%' " \
                        "ORDER BY pdate;"
                connection = engine.connect()
                connection.execute(query)
            elif choice == 8:
                # Fill data to table ubuntu_database_server
                print("Filling data in ubuntu_database_server table...")
                query = "INSERT INTO ubuntu_database_server(cve_id, published_datetime, score, " \
                        "vulnerable_software_list) " \
                        "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
                        "FROM cve_items " \
                        "where LOWER(vlist) LIKE '%linux_kernel%' " \
                        "or LOWER(vlist) LIKE '%iptables%' " \
                        "or LOWER(vlist) LIKE '%ntp%' " \
                        "or LOWER(vlist) LIKE '%fail2ban%' " \
                        "or LOWER(vlist) LIKE '%ubuntu%' " \
                        "or LOWER(vlist) LIKE '%clamav%' " \
                        "or LOWER(vlist) LIKE '%ufw%' " \
                        "or LOWER(vlist) LIKE '%mysql%' " \
                        "or LOWER(vlist) LIKE '%postgres%' " \
                        "or LOWER(vlist) LIKE '%mongodb%' " \
                        "or LOWER(vlist) LIKE '%ldap%' " \
                        "ORDER BY pdate;"
                connection = engine.connect()
                connection.execute(query)
            elif choice == 9:
                # Fill data to table ubuntu_mail_server
                print("Filling data in ubuntu_mail_server table...")
                query = "INSERT INTO ubuntu_mail_server(cve_id, published_datetime, score, vulnerable_software_list) " \
                        "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
                        "FROM cve_items " \
                        "where LOWER(vlist) LIKE '%linux_kernel%' " \
                        "or LOWER(vlist) LIKE '%iptables%' " \
                        "or LOWER(vlist) LIKE '%ntp%' " \
                        "or LOWER(vlist) LIKE '%fail2ban%' " \
                        "or LOWER(vlist) LIKE '%ubuntu%' " \
                        "or LOWER(vlist) LIKE '%clamav%' " \
                        "or LOWER(vlist) LIKE '%ufw%' " \
                        "or LOWER(vlist) LIKE '%ldap%' " \
                        "or LOWER(vlist) LIKE '%zimbra%' " \
                        "or LOWER(vlist) LIKE '%spamassassin%' " \
                        "ORDER BY pdate;"
                connection = engine.connect()
                connection.execute(query)
            elif choice == -1:
                print("Exiting current third step subroutine execution...\n\n")
                break
            else:
                print("You entered a wrong choice...\n\n")

    print("Data entry to main tables has finished...\n\n")


def fill_data_to_sub_tables():
    # TODO: Add check if tables are full or not.
    print("Filling data to sub tables...")
    engine, session = sqlalchemy_engine_start()

    # Fill data to table active_directory
    query = "INSERT INTO active_directory(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%microsoft%active%directory%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table apache
    query = "INSERT INTO apache(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%apache2%' " \
            "OR LOWER(vlist) LIKE '%apache%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table clamav
    query = "INSERT INTO clamav(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%clamav%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table dot_Net
    query = "INSERT INTO dot_Net(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%.net%framework%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table fail2ban
    query = "INSERT INTO fail2ban(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%fail2ban%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table iis
    query = "INSERT INTO iis(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%microsoft%iis%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table iptables
    query = "INSERT INTO iptables(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%iptables%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table java
    query = "INSERT INTO java(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%java%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table jboss
    query = "INSERT INTO jboss(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%jboss%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table ldap
    query = "INSERT INTO ldap(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%ldap%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table linux_kernel
    query = "INSERT INTO linux_kernel(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%linux_kernel%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table mcafee
    query = "INSERT INTO mcafee(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%mcafee%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table memcached
    query = "INSERT INTO memcached(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%memcached%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table microsoft_exchange
    query = "INSERT INTO microsoft_exchange(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%microsoft%exchange%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table microsoft_sql_server
    query = "INSERT INTO microsoft_sql_server(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%microsoft%sql%server%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table mongodb
    query = "INSERT INTO mongodb(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%mongodb%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table mysql
    query = "INSERT INTO mysql(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%mysql%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table ntp
    query = "INSERT INTO ntp(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%ntp%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table openstack_ceilometer
    query = "INSERT INTO openstack_ceilometer(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%openstack%ceilometer%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table openstack_glance
    query = "INSERT INTO openstack_glance(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%openstack%glance%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table openstack_horizon
    query = "INSERT INTO openstack_horizon(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%openstack%horizon%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table openstack_keystone
    query = "INSERT INTO openstack_keystone(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%openstack%keystone%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table openstack_neutron
    query = "INSERT INTO openstack_neutron(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%openstack%neutron%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table openstack_nova
    query = "INSERT INTO openstack_nova(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%openstack%nova%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table postgres
    query = "INSERT INTO postgres(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%postgres%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table rabbitmq
    query = "INSERT INTO rabbitmq(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%rabbitmq%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table spamassassin
    query = "INSERT INTO spamassassin(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%spamassassin%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table ubuntu_os
    query = "INSERT INTO ubuntu_os(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%ubuntu%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table ufw
    query = "INSERT INTO ufw(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%ufw%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table windows_os
    query = "INSERT INTO windows_os(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist " \
            "FROM cve_items " \
            "WHERE LOWER(vlist) LIKE '%microsoft%windows%' " \
            "ORDER BY pdate;"
    connection = engine.connect()
    connection.execute(query)
    # Fill data to table zimbra
    query = "INSERT INTO zimbra(cve_id, published_datetime, score, vulnerable_software_list) " \
            "SELECT cve_id, date(published_datetime) AS pdate, score, vulnerable_software_list AS vlist  " \
            "FROM cve_items  " \
            "WHERE LOWER(vlist) LIKE '%zimbra%'  " \
            "ORDER BY pdate"
    connection = engine.connect()
    connection.execute(query)

    print("Data entry to sub tables has finished...\n\n")


def create_final_tables():
    # TODO: Add check if tables exist and if they are full or not.
    print("Performing value reduction & create final tables...")
    engine, session = sqlalchemy_engine_start()

    # Create tables
    print("Creating final tables in sqlite...")
    print("Available tables before creation: ", engine.table_names())
    # microsoft_application_server_final
    query = "CREATE TABLE `microsoft_application_server_final` ( " \
            "cve_id TEXT NOT NULL, " \
            "published_datetime DATETIME, " \
            "score FLOAT, " \
            "vulnerable_software_list TEXT, " \
            "PRIMARY KEY (cve_id));"
    connection = engine.connect()
    connection.execute(query)
    # microsoft_database_server_final
    query = "CREATE TABLE `microsoft_database_server_final` ( " \
            "cve_id TEXT NOT NULL, " \
            "published_datetime DATETIME, " \
            "score FLOAT, " \
            "vulnerable_software_list TEXT, " \
            "PRIMARY KEY (cve_id));"
    connection = engine.connect()
    connection.execute(query)
    # microsoft_mail_server_final
    query = "CREATE TABLE `microsoft_mail_server_final` ( " \
            "cve_id TEXT NOT NULL, " \
            "published_datetime DATETIME, " \
            "score FLOAT, " \
            "vulnerable_software_list TEXT, " \
            "PRIMARY KEY (cve_id));"
    connection = engine.connect()
    connection.execute(query)
    # openstack_compute_server_final
    query = "CREATE TABLE `openstack_compute_server_final` ( " \
            "cve_id TEXT NOT NULL, " \
            "published_datetime DATETIME, " \
            "score FLOAT, " \
            "vulnerable_software_list TEXT, " \
            "PRIMARY KEY (cve_id));"
    connection = engine.connect()
    connection.execute(query)
    # openstack_controller_server_final
    query = "CREATE TABLE `openstack_controller_server_final` ( " \
            "cve_id TEXT NOT NULL, " \
            "published_datetime DATETIME, " \
            "score FLOAT, " \
            "vulnerable_software_list TEXT, " \
            "PRIMARY KEY (cve_id));"
    connection = engine.connect()
    connection.execute(query)
    # ubuntu_application_server_final
    query = "CREATE TABLE `ubuntu_application_server_final` ( " \
            "cve_id TEXT NOT NULL, " \
            "published_datetime DATETIME, " \
            "score FLOAT, " \
            "vulnerable_software_list TEXT, " \
            "PRIMARY KEY (cve_id));"
    connection = engine.connect()
    connection.execute(query)
    # ubuntu_database_server_final
    query = "CREATE TABLE `ubuntu_database_server_final` ( " \
            "cve_id TEXT NOT NULL, " \
            "published_datetime DATETIME, " \
            "score FLOAT, " \
            "vulnerable_software_list TEXT, " \
            "PRIMARY KEY (cve_id));"
    connection = engine.connect()
    connection.execute(query)
    # ubuntu_mail_server_final
    query = "CREATE TABLE `ubuntu_mail_server_final` ( " \
            "cve_id TEXT NOT NULL,  " \
            "published_datetime DATETIME, " \
            "score FLOAT,  " \
            "vulnerable_software_list TEXT, " \
            "PRIMARY KEY (cve_id));"
    connection = engine.connect()
    connection.execute(query)
    # Check things upon finishing
    print("Available tables after creation: ", engine.table_names())
    print("Final tables creation has finished...\n")

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
    connection = engine.connect()
    connection.execute(query)
    # microsoft_database_server_final
    query = "INSERT INTO microsoft_database_server_final(cve_id, published_datetime, score, " \
            "vulnerable_software_list) " \
            "SELECT tt.* " \
            "FROM microsoft_database_server AS tt " \
            "INNER JOIN " \
            "(SELECT published_datetime, MAX(score) as MaxScore " \
            "FROM microsoft_database_server " \
            "GROUP BY published_datetime) AS groupedtt " \
            "ON  tt.score = groupedtt.MaxScore " \
            "AND tt.published_datetime = groupedtt.published_datetime;"
    connection = engine.connect()
    connection.execute(query)
    # microsoft_mail_server_final
    query = "INSERT INTO microsoft_mail_server_final(cve_id, published_datetime, score, " \
            "vulnerable_software_list) " \
            "SELECT tt.* " \
            "FROM microsoft_mail_server AS tt " \
            "INNER JOIN " \
            "(SELECT published_datetime, MAX(score) as MaxScore " \
            "FROM microsoft_mail_server " \
            "GROUP BY published_datetime) AS groupedtt " \
            "ON  tt.score = groupedtt.MaxScore " \
            "AND tt.published_datetime = groupedtt.published_datetime;"
    connection.execute(query)
    # openstack_compute_server_final
    query = "INSERT INTO openstack_compute_server_final(cve_id, published_datetime, score, " \
            "vulnerable_software_list) " \
            "SELECT tt.* " \
            "FROM openstack_compute_server AS tt " \
            "INNER JOIN " \
            "(SELECT published_datetime, MAX(score) as MaxScore " \
            "FROM openstack_compute_server " \
            "GROUP BY published_datetime) AS groupedtt " \
            "ON  tt.score = groupedtt.MaxScore " \
            "AND tt.published_datetime = groupedtt.published_datetime;"
    connection = engine.connect()
    connection.execute(query)
    # openstack_controller_server_final
    query = "INSERT INTO openstack_controller_server_final(cve_id, published_datetime, score, " \
            "vulnerable_software_list) " \
            "SELECT tt.* " \
            "FROM openstack_controller_server AS tt " \
            "INNER JOIN " \
            "(SELECT published_datetime, MAX(score) as MaxScore " \
            "FROM openstack_controller_server " \
            "GROUP BY published_datetime) AS groupedtt " \
            "ON  tt.score = groupedtt.MaxScore " \
            "AND tt.published_datetime = groupedtt.published_datetime;"
    connection = engine.connect()
    connection.execute(query)
    # ubuntu_application_server_final
    query = "INSERT INTO ubuntu_application_server_final(cve_id, published_datetime, score, " \
            "vulnerable_software_list) " \
            "SELECT tt.* " \
            "FROM ubuntu_application_server AS tt " \
            "INNER JOIN " \
            "(SELECT published_datetime, MAX(score) as MaxScore " \
            "FROM ubuntu_application_server " \
            "GROUP BY published_datetime) AS groupedtt " \
            "ON  tt.score = groupedtt.MaxScore " \
            "AND tt.published_datetime = groupedtt.published_datetime;"
    connection = engine.connect()
    connection.execute(query)
    # ubuntu_database_server_final
    query = "INSERT INTO ubuntu_database_server_final(cve_id, published_datetime, score, " \
            "vulnerable_software_list) " \
            "SELECT tt.* " \
            "FROM ubuntu_database_server AS tt " \
            "INNER JOIN " \
            "(SELECT published_datetime, MAX(score) as MaxScore " \
            "FROM ubuntu_database_server " \
            "GROUP BY published_datetime) AS groupedtt " \
            "ON  tt.score = groupedtt.MaxScore " \
            "AND tt.published_datetime = groupedtt.published_datetime;"
    connection = engine.connect()
    connection.execute(query)
    # ubuntu_mail_server_final
    query = "INSERT INTO ubuntu_mail_server_final(cve_id, published_datetime, score, " \
            "vulnerable_software_list) " \
            "SELECT tt.* " \
            "FROM ubuntu_mail_server AS tt " \
            "INNER JOIN " \
            "(SELECT published_datetime, MAX(score) as MaxScore " \
            "FROM ubuntu_mail_server " \
            "GROUP BY published_datetime) AS groupedtt " \
            "ON  tt.score = groupedtt.MaxScore " \
            "AND tt.published_datetime = groupedtt.published_datetime;"
    connection = engine.connect()
    connection.execute(query)
    print("Value reduction in final tables has finished...\n")

    print("Value reduction & data entry to final tables has finished...\n\n")


def calculate_missing_values_dataframe(ts_wmv, years_list, table_name, time_granularity_val):
    missing_values_df = DataFrame(index=range(len(years_list)), columns=['System_Examined',
                                                                         'Start_Year_Range',
                                                                         'Stop_Year_Range',
                                                                         'Total_num_of_Values',
                                                                         'Non_Zero_values',
                                                                         'Percentage_of_missing_values',
                                                                         'time_granularity'])
    missing_values_df['System_Examined'] = table_name
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

    # Display full dataframe
    print("missing_values dataframe presentation: ")
    print()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None,
                           'display.max_colwidth', -1):
        print(missing_values_df)
    print("===============================================")
    print()

    return missing_values_df


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
        warnings.warn("signal preferred to be even in size, autoFixing it...")
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


def fill_missing_values(ts):
    print('Missing: %d' % sum(isnan(ts.values).flatten()))
    # define imputer
    imputer = KNNImputer()
    temp = ts.values.reshape(-1, 1)
    # fit on the dataset
    imputer.fit(temp)
    # transform the dataset and cast table from 2D to 1D
    Xtrans = imputer.transform(temp).flatten()
    ts.iloc[:] = Xtrans
    # summarize total missing
    print('Missing: %d' % sum(isnan(ts.values).flatten()))

    return ts


# noinspection DuplicatedCode
def create_input_dataset_for_forecast_methods(table_name):
    print()
    print("Create input dataset for " + table_name + "...")

    # Allocate initial values in flags
    eligible_range_for_week = False

    # Create a SQL connection to SQLite database that holds final tables
    con = sqlite3.connect("/home/lumi/Dropbox/unipi/paper_NVD_forcasting/sqlight_db/nvd_nist.db")
    # Read sqlite query results into a pandas DataFrame
    query = "SELECT published_datetime, score from " + table_name
    initial_df = pd.read_sql_query(query, con)
    # Close connection when done
    con.close()

    initial_df.published_datetime = pd.to_datetime(initial_df.published_datetime)
    # Check if duplicate values exist in published_datetime column and keep the maximum value
    if initial_df.published_datetime.duplicated().any():
        # Remove duplicate values for the same date
        initial_df = initial_df.groupby('published_datetime').max()
        initial_df = initial_df.reset_index()

    # ==================================================================================================================
    # First check for day
    # ==================================================================================================================
    # create series from dataframe
    ts = pd.Series(initial_df['score'].values, index=initial_df['published_datetime'])
    ts.index = pd.DatetimeIndex(ts.index)
    # Create time series with all missing date values and fill them with zeros
    idx = pd.date_range(initial_df['published_datetime'].min(), initial_df['published_datetime'].max())
    # Create day time interval
    ts_wmv_day = ts.reindex(idx, fill_value=0)

    # Verify that result of SQL query is stored in the dataframe
    print("===============================================")
    print("ts day dataframe description: ")
    print(ts_wmv_day.describe())
    print("===============================================")
    print()

    # Create missing values dataframe
    years_list = ts_wmv_day.index.strftime("%Y").drop_duplicates().tolist()
    missing_values = calculate_missing_values_dataframe(ts_wmv_day, years_list, table_name, "day")
    # Check for full time range
    eligible_range_for_day, missing_values_percentage_threshold, start_year, end_year = \
        check_missing_value_percentage(ts, missing_values)
    if eligible_range_for_day:
        if missing_values_percentage_threshold == 20:
            print("Missing values percentage threshold used is 20%...")
        elif missing_values_percentage_threshold == 10:
            print("Missing values percentage threshold used is 10%...")
    # Check for reducing (rolling window) year ranges
    # i = 0
    # while not eligible_range_for_day and i < (len(years_list) - 1):
    #     ts = ts.drop(ts.index[ts.index.year.isin([int(years_list[i])])])
    #     eligible_range_for_day, missing_values_percentage_threshold, start_year, end_year = \
    #         check_missing_value_percentage(ts, missing_values)
    #     if eligible_range_for_day:
    #         if missing_values_percentage_threshold == 20:
    #             print("Missing values percentage threshold used is 20%...")
    #         elif missing_values_percentage_threshold == 10:
    #             print("Missing values percentage threshold used is 10%...")
    #         break
    #     i += 1

    if eligible_range_for_day:
        # TODO: Implement part to impute missing values & save to csv method
        print("Day time_granularity was found eligible...")
        for year in years_list:
            if year < start_year:
                ts_wmv_day = ts_wmv_day.drop(ts_wmv_day.index[ts_wmv_day.index.year.isin([int(year)])])
            elif year >= start_year:
                break
        # Series within the selected time range, with 0 replaced by NaN
        ts_wmv_day = ts_wmv_day.replace({0: nan})
        ts_wmv_day = fill_missing_values(ts_wmv_day)
        print("Saving series to csv to be used by forecast methods...")
        ts_wmv_day.to_csv(r'data/datasets/' + table_name + '_day.csv', header=None)
        print("Input dataset creation for " + table_name + " has finished successfully...\n\n")

        return True

    # ==================================================================================================================
    # Second check for week
    # ==================================================================================================================
    if not eligible_range_for_day:
        print("Day time_granularity was not eligible. Switching to week...")
        # create series from dataframe
        ts = pd.Series(initial_df['score'].values, index=initial_df['published_datetime'])
        ts.index = pd.DatetimeIndex(ts.index)
        # Create all missing date values and fill them with zero
        idx = pd.date_range(initial_df['published_datetime'].min(), initial_df['published_datetime'].max())
        ts_wmv_week = ts.reindex(idx, fill_value=0)
        # Create week time interval
        ts_wmv_week = ts_wmv_week.resample('W-MON', label='left', closed='left').max()

        # Verify that result of SQL query is stored in the dataframe
        print("===============================================")
        print("ts week dataframe description: ")
        print(ts_wmv_week.describe())
        print("===============================================")
        print()

        # Create missing values dataframe
        years_list = ts_wmv_week.index.strftime("%Y").drop_duplicates().tolist()
        missing_values = calculate_missing_values_dataframe(ts_wmv_week, years_list, table_name, "week")
        # Check for full time range
        eligible_range_for_week, missing_values_percentage_threshold, start_year, end_year = \
            check_missing_value_percentage(ts, missing_values)
        if eligible_range_for_week:
            if missing_values_percentage_threshold == 20:
                print("Missing values percentage threshold used is 20%...")
            elif missing_values_percentage_threshold == 10:
                print("Missing values percentage threshold used is 10%...")
        # Check for reducing (rolling window) year ranges
        # i = 0
        # while not eligible_range_for_week and i < (len(years_list) - 1):
        #     ts = ts.drop(ts.index[ts.index.year.isin([int(years_list[i])])])
        #     eligible_range_for_week, missing_values_percentage_threshold, start_year, end_year = \
        #         check_missing_value_percentage(ts, missing_values)
        #     if eligible_range_for_week:
        #         if missing_values_percentage_threshold == 20:
        #             print("Missing values percentage threshold used is 20%...")
        #         elif missing_values_percentage_threshold == 10:
        #             print("Missing values percentage threshold used is 10%...")
        #         break
        #     i += 1

        if eligible_range_for_week:
            print("Week time_granularity was found eligible...")
            print("Creating time series with imputed missing values...")
            for year in years_list:
                if year < start_year:
                    ts_wmv_week = ts_wmv_week.drop(ts_wmv_week.index[ts_wmv_week.index.year.isin([int(year)])])
                elif year >= start_year:
                    break
            # Series within the selected time range, with 0 replaced by NaN
            ts_wmv_week = ts_wmv_week.replace({0: nan})
            ts_wmv_week = fill_missing_values(ts_wmv_week)
            print("Saving series to csv to be used by forecast methods...")
            ts_wmv_week.to_csv(r'data/datasets/' + table_name + '_week.csv', header=None)
            print("Input dataset creation for " + table_name + " has finished successfully...\n\n")

            return True

    # ==================================================================================================================
    # Third check for month
    # ==================================================================================================================
    if eligible_range_for_week:
        print("Week time_granularity was not eligible. Switching to month...")
        # create series from dataframe
        ts = pd.Series(initial_df['score'].values, index=initial_df['published_datetime'])
        ts.index = pd.DatetimeIndex(ts.index)
        # Create all missing date values and fill them with zero
        idx = pd.date_range(initial_df['published_datetime'].min(), initial_df['published_datetime'].max())
        ts_wmv_month = ts.reindex(idx, fill_value=0)
        # Create week time interval
        ts_wmv_month = ts_wmv_month.resample('M-MON', label='left', closed='left').max()

        # Verify that result of SQL query is stored in the dataframe
        print("===============================================")
        print("ts month dataframe description: ")
        print(ts_wmv_month.describe())
        print("===============================================")
        print()

        # Create missing values dataframe
        years_list = ts_wmv_month.index.strftime("%Y").drop_duplicates().tolist()
        missing_values = calculate_missing_values_dataframe(ts_wmv_month, years_list, table_name, "month")
        # Check for full time range
        eligible_range_for_month, missing_values_percentage_threshold, start_year, end_year = \
            check_missing_value_percentage(ts, missing_values)
        if eligible_range_for_month:
            if missing_values_percentage_threshold == 20:
                print("Missing values percentage threshold used is 20%...")
            elif missing_values_percentage_threshold == 10:
                print("Missing values percentage threshold used is 10%...")

        if eligible_range_for_month:
            print("Month time_granularity was found eligible...")
            print("Creating time series with imputed missing values...")
            for year in years_list:
                if year < start_year:
                    ts_wmv_month = ts_wmv_month.drop(ts_wmv_month.index[ts_wmv_month.index.year.isin([int(year)])])
                elif year >= start_year:
                    break
            # Series within the selected time range, with 0 replaced by NaN
            ts_wmv_month = ts_wmv_month.replace({0: nan})
            ts_wmv_month = fill_missing_values(ts_wmv_month)
            print("Saving series to csv to be used by forecast methods...")
            ts_wmv_month.to_csv(r'data/datasets/' + table_name + '_month.csv', header=None)
            print("Input dataset creation for " + table_name + " has finished successfully...\n\n")

            return True

    # ==================================================================================================================

    print("Input dataset creation for " + table_name + " could not produce/find any suitable dataset...\n\n")

    return False


def choose_forecast_method(csv_file_name):
    # ======================================================================================================
    # Common steps
    # ======================================================================================================
    print("Performing initial common steps for " + csv_file_name + " dataset...")
    print("==========================================================")
    print("First step split initial dataset to training & validation ones...")
    common_steps.split_dataset(csv_file_name)
    input("Press Enter to continue...")

    print("Second step create persistence model to establish a baseline of performance...")
    common_steps.create_persistence_model_from_dataset(csv_file_name)
    input("Press Enter to continue...")

    print("Third step data analysis through summary statistics and plots...")
    common_steps.summary_statistics(csv_file_name)
    input("Press Enter to continue...")
    common_steps.create_line_plots(csv_file_name)
    input("Press Enter to continue...")
    common_steps.create_density_plots(csv_file_name)
    input("Press Enter to continue...")
    # common_steps.create_boxplots_plots(csv_file_name)
    # input("Press Enter to continue...")

    print("Fourth step statistical test for the stationarity of the time series...")
    common_steps.check_stationarity(csv_file_name)
    input("Press Enter to continue...")

    print("Fifth step ACF/PACF plots to determine ARIMA's (p, q)...")
    common_steps.acf_pacf_plots(csv_file_name)
    input("Press Enter to continue...")
    print("==========================================================")
    # ==================================================================================================================
    print()
    print("Choose forecast method to proceed...")
    print()

    while True:
        try:
            choice = int(input("Enter 1 for linear model...\n"
                               "Enter 2 for autoregression model...\n"
                               "Enter 3 for moving average model...\n"
                               "Enter 4 for ARMA model...\n"
                               "Enter 5 for ARIMA model...\n"
                               "Enter 6 for polynomial regression model...\n"
                               "Enter 7 for Multilayer perceptron model...\n"
                               "Enter 8 for Convolutional Neural Network model...\n"
                               "Enter 9 for Long Short-Term Memory model...\n"
                               "Enter 10 for Convolutional Neural Network - Long Short-Term Memory hybrid model...\n"
                               "Enter 11 for Convolutional Long Short-Term Memory hybrid model...\n"
                               "Enter -1 to exit third step subroutine execution...\n"))
        except ValueError:
            print("You entered a wrong choice...\n\n")
        else:
            # ======================================================================================================
            # Initial steps
            # ======================================================================================================
            if choice == 1:  # Choose linear model
                print("Linear model steps...")
            elif choice == 2:  # Choose autoregression model
                print("Autoregression model steps...")
            elif choice == 3:  # Choose Moving average model
                print("Moving average model steps...")
            elif choice == 4:  # Choose ARMA model
                print("ARMA model steps...")
                # ======================================================================================================
                # Modeling steps
                # ======================================================================================================
                print("Sixth step run manual ARIMA for specific (p, q, d) values...")
                arma_modeling_steps.manual_arma(csv_file_name)
                input("Press Enter to continue...")

                print("Eighth step plot residual errors plots...")
                arma_modeling_steps.residual_errors_plot_arma(csv_file_name)
                arma_modeling_steps.residual_acf_errors_plot_arma(csv_file_name)
                input("Press Enter to continue...")
                # ======================================================================================================
                # Validation steps
                # ======================================================================================================
                print("Ninth step save fitted model...")
                arma_validation_steps.save_fitted_model_arma(csv_file_name)
                input("Press Enter to continue...")

                print("Tenth step validate fitted model...")
                arma_validation_steps.validate_arma_model(csv_file_name)
                input("Press Enter to continue...")
            elif choice == 5:  # Choose ARIMA model
                print("ARIMA model steps...")
                # ======================================================================================================
                # Modeling steps
                # ======================================================================================================
                print("Sixth step run manual ARIMA for specific (p, q, d) values...")
                arima_modeling_steps.manual_arima(csv_file_name)
                input("Press Enter to continue...")

                # print("Seventh step run grid search ARIMA to determine best (p, q, d) among an array of values...")
                # arima_modeling_steps.grid_search_arima(csv_file_name)
                # input("Press Enter to continue...")

                print("Eighth step plot residual errors plots...")
                arima_modeling_steps.residual_errors_plot_arima(csv_file_name)
                arima_modeling_steps.residual_acf_errors_plot_arima(csv_file_name)
                input("Press Enter to continue...")
                # ======================================================================================================
                # Validation steps
                # ======================================================================================================
                print("Ninth step save fitted model...")
                arima_validation_steps.save_fitted_model_arima(csv_file_name)
                input("Press Enter to continue...")

                print("Tenth step validate fitted model...")
                arima_validation_steps.validate_arima_model(csv_file_name)
                input("Press Enter to continue...")
            elif choice == 6:  # Choose polynomial regression model
                print("Polynomial regression model steps...")
            elif choice == 7:  # Choose MLP model
                print("Multilayer perceptron model steps...")
                # ======================================================================================================
                # Modeling & validation steps
                # ======================================================================================================
                print("Sixth step run MLP & validate fitted model...")
                mlp_modeling.repeat_evaluate(csv_file_name)
                input("Press Enter to continue...")
            elif choice == 8:  # Choose CNN model
                print("Convolutional Neural Network model steps...")
                # ======================================================================================================
                # Modeling & validation steps
                # ======================================================================================================
                print("Sixth step run CNN & validate fitted model...")
                cnn_modeling.repeat_evaluate(csv_file_name)
                input("Press Enter to continue...")
            elif choice == 9:  # Choose LSTM model
                print("Long Short-Term Memory model steps...")
                # ======================================================================================================
                # Modeling & validation steps
                # ======================================================================================================
                print("Sixth step run LSTM & validate fitted model...")
                cnn_lstm_modeling.repeat_evaluate()
                input("Press Enter to continue...")
            elif choice == 10:  # Choose CNN-LSTM model
                print("Convolutional Neural Network - Long Short-Term Memory hybrid model steps...")
                # ======================================================================================================
                # Modeling & validation steps
                # ======================================================================================================
                print("Sixth step run CNN-LSTM hybrid & validate fitted model...")

                input("Press Enter to continue...")
            elif choice == 11:  # Choose ConvLSTM model
                print("Convolutional Long Short-Term Memory hybrid model steps...")
                # ======================================================================================================
                # Modeling & validation steps
                # ======================================================================================================
                print("Sixth step run ConvLSTM hybrid & validate fitted model...")

                input("Press Enter to continue...")
            elif choice == -1:
                print("Exiting current fifth step subroutine execution...\n\n")
                break
            else:
                print("You entered a wrong choice...\n\n")


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
                               "Extract CVE_Items[] from json files, prepare dataset & save it to sqlite...\n"
                               "Enter 4 to run fourth step ---> "
                               "Create the input datasets to use with the forecast methods and save them as csv...\n"
                               "Enter 5 to run fifth step ---> Execute forecasting algorithms...\n"
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

                print("First step has finished...\n\n")
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

                print("Second step has finished...\n\n")
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
                                            "Enter 2 to fill data to main tables, perform values reduction & create "
                                            "final tables...\n"
                                            "Enter 3 to fill data to sub tables...\n"
                                            "Enter 4 To create dataset input for forecast methods...\n"
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
                            # Fill data to main tables
                            fill_data_to_main_tables(cve_items_dict)
                            # Fill data to final tables
                            create_final_tables()
                        elif _choice == 3:
                            # TODO: Add check so that this routine is executed only if cve_items table has been
                            #  created/loaded Then fill data to the rest of the sub tables
                            # Fill data to sub tables
                            fill_data_to_sub_tables()
                        elif _choice == -1:
                            print("Exiting current third step subroutine execution...\n\n")
                            break
                        else:
                            print("You entered a wrong choice...\n\n")

                print("Third step has finished...\n\n")
            elif choice == 4:
                print("Started fourth step! Create the input datasets to use with the forecast methods...")
                print("There is total of 8 systems to choose from...")
                while True:
                    try:
                        _choice = int(input("Enter 1 to create dataset for microsoft_application_server_final...\n"
                                            "Enter 2 to create dataset for microsoft_database_server_final...\n"
                                            "Enter 3 to create dataset for microsoft_mail_server_final...\n"
                                            "Enter 4 to create dataset for openstack_compute_server_final...\n"
                                            "Enter 5 to create dataset for openstack_controller_server_final...\n"
                                            "Enter 6 to create dataset for ubuntu_application_server_final...\n"
                                            "Enter 7 to create dataset for ubuntu_database_server_final...\n"
                                            "Enter 8 to create dataset for ubuntu_mail_server_final...\n"
                                            "Enter -1 to exit third step subroutine execution...\n"))
                    except ValueError:
                        print("You entered a wrong choice...\n\n")
                    else:
                        if _choice == 1:
                            # Create the input dataset for microsoft_application_server_final
                            create_input_dataset_for_forecast_methods("microsoft_application_server_final")
                        elif _choice == 2:
                            # Create the input dataset for microsoft_database_server_final
                            create_input_dataset_for_forecast_methods("microsoft_database_server_final")
                        elif _choice == 3:
                            # Create the input dataset for microsoft_mail_server_final
                            create_input_dataset_for_forecast_methods("microsoft_mail_server_final")
                        elif _choice == 4:
                            # Create the input dataset for openstack_compute_server_final
                            create_input_dataset_for_forecast_methods("openstack_compute_server_final")
                        elif _choice == 5:
                            # Create the input dataset for openstack_controller_server_final
                            create_input_dataset_for_forecast_methods("openstack_controller_server_final")
                        elif _choice == 6:
                            # Create the input dataset for ubuntu_application_server_final
                            create_input_dataset_for_forecast_methods("ubuntu_application_server_final")
                        elif _choice == 7:
                            # Create the input dataset for ubuntu_database_server_final
                            create_input_dataset_for_forecast_methods("ubuntu_database_server_final")
                        elif _choice == 8:
                            # Create the input dataset for ubuntu_mail_server_final
                            create_input_dataset_for_forecast_methods("ubuntu_mail_server_final")
                        elif _choice == -1:
                            print("Exiting current fourth step subroutine execution...\n\n")
                            break
                        else:
                            print("You entered a wrong choice...\n\n")

                # TODO: Show successful and not successful datasets
                print("Fourth step has finished...\n\n")
            elif choice == 5:
                print("Started fifth step! Execute forecasting algorithms...")
                print("There is total of 8 systems to choose from...")
                while True:
                    try:
                        _choice = int(input("Enter 1 to use microsoft_application_server_final dataset...\n"
                                            "Enter 2 to use microsoft_database_server_final dataset...\n"
                                            "Enter 3 to use microsoft_mail_server_final dataset...\n"
                                            "Enter 4 to use openstack_compute_server_final dataset...\n"
                                            "Enter 5 to use openstack_controller_server_final dataset...\n"
                                            "Enter 6 to use ubuntu_application_server_final dataset...\n"
                                            "Enter 7 to use ubuntu_database_server_final dataset...\n"
                                            "Enter 8 to use ubuntu_mail_server_final dataset...\n"
                                            "Enter -1 to exit third step subroutine execution...\n"))
                    except ValueError:
                        print("You entered a wrong choice...\n\n")
                    # TODO: Add check to differentiate for day, week & month
                    else:
                        if _choice == 1:
                            # Use microsoft_application_server_final dataset
                            choose_forecast_method("microsoft_application_server_final_week.csv")
                        elif _choice == 2:
                            # Use microsoft_database_server_final dataset
                            choose_forecast_method("microsoft_database_server_final_week.csv")
                        elif _choice == 3:
                            # Use microsoft_mail_server_final dataset
                            choose_forecast_method("microsoft_mail_server_final_week.csv")
                        elif _choice == 4:
                            # Use openstack_compute_server_final dataset
                            choose_forecast_method("openstack_compute_server_final_week.csv")
                        elif _choice == 5:
                            # Use openstack_controller_server_final dataset
                            choose_forecast_method("openstack_controller_server_final_week.csv")
                        elif _choice == 6:
                            # Use ubuntu_application_server_final dataset
                            choose_forecast_method("ubuntu_application_server_final_week.csv")
                        elif _choice == 7:
                            # Use ubuntu_database_server_final dataset
                            choose_forecast_method("ubuntu_database_server_final_week.csv")
                        elif _choice == 8:
                            # Use ubuntu_mail_server_final dataset
                            choose_forecast_method("ubuntu_mail_server_final_week.csv")
                        elif _choice == -1:
                            print("Exiting current fourth step subroutine execution...\n\n")
                            break
                        else:
                            print("You entered a wrong choice...\n\n")

                print("Fifth step has finished...\n\n")
            elif choice == -1:
                print("Thank you for using Calchas! Now exiting program...\n\n")
                break
            else:
                print("You entered a wrong choice...\n\n")


if __name__ == "__main__":
    main()
