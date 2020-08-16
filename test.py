import requests
import re
import zipfile
import io
import json
import csv

with open("nvdcve-1.0-2017.json") as f_json:
    r = requests.get('https://nvd.nist.gov/vuln/data-feeds#JSON_FEED')

with open('output.csv', 'w', newline='') as f_output:
    csv_output = csv.writer(f_output)
    csv_output.writerow(['ID', 'VendorName', 'Description', 'VersionValues'])

    for filename in re.findall("nvdcve-1.0-[0-9]*\.json\.zip", r.text):
        print("Downloading {}".format(filename))
        r_zip_file = requests.get("https://static.nvd.nist.gov/feeds/json/cve/1.0/" + filename, stream=True)
        zip_file_bytes = io.BytesIO()

        for chunk in r_zip_file:
            zip_file_bytes.write(chunk)

        zip_file = zipfile.ZipFile(zip_file_bytes)

        for json_filename in zip_file.namelist():
            print("Extracting {}".format(json_filename))
            json_raw = zip_file.read(json_filename).decode('utf-8')
            json_data = json.loads(json_raw)

            for entry in json_data['CVE_Items']:
                try:
                    vendor_name = entry['cve']['affects']['vendor']['vendor_data'][0]['vendor_name']
                except IndexError:
                    vendor_name = "unknown"

                try:
                    url = entry['cve']['references']['reference_data'][0]['url']
                except IndexError:
                    url = ''

                try:
                    vv = []

                    for pd in entry['cve']['affects']['vendor']['vendor_data'][0]['product']['product_data']:
                        for vd in pd['version']['version_data']:
                            vv.append(vd['version_value'])

                    version_values = '/'.join(vv)
                except IndexError:
                    version_values = ''

                csv_output.writerow([
                    entry['cve']['CVE_data_meta']['ID'],
                    url,
                    vendor_name,
                    entry['cve']['description']['description_data'][0]['value'],
                    version_values])
