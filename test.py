from sqlalchemy.sql import text
from sqlalchemy import create_engine

engine = create_engine('sqlite:////home/lumi/Dropbox/unipi/paper_NVD_forcasting/sqlight_db/nvd_nist.db', echo=True)
with engine.connect() as con:
    rs = con.execute('SELECT * FROM book')

    for row in rs:
        print(row)
