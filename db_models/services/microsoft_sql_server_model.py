from sqlalchemy import *
from db_models.base import Base


class MicrosoftSqlServer(Base):
    __tablename__ = 'microsoft_sql_server'

    cve_id = Column(Text, primary_key=True, autoincrement=False)
    published_datetime = Column(Numeric)
    score = Column(Float)
    vulnerable_software_list = Column(Text)

    def __repr__(self):
        return f'MicrosoftSqlServer {self.__name__}'
