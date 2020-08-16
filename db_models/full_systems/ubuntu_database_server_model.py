from sqlalchemy import *
from db_models.base import Base


class UbuntuDatabaseServer(Base):
    __tablename__ = 'ubuntu_database_server'

    cve_id = Column(Text, primary_key=True, autoincrement=False)
    published_datetime = Column(Numeric)
    score = Column(Float)
    vulnerable_software_list = Column(Text)

    def __repr__(self):
        return f'UbuntuDatabaseServer {self.__name__}'
