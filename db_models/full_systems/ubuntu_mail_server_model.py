from db_models.base import Base
from sqlalchemy import Column, Float, Text, DateTime


class UbuntuMailServer(Base):
    __tablename__ = 'ubuntu_mail_server'

    cve_id = Column(Text, primary_key=True, autoincrement=False)
    published_datetime = Column(DateTime)
    score = Column(Float)
    vulnerable_software_list = Column(Text)

    def __repr__(self):
        return f'UbuntuMailServer {self.__name__}'
