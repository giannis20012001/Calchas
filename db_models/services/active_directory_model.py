from db_models.base import Base
from sqlalchemy import Column, Float, Text, DateTime


class ActiveDirectory(Base):
    __tablename__ = 'active_directory'

    cve_id = Column(Text, primary_key=True, autoincrement=False)
    published_datetime = Column(DateTime)
    score = Column(Float)
    vulnerable_software_list = Column(Text)

    def __repr__(self):
        return f'ActiveDirectory {self.__name__}'
