from db_models.base import Base
from sqlalchemy import Column, Float, Text, DateTime


class Java(Base):
    __tablename__ = 'java'

    cve_id = Column(Text, primary_key=True, autoincrement=False)
    published_datetime = Column(DateTime)
    score = Column(Float)
    vulnerable_software_list = Column(Text)

    def __repr__(self):
        return f'Java {self.__name__}'
