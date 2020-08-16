from sqlalchemy import *
from db_models.base import Base


class Java(Base):
    __tablename__ = 'java'

    cve_id = Column(Text, primary_key=True, autoincrement=False)
    published_datetime = Column(Numeric)
    score = Column(Float)
    vulnerable_software_list = Column(Text)

    def __repr__(self):
        return f'Java {self.__name__}'
