from sqlalchemy import *
from db_models.base import Base


class Jboss(Base):
    __tablename__ = 'jboss'

    cve_id = Column(Text, primary_key=True, autoincrement=False)
    published_datetime = Column(Numeric)
    score = Column(Float)
    vulnerable_software_list = Column(Text)

    def __repr__(self):
        return f'Jboss {self.__name__}'
