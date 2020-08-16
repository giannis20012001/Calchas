from sqlalchemy import *
from db_models.base import Base


class Fail2Ban(Base):
    __tablename__ = 'fail2ban'

    cve_id = Column(Text, primary_key=True, autoincrement=False)
    published_datetime = Column(Numeric)
    score = Column(Float)
    vulnerable_software_list = Column(Text)

    def __repr__(self):
        return f'Fail2Ban {self.__name__}'
