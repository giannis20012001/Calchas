from sqlalchemy import *
from db_models.base import Base


class WindowsOs(Base):
    __tablename__ = 'windows_os'

    cve_id = Column(Text, primary_key=True, autoincrement=False)
    published_datetime = Column(Numeric)
    score = Column(Float)
    vulnerable_software_list = Column(Text)

    def __repr__(self):
        return f'WindowsOs {self.__name__}'
