from db_models.base import Base
from sqlalchemy import Column, Float, Text, DateTime


class SpamAssassin(Base):
    __tablename__ = 'spamassassin'

    cve_id = Column(Text, primary_key=True, autoincrement=False)
    published_datetime = Column(DateTime)
    score = Column(Float)
    vulnerable_software_list = Column(Text)

    def __repr__(self):
        return f'SpamAssassin {self.__name__}'
