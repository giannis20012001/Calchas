from db_models.base import Base
from sqlalchemy import Column, Float, String, Text, DateTime


class CveItem(Base):
    __tablename__ = 'cve_items'

    cve_id = Column(Text, primary_key=True, autoincrement=False)
    # published_datetime = Column(Numeric)
    published_datetime = Column(DateTime)
    score = Column(Float)
    access_vector = Column(String)
    access_complexity = Column(String)
    authentication = Column(String)
    availability_impact = Column(String)
    confidentiality_impact = Column(String)
    integrity_impact = Column(String)
    last_modified_datetime = Column(String)
    urls = Column(String)
    summary = Column(String)
    vulnerable_software_list = Column(Text)
