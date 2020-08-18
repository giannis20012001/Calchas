from sqlalchemy.ext import declarative


class BaseModel(object):
    __abstract__ = True

    def __repr__(self):
        fmt = u'{}.{}({})'
        package = self.__class__.__module__
        class_ = self.__class__.__name__
        attrs = sorted((c.name, getattr(self, c.name)) for c in self.__table__.columns)
        sattrs = u', '.join('{}={!r}'.format(*x) for x in attrs)
        return fmt.format(package, class_, sattrs)


# Global Base var for sqlalchemy
Base = declarative.declarative_base(cls=BaseModel)
