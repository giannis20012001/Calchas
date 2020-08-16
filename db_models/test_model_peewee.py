import peewee
import datetime

db = peewee.SqliteDatabase('db_models.db')


class Cve(peewee.Model):
    text = peewee.CharField()
    created = peewee.DateField(default=datetime.date.today)

    class Meta:
        database = db
        db_table = 'notes'
