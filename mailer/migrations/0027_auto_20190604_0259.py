# Generated by Django 2.1.7 on 2019-06-04 02:59

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('mailer', '0026_auto_20190404_1132'),
    ]

    operations = [
        migrations.AlterField(
            model_name='comment',
            name='date_time',
            field=models.DateTimeField(default=datetime.datetime(2019, 6, 4, 2, 59, 47, 64579, tzinfo=utc)),
        ),
    ]
