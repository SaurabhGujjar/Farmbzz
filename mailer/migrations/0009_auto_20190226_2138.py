# Generated by Django 2.1.7 on 2019-02-26 21:38

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('mailer', '0008_auto_20190226_2132'),
    ]

    operations = [
        migrations.AlterField(
            model_name='comment',
            name='date_time',
            field=models.DateTimeField(default=datetime.datetime(2019, 2, 26, 21, 38, 33, 494782, tzinfo=utc)),
        ),
    ]
