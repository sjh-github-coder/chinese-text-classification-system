# -*- coding: utf-8 -*-
# Generated by Django 1.11.3 on 2020-03-17 06:45
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('text_emo', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='CorpusText',
            fields=[
                ('corpus_ID', models.AutoField(primary_key=True, serialize=False)),
                ('corpus_emo', models.FloatField()),
                ('corpus_content', models.TextField()),
            ],
        ),
    ]
