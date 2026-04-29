from django.db import models

class Symbol(models.Model):
    ticker = models.CharField(max_length=20, unique=True)
    name = models.CharField(max_length=200)
    exchange = models.CharField(max_length=10, default='NSE')
    instrument_token = models.CharField(max_length=20, blank=True, null=True)
    sector = models.CharField(max_length=100, blank=True, null=True)

    def __str__(self):
        return f"{self.ticker} - {self.name}"