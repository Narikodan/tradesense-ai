import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'tradesense_ai.settings') # might not be needed if standalone

from analysis.engine import _grade_trade
print(_grade_trade(2.0, 60, 1.0))
print(_grade_trade(1.0, 50, 1.0))
print(_grade_trade(3.0, 80, 0.4))
print(_grade_trade(3.0, 80, 0.2))
