import json
import os
from django.core.management.base import BaseCommand
from stocks.models import Symbol

class Command(BaseCommand):
    help = 'Populate stock symbols from a JSON file (e.g., nse_symbols.json)'

    def handle(self, *args, **options):
        # Path to the JSON file; adjust as needed
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'fixtures', 'nse_symbols.json')
        if not os.path.exists(file_path):
            self.stdout.write(self.style.ERROR(f'File not found: {file_path}'))
            return

        with open(file_path, 'r') as f:
            data = json.load(f)

        count = 0
        for item in data:
            Symbol.objects.update_or_create(
                ticker=item.get('symbol'),
                defaults={
                    'name': item.get('name', ''),
                    'exchange': item.get('exchange', 'NSE'),
                    'instrument_token': item.get('instrument_token', ''),
                    'sector': item.get('sector', ''),
                }
            )
            count += 1

        self.stdout.write(self.style.SUCCESS(f'Successfully populated {count} symbols'))