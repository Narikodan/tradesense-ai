from django.contrib import admin
from .models import Symbol

@admin.register(Symbol)
class SymbolAdmin(admin.ModelAdmin):
    list_display = ('ticker', 'name', 'exchange')
    search_fields = ('ticker', 'name')