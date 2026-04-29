from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('search/', views.search_suggestions, name='search_suggestions'),
    path('analyze/', views.analyze_stock, name='analyze_stock'),
    path('picks/', views.top_picks_view, name='top_picks'),   # new
]