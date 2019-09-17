from django.contrib import admin
from django.urls import path
from . import views
from .views import SearchView


urlpatterns = [
    path('login/', views.index, name='login'),
    path('weather/', views.weather, name='weather'),
    path('', views.explore, name='explore'),
    path('search/', SearchView.as_view(), name='explore'),
    path('wheat/', views.wheat, name='wheat'),
    path('rice/', views.rice, name='rice'),
    path('sugarcane/', views.sugarcane, name='sugarcane'),
    path('potato/', views.potato, name='potato'),
    path('groundnut/', views.groundnut, name='groundnut'),
    path('cotton/', views.cotton, name='cotton'),
    path('diagnose/', views.autodiog, name='auto'),
    path('search/insect/', views.SearchInsectView.as_view(), name='inseact'),
    path('mango/', views.mango, name='mango'),
    path('i_rice/', views.irice, name='irice'),
    path('i_sugarcane/', views.isugarcane, name='isugarcane'),
    path('i_potato/', views.ipotato, name='ipotato'),
    path('tomato/', views.tomato, name='tomato'),
    path('i_cotton/', views.icotton, name='icotton'),


]
