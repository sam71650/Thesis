from django.urls import path
from . import views

app_name = 'research'

urlpatterns = [
    path('', views.main, name='research'),
]
