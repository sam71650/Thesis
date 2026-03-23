from django.urls import path
from . import views

app_name = 'users'

urlpatterns = [
    path('', views.main, name='main'),
    path('register/', views.register, name='register'),
    path('login/', views.login, name='login'),
    path("delete/", views.delete_user, name="delete_user"),
    path('dashboard/', views.dashboard, name='dashboard'),
]