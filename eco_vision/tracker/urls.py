from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('plastic/', views.plastic_upload, name='plastic_upload'),
    path('water/', views.water_tracking, name='water_tracking'),
    path('electricity/', views.electricity_tracking, name='electricity_tracking'),
    path('profile/', views.profile_view, name='profile'),
    path('water/forecast/', views.water_forecast, name='water_forecast'),
    path('electricity/forecast/', views.electricity_forecast, name='electricity_forecast'),
    path('ml-graph/<str:model_type>/', views.serve_ml_graph, name='ml_graph'),
]
