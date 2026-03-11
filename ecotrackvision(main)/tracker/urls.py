from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('plastic/', views.plastic_upload, name='plastic_upload'),
    path('plastic/history/', views.plastic_history, name='plastic_history'),
    path('water/', views.water_tracking, name='water_tracking'),
    path('electricity/', views.electricity_tracking, name='electricity_tracking'),
    path('profile/', views.profile_view, name='profile'),
    path('water/forecast/', views.water_forecast, name='water_forecast'),
    path('water/forecast/history/', views.water_forecast_history, name='water_forecast_history'),
    path('electricity/forecast/', views.electricity_forecast, name='electricity_forecast'),
    path('electricity/forecast/history/', views.electricity_forecast_history, name='electricity_forecast_history'),
    path('recommendations/', views.recommendations, name='recommendations'),
    path('ml-graph/<str:model_type>/', views.serve_ml_graph, name='ml_graph'),
    path('members/', views.admin_users_view, name='admin_users'),
]
