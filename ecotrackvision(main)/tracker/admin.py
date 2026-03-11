from django.contrib import admin
from .models import Profile, PlasticEntry, WaterUsage, ElectricityUsage, WaterForecast, ElectricityForecast

@admin.register(WaterForecast)
class WaterForecastAdmin(admin.ModelAdmin):
    list_display = ('user', 'date', 'consumption_forecast')
    list_filter = ('date',)

@admin.register(ElectricityForecast)
class ElectricityForecastAdmin(admin.ModelAdmin):
    list_display = ('user', 'date', 'consumption_forecast')
    list_filter = ('date',)

@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'household_size', 'location')
    search_fields = ('user__username', 'location')

@admin.register(PlasticEntry)
class PlasticEntryAdmin(admin.ModelAdmin):
    list_display = ('user', 'plastic_type', 'is_recyclable', 'created_at')
    list_filter = ('is_recyclable', 'plastic_type')

@admin.register(WaterUsage)
class WaterUsageAdmin(admin.ModelAdmin):
    list_display = ('user', 'consumption', 'date', 'score')
    list_filter = ('date',)

@admin.register(ElectricityUsage)
class ElectricityUsageAdmin(admin.ModelAdmin):
    list_display = ('user', 'units', 'date', 'score')
    list_filter = ('date',)
