from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    household_size = models.IntegerField(default=1)
    location = models.CharField(max_length=255, blank=True, null=True)

    def __str__(self):
        return f"{self.user.username}'s Profile"

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.profile.save()

class PlasticEntry(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='plastic_images/')
    annotated_image = models.ImageField(upload_to='annotated_images/', blank=True, null=True)
    plastic_type = models.CharField(max_length=100, blank=True)
    is_recyclable = models.BooleanField(default=False)
    plastic_score = models.FloatField(default=0.0)
    confidence_score = models.FloatField(default=0.0)
    object_count = models.IntegerField(default=0)
    inference_time = models.FloatField(default=0.0, help_text="Inference time in ms")
    created_at = models.DateTimeField(auto_now_add=True)

class WaterUsage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    consumption = models.FloatField(help_text="Consumption in Liters")
    date = models.DateField()
    score = models.FloatField(default=0)

class ElectricityUsage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    units = models.FloatField(help_text="Units in kWh")
    date = models.DateField()
    score = models.FloatField(default=0)

class WaterForecast(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateField(auto_now_add=True)
    consumption_forecast = models.FloatField()

    @property
    def monthly_forecast(self):
        return self.consumption_forecast * 30

    def __str__(self):
        return f"{self.user.username} - {self.date} - {self.consumption_forecast}L"

class ElectricityForecast(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateField(auto_now_add=True)
    consumption_forecast = models.FloatField()

    @property
    def monthly_forecast(self):
        return self.consumption_forecast * 30

    def __str__(self):
        return f"{self.user.username} - {self.date} - {self.consumption_forecast}kWh"
