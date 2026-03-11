import os
import sys
import json
import datetime
from django.shortcuts import render, redirect
from django.http import HttpResponse, FileResponse
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from .models import Profile, WaterUsage, ElectricityUsage, PlasticEntry, WaterForecast, ElectricityForecast
from ultralytics import YOLO
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import cv2
from io import BytesIO
from PIL import Image
from django.conf import settings
from pathlib import Path

# ML path for data and models
ML_DIR = str(Path(settings.BASE_DIR) / 'ML')
ML_RESULTS_PATH = os.path.join(ML_DIR, 'models', 'training_results.json')

# Global variable for lazy loading YOLO model
_yolo_model = None

def get_yolo_model():
    """Lazy-load the YOLO model once during the application lifetime."""
    global _yolo_model
    if _yolo_model is None:
        model_path = os.path.join(ML_DIR, 'best (2).pt')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"YOLO model not found at {model_path}")
        _yolo_model = YOLO(model_path)
    return _yolo_model

def index(request):
    return render(request, 'tracker/index.html')

def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('dashboard')
    else:
        form = UserCreationForm()
    return render(request, 'tracker/register.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('dashboard')
    else:
        form = AuthenticationForm()
    return render(request, 'tracker/login.html', {'form': form})

@login_required
def logout_view(request):
    logout(request)
    return redirect('index')

@login_required
def dashboard(request):
    from django.db.models import Avg
    from datetime import date
    
    # Check if entries exist for today
    today_water = WaterUsage.objects.filter(user=request.user, date=date.today()).exists()
    today_elec = ElectricityUsage.objects.filter(user=request.user, date=date.today()).exists()
    missing_entry = not (today_water or today_elec)
    
    # 1. Plastic Score (Lower usage is better)
    # Threshold: 10 items per day
    from django.db.models import Sum, Count
    plastic_entries = PlasticEntry.objects.filter(user=request.user)
    total_plastic_items = plastic_entries.aggregate(Sum('object_count'))['object_count__sum'] or 0
    total_plastic_days = plastic_entries.dates('created_at', 'day').count() or 1
    avg_plastics_per_day = total_plastic_items / total_plastic_days
    plastic_score = round(max(0, 100 * (1 - (avg_plastics_per_day / 10))), 1)

    # 2. Water Consumption (Daily Average vs 300L Threshold)
    water_logs = WaterUsage.objects.filter(user=request.user)
    if water_logs.exists():
        total_water = water_logs.aggregate(Sum('consumption'))['consumption__sum'] or 0
        total_water_days = water_logs.dates('date', 'day').count() or 1
        water_usage = total_water / total_water_days
    else:
        latest_forecast = WaterForecast.objects.filter(user=request.user).order_by('-date', '-id').first()
        water_usage = latest_forecast.consumption_forecast if latest_forecast else 0
    water_usage = round(water_usage, 1)

    # 3. Electricity Usage (Daily Average vs 30kWh Threshold)
    elec_logs = ElectricityUsage.objects.filter(user=request.user)
    if elec_logs.exists():
        total_elec = elec_logs.aggregate(Sum('units'))['units__sum'] or 0
        total_elec_days = elec_logs.dates('date', 'day').count() or 1
        electricity_usage = total_elec / total_elec_days
    else:
        latest_forecast = ElectricityForecast.objects.filter(user=request.user).order_by('-date', '-id').first()
        electricity_usage = latest_forecast.consumption_forecast if latest_forecast else 0
    electricity_usage = round(electricity_usage, 1)

    # Calculate Eco Score based on thresholds (Water: 300L, Electricity: 30kWh)
    w_score = max(0, 100 * (1 - (water_usage / 300 if water_usage > 0 else 0)))
    e_score = max(0, 100 * (1 - (electricity_usage / 30 if electricity_usage > 0 else 0)))
    p_score = plastic_score
    
    # EcoScore = (PlasticScore × 0.40) + (WaterScore × 0.30) + (EnergyScore × 0.30)
    eco_score = round((p_score * 0.4) + (w_score * 0.3) + (e_score * 0.3), 1)

    # Determine Category
    if eco_score < 40:
        eco_label = "Poor"
        eco_meaning = "Unsustainable usage"
        eco_class = "danger"
    elif eco_score < 70:
        eco_label = "Moderate"
        eco_meaning = "Average sustainability"
        eco_class = "warning text-dark"
    else:
        eco_label = "Good"
        eco_meaning = "Environmentally responsible"
        eco_class = "success"

    context = {
        'eco_score': eco_score,
        'eco_label': eco_label,
        'eco_meaning': eco_meaning,
        'eco_class': eco_class,
        'plastic_score': plastic_score,
        'water_usage': water_usage,
        'electricity_usage': electricity_usage,
        'water_score': round(w_score, 1),
        'elec_score': round(e_score, 1),
        'missing_daily_entry': missing_entry,
        # Achievement Badges (Score > 70)
        'plastic_badge': p_score >= 70 and plastic_entries.exists(),
        'water_badge': w_score >= 70 and (water_logs.exists() or WaterForecast.objects.filter(user=request.user).exists()),
        'energy_badge': e_score >= 70 and (elec_logs.exists() or ElectricityForecast.objects.filter(user=request.user).exists()),
    }

    # Admin Statistics
    if request.user.is_superuser:
        from django.contrib.auth.models import User
        context['is_admin'] = True
        context['total_users'] = User.objects.count()
        
        # Total Analysis Counts (Entries + Forecasts)
        p_count = PlasticEntry.objects.count()
        w_count = WaterUsage.objects.count() + WaterForecast.objects.count()
        e_count = ElectricityUsage.objects.count() + ElectricityForecast.objects.count()
        
        context['total_analysis'] = p_count + w_count + e_count
        context['admin_stats'] = {
            'plastic': p_count,
            'water': w_count,
            'electricity': e_count
        }

    return render(request, 'tracker/dashboard.html', context)

@login_required
def plastic_upload(request):
    prediction_result = None
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        
        # Save the uploaded file temporarily
        path = default_storage.save('tmp/current_waste_image.jpg', ContentFile(image_file.read()))
        # Use settings.MEDIA_ROOT for reliable path resolution
        tmp_file_path = str(settings.MEDIA_ROOT / path)
        
        try:
            # Load and run inference using lazy loading
            model = get_yolo_model()
            # use workers=0 to avoid multiprocessing issues on Windows (WinError 123)
            results = model.predict(tmp_file_path, workers=0)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Count total detections
                object_count = len(results[0].boxes)
                
                # Extract inference time (in ms)
                inference_time = results[0].speed.get('inference', 0.0)
                
                # Get the highest confidence detection for primary labeling
                best_box = results[0].boxes[0]
                class_id = int(best_box.cls[0])
                label = model.names[class_id]
                confidence = float(best_box.conf[0])
                
                # Categorization mapping
                recyclable_items = ['plastic', 'pet', 'glass', 'can', 'paper', 'cardboard']
                biodegradable_items = ['biodegradable', 'food', 'vegetable', 'fruit', 'plant', 'organic', 'leaf']
                
                is_recyclable = any(item in label.lower() for item in recyclable_items)
                is_biodegradable = any(item in label.lower() for item in biodegradable_items)
                
                # Determine guidance and labels
                if is_biodegradable:
                    guidance = "Add to your compost bin or organic waste collection."
                    status_label = "Compostable"
                    status_class = "success" # Green
                elif is_recyclable:
                    guidance = "Rinse thoroughly before recycling."
                    status_label = "Recyclable"
                    status_class = "success" # Green
                else:
                    guidance = "Dispose of in general waste."
                    status_label = "Non-Recyclable"
                    status_class = "danger" # Red
                
                # Generate annotated image
                annotated_img_array = results[0].plot() # BGR array
                annotated_img_rgb = cv2.cvtColor(annotated_img_array, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(annotated_img_rgb)
                
                # Save annotated image to memory
                buffer = BytesIO()
                pil_img.save(buffer, format="JPEG")
                annotated_content = ContentFile(buffer.getvalue())
                
                # Save to database
                entry = PlasticEntry.objects.create(
                    user=request.user,
                    image=image_file,
                    plastic_type=label.capitalize(),
                    is_recyclable=is_recyclable,
                    plastic_score=round(confidence * 100, 2),
                    confidence_score=round(confidence, 4),
                    object_count=object_count,
                    inference_time=inference_time
                )
                
                # Save annotated image
                entry.annotated_image.save(f'annotated_{entry.id}.jpg', annotated_content)
                
                prediction_result = {
                    'label': label.capitalize(),
                    'is_recyclable': is_recyclable,
                    'is_biodegradable': is_biodegradable,
                    'status_label': status_label,
                    'status_class': status_class,
                    'confidence': round(confidence * 100, 1),
                    'object_count': object_count,
                    'inference_time': round(inference_time, 2),
                    'original_image_url': entry.image.url,
                    'annotated_image_url': entry.annotated_image.url,
                    'guidance': guidance,
                    'alternatives': ["Glass bottles", "Stainless steel containers"] if "plastic" in label.lower() else []
                }
            else:
                prediction_result = {'error': "No waste items detected in the image."}
                
        except Exception as e:
            prediction_result = {'error': f"Error during analysis: {str(e)}"}
        finally:
            # Clean up temp file
            if default_storage.exists(path):
                default_storage.delete(path)

    return render(request, 'tracker/plastic_upload.html', {'prediction_result': prediction_result})

@login_required
def plastic_history(request):
    if request.user.is_superuser:
        entries = PlasticEntry.objects.all().order_by('-created_at')
    else:
        entries = PlasticEntry.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'tracker/plastic_history.html', {'entries': entries})

@login_required
def water_tracking(request):
    return render(request, 'tracker/water_tracking.html')

@login_required
def electricity_tracking(request):
    return render(request, 'tracker/electricity_tracking.html')

@login_required
def profile_view(request):
    from django.db.models import Sum
    from django.contrib import messages
    from django.contrib.auth import update_session_auth_hash

    if request.method == 'POST':
        full_name = request.POST.get('full_name', '').split(' ')
        request.user.first_name = full_name[0]
        request.user.last_name = ' '.join(full_name[1:]) if len(full_name) > 1 else ''
        request.user.email = request.POST.get('email', '')
        
        new_password = request.POST.get('password', '')
        if new_password:
            request.user.set_password(new_password)
            request.user.save()
            update_session_auth_hash(request, request.user)  # Keep user logged in
        else:
            request.user.save()
            
        messages.success(request, 'Your profile has been updated successfully!')
        return redirect('profile')

    # 1. Plastic Score
    plastic_entries = PlasticEntry.objects.filter(user=request.user)
    total_plastic_items = plastic_entries.aggregate(Sum('object_count'))['object_count__sum'] or 0
    total_plastic_days = plastic_entries.dates('created_at', 'day').count() or 1
    avg_plastics_per_day = total_plastic_items / total_plastic_days
    p_score = round(max(0, 100 * (1 - (avg_plastics_per_day / 10))), 1)

    # 2. Water Score
    water_logs = WaterUsage.objects.filter(user=request.user)
    if water_logs.exists():
        total_water = water_logs.aggregate(Sum('consumption'))['consumption__sum'] or 0
        total_water_days = water_logs.dates('date', 'day').count() or 1
        water_usage = total_water / total_water_days
    else:
        latest_forecast = WaterForecast.objects.filter(user=request.user).order_by('-date', '-id').first()
        water_usage = latest_forecast.consumption_forecast if latest_forecast else 0
    w_score = max(0, 100 * (1 - (water_usage / 300 if water_usage > 0 else 0)))

    # 3. Energy Score
    elec_logs = ElectricityUsage.objects.filter(user=request.user)
    if elec_logs.exists():
        total_elec = elec_logs.aggregate(Sum('units'))['units__sum'] or 0
        total_elec_days = elec_logs.dates('date', 'day').count() or 1
        electricity_usage = total_elec / total_elec_days
    else:
        latest_forecast = ElectricityForecast.objects.filter(user=request.user).order_by('-date', '-id').first()
        electricity_usage = latest_forecast.consumption_forecast if latest_forecast else 0
    e_score = max(0, 100 * (1 - (electricity_usage / 30 if electricity_usage > 0 else 0)))

    context = {
        'plastic_badge': p_score >= 70 and plastic_entries.exists(),
        'water_badge': w_score >= 70 and (water_logs.exists() or WaterForecast.objects.filter(user=request.user).exists()),
        'energy_badge': e_score >= 70 and (elec_logs.exists() or ElectricityForecast.objects.filter(user=request.user).exists()),
    }
    return render(request, 'tracker/profile.html', context)

@login_required
def water_forecast(request):
    prediction = None
    form_data  = {}
    error_msg  = None

    if request.method == 'POST':
        try:
            from ML.predictor import predict_water
            
            # Get data with defaults, handling switches (checkboxes) correctly
            form_data = {
                'household_size':       request.POST.get('household_size', '4'),
                'income_level':         request.POST.get('income_level', 'middle'),
                'property_type':        request.POST.get('property_type', 'house'),
                'dwelling_area_sqm':    request.POST.get('dwelling_area_sqm', '100'),
                'has_garden':           '1' if request.POST.get('has_garden') else '0',
                'num_bathrooms':        request.POST.get('num_bathrooms', '2'),
                'has_dishwasher':       '1' if request.POST.get('has_dishwasher') else '0',
                'has_washing_machine':  '1' if request.POST.get('has_washing_machine') else '0',
                'occupants_children':   request.POST.get('occupants_children', '0'),
                'water_price_per_m3':   request.POST.get('water_price_per_m3', '1.5'),
                'temperature_c':        request.POST.get('temperature_c', '25'),
            }
            
            # 1. Run inference
            prediction = predict_water(
                household_size      = int(form_data['household_size']),
                income_level        = form_data['income_level'],
                property_type       = form_data['property_type'],
                dwelling_area_sqm   = float(form_data['dwelling_area_sqm']),
                has_garden          = int(form_data['has_garden']),
                num_bathrooms       = int(form_data['num_bathrooms']),
                has_dishwasher      = int(form_data['has_dishwasher']),
                has_washing_machine = int(form_data['has_washing_machine']),
                occupants_children  = int(form_data['occupants_children']),
                water_price_per_m3  = float(form_data['water_price_per_m3']),
                temperature_c       = float(form_data['temperature_c']),
            )
            
            # 2. Save result to DB
            if prediction is not None:
                WaterForecast.objects.create(
                    user=request.user,
                    consumption_forecast=prediction
                )
        except Exception as e:
            error_msg = f"Inference Error: {str(e)}"
            import traceback
            traceback.print_exc()

    # UI Helpers for templates
    if form_data is None:
        form_data = {}
        
    form_data['is_house'] = form_data.get('property_type') == 'house'
    form_data['is_apartment'] = form_data.get('property_type') == 'apartment'
    form_data['is_low_income'] = form_data.get('income_level') == 'low'
    form_data['is_middle_income'] = form_data.get('income_level') == 'middle' or not form_data.get('income_level')
    form_data['is_high_income'] = form_data.get('income_level') == 'high'
    form_data['is_garden_on'] = form_data.get('has_garden') == '1'
    form_data['is_dishwasher_on'] = form_data.get('has_dishwasher') == '1'
    form_data['is_laundry_on'] = form_data.get('has_washing_machine') == '1'

    prediction_per_person = None
    if prediction is not None and form_data.get('household_size'):
        try:
            h_size = int(form_data['household_size'])
            if h_size > 0:
                prediction_per_person = round(prediction / h_size, 1)
        except:
            pass

    return render(request, 'tracker/water_forecast.html', {
        'prediction':            prediction,
        'prediction_per_person': prediction_per_person,
        'form_data':             form_data,
        'error_msg':             error_msg,
    })

@login_required
def electricity_forecast(request):
    prediction = None
    form_data  = {}
    error_msg  = None

    if request.method == 'POST':
        try:
            from ML.predictor import predict_electricity
            form_data = {
                'household_size':               request.POST.get('household_size', '4'),
                'income_level':                 request.POST.get('income_level', 'middle'),
                'property_type':                request.POST.get('property_type', 'house'),
                'dwelling_area_sqm':            request.POST.get('dwelling_area_sqm', '100'),
                'num_occupants_work_from_home': request.POST.get('num_occupants_work_from_home', '0'),
                'has_air_conditioner':          '1' if request.POST.get('has_air_conditioner') else '0',
                'has_electric_heating':         '1' if request.POST.get('has_electric_heating') else '0',
                'has_ev':                       '1' if request.POST.get('has_ev') else '0',
                'num_major_appliances':         request.POST.get('num_major_appliances', '5'),
                'temperature_c':                request.POST.get('temperature_c', '25'),
                'electricity_price_per_kwh':    request.POST.get('electricity_price_per_kwh', '0.15'),
            }
            prediction = predict_electricity(
                household_size               = int(form_data['household_size']),
                income_level                 = form_data['income_level'],
                property_type                = form_data['property_type'],
                dwelling_area_sqm            = float(form_data['dwelling_area_sqm']),
                num_occupants_work_from_home = int(form_data['num_occupants_work_from_home']),
                has_air_conditioner          = int(form_data['has_air_conditioner']),
                has_electric_heating         = int(form_data['has_electric_heating']),
                has_ev                       = int(form_data['has_ev']),
                num_major_appliances         = int(form_data['num_major_appliances']),
                temperature_c                = float(form_data['temperature_c']),
                electricity_price_per_kwh    = float(form_data['electricity_price_per_kwh']),
            )
            
            # Save the forecast to the database
            if prediction is not None:
                ElectricityForecast.objects.create(
                    user=request.user,
                    consumption_forecast=prediction
                )
        except Exception as e:
            error_msg = f"Inference Error: {str(e)}"
            import traceback
            traceback.print_exc()

    # UI Helpers for templates
    if form_data is None:
        form_data = {}
    form_data['is_house'] = form_data.get('property_type') == 'house'
    form_data['is_apartment'] = form_data.get('property_type') == 'apartment'
    form_data['is_low_income'] = form_data.get('income_level') == 'low'
    form_data['is_middle_income'] = form_data.get('income_level') == 'middle' or not form_data.get('income_level')
    form_data['is_high_income'] = form_data.get('income_level') == 'high'
    form_data['is_ac_on'] = form_data.get('has_air_conditioner') == '1'
    form_data['is_heating_on'] = form_data.get('has_electric_heating') == '1'
    form_data['is_ev_on'] = form_data.get('has_ev') == '1'

    prediction_per_person = None
    if prediction is not None and form_data.get('household_size'):
        try:
            h_size = int(form_data['household_size'])
            if h_size > 0:
                prediction_per_person = round(prediction / h_size, 2)
        except:
            pass

    return render(request, 'tracker/electricity_forecast.html', {
        'prediction':            prediction,
        'prediction_per_person': prediction_per_person,
        'form_data':             form_data,
        'error_msg':             error_msg,
    })

@login_required
def water_forecast_history(request):
    if request.user.is_superuser:
        forecasts = WaterForecast.objects.all().order_by('-date', '-id')
    else:
        forecasts = WaterForecast.objects.filter(user=request.user).order_by('-date', '-id')
    return render(request, 'tracker/water_forecast_history.html', {'forecasts': forecasts})

@login_required
def electricity_forecast_history(request):
    if request.user.is_superuser:
        forecasts = ElectricityForecast.objects.all().order_by('-date', '-id')
    else:
        forecasts = ElectricityForecast.objects.filter(user=request.user).order_by('-date', '-id')
    return render(request, 'tracker/electricity_forecast_history.html', {'forecasts': forecasts})

@login_required
def recommendations(request):
    from django.db.models import Avg
    
    # Calculate scores (Same logic as dashboard)
    # Plastic
    from django.db.models import Sum
    plastic_entries = PlasticEntry.objects.filter(user=request.user)
    total_plastic_items = plastic_entries.aggregate(Sum('object_count'))['object_count__sum'] or 0
    total_plastic_days = plastic_entries.dates('created_at', 'day').count() or 1
    avg_plastics_per_day = total_plastic_items / total_plastic_days
    p_score = round(max(0, 100 * (1 - (avg_plastics_per_day / 10))), 1)

    # Water
    water_logs = WaterUsage.objects.filter(user=request.user)
    if water_logs.exists():
        total_water = water_logs.aggregate(Sum('consumption'))['consumption__sum'] or 0
        total_water_days = water_logs.dates('date', 'day').count() or 1
        water_usage = total_water / total_water_days
    else:
        latest_forecast = WaterForecast.objects.filter(user=request.user).order_by('-date', '-id').first()
        water_usage = latest_forecast.consumption_forecast if latest_forecast else 0
    w_score = max(0, 100 * (1 - (water_usage / 300 if water_usage > 0 else 0)))

    # Electricity
    elec_logs = ElectricityUsage.objects.filter(user=request.user)
    if elec_logs.exists():
        total_elec = elec_logs.aggregate(Sum('units'))['units__sum'] or 0
        total_elec_days = elec_logs.dates('date', 'day').count() or 1
        electricity_usage = total_elec / total_elec_days
    else:
        latest_forecast = ElectricityForecast.objects.filter(user=request.user).order_by('-date', '-id').first()
        electricity_usage = latest_forecast.consumption_forecast if latest_forecast else 0
    e_score = max(0, 100 * (1 - (electricity_usage / 30 if electricity_usage > 0 else 0)))

    # Compile Recommendations
    recs = []
    
    # 1. Plastic: DIY, Reuse & Alternatives (Only if entries exist)
    if plastic_entries.exists():
        recyclable_count = plastic_entries.filter(is_recyclable=True).count()
        if recyclable_count > 0:
            recs.append({
                'cat': 'Plastic DIY', 
                'icon': 'fa-hammer', 
                'color': 'success', 
                'title': 'Upcycling & Reuse', 
                'text': f'You have {recyclable_count} recyclable items! Transform plastic bottles into self-watering vertical gardens or use rigid containers as durable drawer organizers.'
            })
        
        if p_score < 70:
            recs.append({
                'cat': 'Plastic Reduction', 
                'icon': 'fa-leaf', 
                'color': 'success', 
                'title': 'Eco-Friendly Alternatives', 
                'text': 'Switch to bamboo toothbrushes, glass storage containers, and beeswax wraps. These small swaps significantly reduce your long-term plastic waste.'
            })

    # 2. Water: Tips & Conservation (Only if logs or forecasts exist)
    if water_logs.exists() or WaterForecast.objects.filter(user=request.user).exists():
        if w_score < 60:
            recs.append({
                'cat': 'Water Saving', 
                'icon': 'fa-faucet-drip', 
                'color': 'info', 
                'title': 'High Usage Detected', 
                'text': 'Fix any leaking taps immediately—a single drip can waste 15L a day. Consider installing low-flow showerheads and dual-flush toilet converters to save up to 40% more water.'
            })
        
        recs.append({
            'cat': 'Water Reuse', 
            'icon': 'fa-recycle', 
            'color': 'info', 
            'title': 'Greywater Recycling', 
            'text': 'Reuse water from washing vegetables or RO waste to water your plants. This simple habit turns "used" water into a resource for your home garden.'
        })

    # 3. Electricity: Efficiency & Alternatives (Only if logs or forecasts exist)
    if elec_logs.exists() or ElectricityForecast.objects.filter(user=request.user).exists():
        if e_score < 60:
            recs.append({
                'cat': 'Energy Efficiency', 
                'icon': 'fa-bolt-lightning', 
                'color': 'warning', 
                'title': 'Power Savings', 
                'text': 'Switch to 5-star rated BLDC fans and LED bulbs. They consume 50-70% less power than traditional alternatives, directly cutting your monthly bills.'
            })
        
        recs.append({
            'cat': 'Energy Habit', 
            'icon': 'fa-power-off', 
            'color': 'warning', 
            'title': 'Vampire Power', 
            'text': 'Appliances on standby still consume "vampire" energy. Unplug chargers, microwave, and TVs when not in use to shave 5-10% off your daily consumption.'
        })

    if not recs:
        recs.append({'cat': 'General', 'icon': 'fa-clipboard-check', 'color': 'primary', 'title': 'Get Started!', 'text': 'Log your first plastic item, water usage, or run an AI forecast to see personalized sustainability tips here.'})

    context = {
        'p_score': p_score,
        'w_score': w_score,
        'e_score': e_score,
        'recommendations': recs,
    }
    return render(request, 'tracker/recommendations.html', context)

def serve_ml_graph(request, model_type='water'):
    """Serve the model comparison PNG graph."""
    if model_type == 'electricity':
        graph_path = os.path.join(ML_DIR, 'models', 'electricity_model_comparison.png')
    else:
        graph_path = os.path.join(ML_DIR, 'models', 'model_comparison.png')
        
    if os.path.exists(graph_path):
        return FileResponse(open(graph_path, 'rb'), content_type='image/png')
    return HttpResponse('Graph not found.', status=404)

@login_required
def admin_users_view(request):
    if not request.user.is_superuser:
        return redirect('dashboard')
    
    from django.contrib.auth.models import User
    users = User.objects.all().order_by('-date_joined')
    return render(request, 'tracker/admin_users.html', {'users': users})

