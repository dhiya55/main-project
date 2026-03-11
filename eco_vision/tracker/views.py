import os
import sys
import json
import datetime
from django.shortcuts import render, redirect
from django.http import HttpResponse, FileResponse
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from .models import Profile, WaterUsage, ElectricityUsage, PlasticEntry
from ultralytics import YOLO
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import cv2
from io import BytesIO
from PIL import Image

def get_user_scores(user):
    if not user.is_authenticated:
        return {}
    current_month = datetime.datetime.now().month
    current_year = datetime.datetime.now().year
    
    plastic_entries = PlasticEntry.objects.filter(
        user=user, 
        created_at__month=current_month, 
        created_at__year=current_year
    )
    total_plastic_scans = plastic_entries.count()
    plastic_score = max(0, 100 - (total_plastic_scans * 5))
    
    water_entries = WaterUsage.objects.filter(
        user=user,
        date__month=current_month,
        date__year=current_year
    )
    total_water_consumption = sum(entry.consumption for entry in water_entries)
    water_score = max(0, 100 - (total_water_consumption / 10)) if total_water_consumption else 100
    if not water_entries:
         water_score = 100

    electricity_entries = ElectricityUsage.objects.filter(
        user=user,
        date__month=current_month,
        date__year=current_year
    )
    total_electricity_units = sum(entry.units for entry in electricity_entries)
    electricity_score = max(0, 100 - (total_electricity_units * 2)) if total_electricity_units else 100
    if not electricity_entries:
         electricity_score = 100
    
    eco_score = round((plastic_score + water_score + electricity_score) / 3)
    
    return {
        'eco_score': eco_score,
        'plastic_score': int(plastic_score),
        'water_usage': int(total_water_consumption) if total_water_consumption else 0,
        'water_score': int(water_score),
        'electricity_usage': int(total_electricity_units) if total_electricity_units else 0,
        'electricity_score': int(electricity_score)
    }



# ── ML Predictor path ──────────────────────────────────────────────────────
ML_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ml')
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

ML_RESULTS_PATH = os.path.join(ML_DIR, 'models', 'training_results.json')

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
    context = get_user_scores(request.user)
    return render(request, 'tracker/dashboard.html', context)

@login_required
def plastic_upload(request):
    prediction_result = None
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        
        # Save the uploaded file temporarily
        path = default_storage.save('tmp/current_waste_image.jpg', ContentFile(image_file.read()))
        tmp_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'media', path)
        
        try:
            # Path to the model
            model_path = os.path.join(ML_DIR, 'best (2).pt')
            
            # Load and run inference
            model = YOLO(model_path)
            results = model.predict(tmp_file_path)
            
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

    scores = get_user_scores(request.user)
    context = {'prediction_result': prediction_result}
    context.update(scores)
    return render(request, 'tracker/plastic_upload.html', context)

@login_required
def water_tracking(request):
    return render(request, 'tracker/water_tracking.html')

@login_required
def electricity_tracking(request):
    return render(request, 'tracker/electricity_tracking.html')

@login_required
def profile_view(request):
    return render(request, 'tracker/profile.html')

@login_required
def water_forecast(request):
    prediction = None
    form_data  = {}
    error_msg  = None

    if request.method == 'POST':
        try:
            from predictor import predict_water
            form_data = {
                'household_size':       request.POST.get('household_size', '4'),
                'income_level':         request.POST.get('income_level', 'middle'),
                'property_type':        request.POST.get('property_type', 'house'),
                'dwelling_area_sqm':    request.POST.get('dwelling_area_sqm', '100'),
                'has_garden':           request.POST.get('has_garden', '0'),
                'num_bathrooms':        request.POST.get('num_bathrooms', '2'),
                'has_dishwasher':       request.POST.get('has_dishwasher', '0'),
                'has_washing_machine':  request.POST.get('has_washing_machine', '1'),
                'occupants_children':   request.POST.get('occupants_children', '0'),
                'water_price_per_m3':   request.POST.get('water_price_per_m3', '1.5'),
                'temperature_c':        request.POST.get('temperature_c', '25'),
            }
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
        except Exception as e:
            error_msg = str(e)

    # UI Helpers for templates to avoid formatter space-stripping on ==
    form_data['is_house'] = form_data.get('property_type') == 'house'
    form_data['is_apartment'] = form_data.get('property_type') == 'apartment'
    form_data['is_low_income'] = form_data.get('income_level') == 'low'
    form_data['is_middle_income'] = form_data.get('income_level') == 'middle' or not form_data.get('income_level')
    form_data['is_high_income'] = form_data.get('income_level') == 'high'
    form_data['is_ac_on'] = form_data.get('has_air_conditioner') == '1'
    form_data['is_heating_on'] = form_data.get('has_electric_heating') == '1'
    form_data['is_ev_on'] = form_data.get('has_ev') == '1'
    form_data['is_garden_on'] = form_data.get('has_garden') == '1'
    form_data['is_dishwasher_on'] = form_data.get('has_dishwasher') == '1'
    form_data['is_laundry_on'] = form_data.get('has_washing_machine') == '1' or not form_data.get('has_washing_machine')

    prediction_per_person = None
    if prediction is not None and form_data.get('household_size'):
        try:
            h_size = int(form_data['household_size'])
            if h_size > 0:
                prediction_per_person = round(prediction / h_size, 1)
        except:
            pass

    scores = get_user_scores(request.user)
    context = {
        'prediction':            prediction,
        'prediction_per_person': prediction_per_person,
        'form_data':             form_data,
        'error_msg':             error_msg,
    }
    context.update(scores)
    return render(request, 'tracker/water_forecast.html', context)

@login_required
def electricity_forecast(request):
    prediction = None
    form_data  = {}
    error_msg  = None

    if request.method == 'POST':
        try:
            from predictor import predict_electricity
            form_data = {
                'household_size':               request.POST.get('household_size', '4'),
                'income_level':                 request.POST.get('income_level', 'middle'),
                'property_type':                request.POST.get('property_type', 'house'),
                'dwelling_area_sqm':            request.POST.get('dwelling_area_sqm', '100'),
                'num_occupants_work_from_home': request.POST.get('num_occupants_work_from_home', '0'),
                'has_air_conditioner':          request.POST.get('has_air_conditioner', '0'),
                'has_electric_heating':         request.POST.get('has_electric_heating', '0'),
                'has_ev':                       request.POST.get('has_ev', '0'),
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
        except Exception as e:
            error_msg = str(e)

    # UI Helpers for templates
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

    scores = get_user_scores(request.user)
    context = {
        'prediction':            prediction,
        'prediction_per_person': prediction_per_person,
        'form_data':             form_data,
        'error_msg':             error_msg,
    }
    context.update(scores)
    return render(request, 'tracker/electricity_forecast.html', context)

def serve_ml_graph(request, model_type='water'):
    """Serve the model comparison PNG graph."""
    if model_type == 'electricity':
        graph_path = os.path.join(ML_DIR, 'models', 'electricity_model_comparison.png')
    else:
        graph_path = os.path.join(ML_DIR, 'models', 'model_comparison.png')
        
    if os.path.exists(graph_path):
        return FileResponse(open(graph_path, 'rb'), content_type='image/png')
    return HttpResponse('Graph not found.', status=404)

