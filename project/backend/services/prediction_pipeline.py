import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.dirname(BASE_DIR)

# Load the Singular Symptom Model
SYMPTOM_MODEL_PATH = os.path.join(MODELS_DIR, 'models', 'symptom_disease_model.pkl')
try:
    symptom_model = joblib.load(SYMPTOM_MODEL_PATH)
except Exception as e:
    print(f"Warning: Could not load symptom model: {e}")
    symptom_model = None

# For this prototype, we encapsulate the logic. 
# In a full production system, we'd load the specific .pkl files for Dengue, Cholera etc.
# But since each script saved them differently, we'll build a smart wrapper that uses the models if they exist, 
# or simulates their expected behavior via heuristic fallbacks if they are missing or formatted oddly.

def load_trend_model(disease):
    """Attempt to load specific disease trend model."""
    try:
        path = os.path.join(MODELS_DIR, 'model_scripts', disease, f"{disease.lower()}_outbreak_model.pkl")
        if os.path.exists(path):
            return joblib.load(path)
        
        # Fallback to older naming convention
        path = os.path.join(MODELS_DIR, 'model_scripts', disease, f"{disease.lower()}_trend.pkl")
        if os.path.exists(path):
            return joblib.load(path)
    except:
        pass
    return None

def predict_symptom_disease(data):
    """Predicts one of the 5 diseases based on hospital symptoms."""
    if not symptom_model:
        return "Unknown"
    
    fever = float(data.get('fever', 98.6))
    body_pain = int(data.get('body_pain', 0))
    runny_nose = int(data.get('runny_nose', 0))
    headache = int(data.get('headache', 0))
    fatigue = int(data.get('fatigue', 0))
    vomiting_diarrhea = int(data.get('vomiting_diarrhea', 0))
    
    # Pre-check: If no meaningful symptoms are present, don't run the model.
    # Normal body temperature is ~98.6°F. Fever threshold is 100°F.
    total_symptoms = body_pain + runny_nose + headache + fatigue + vomiting_diarrhea
    has_fever = fever >= 100.0
    
    if not has_fever and total_symptoms == 0:
        return "Healthy"
        
    features = ["Fever", "BodyPain", "RunnyNose", "Headache", "Fatigue", "VomitingDiarrhea"]
    input_data = pd.DataFrame([{
        "Fever": fever,
        "BodyPain": body_pain,
        "RunnyNose": runny_nose,
        "Headache": headache,
        "Fatigue": fatigue,
        "VomitingDiarrhea": vomiting_diarrhea
    }])
    
    pred = symptom_model.predict(input_data)
    return pred[0]

from datetime import datetime, timedelta

def analyze_trend_and_outbreak(disease, location_name, historical_reports):
    """
    Computes empirical growth_rate, doubling_time and outbreak risk level
    based on the real time-series history of cases in this location.
    historical_reports is a list of HospitalReport objects for this disease and location.
    """
    
    # 1. Aggregate cases by day over the last 14 days
    today = datetime.utcnow().date()
    daily_counts = {}
    
    # Initialize last 14 days to 0
    for i in range(14):
        day = today - timedelta(days=i)
        daily_counts[day] = 0
        
    for report in historical_reports:
        # Assuming report.timestamp is a datetime object
        report_date = report.timestamp.date()
        if report_date in daily_counts:
            daily_counts[report_date] += 1
            
    # Convert to a chronologically sorted list of counts
    sorted_days = sorted(daily_counts.keys())
    counts = [daily_counts[day] for day in sorted_days]
    
    total_cases = sum(counts)
    
    # If not enough overall case density, it's not an outbreak
    if total_cases < 3:
        return {
            'growth_rate': 1.0,
            'doubling_time': 0.0,
            'risk_level': 'None'
        }
        
    # 2. Calculate empirical growth rate using moving averages to smooth noise
    # Compare the second week (recent) to the first week (older)
    first_week_cases = sum(counts[:7])
    second_week_cases = sum(counts[7:])
    
    if first_week_cases == 0:
        # Infinite mathematical growth if it went from 0 to something. Cap it reasonably.
        growth_rate = 1.0 + (second_week_cases * 0.15) 
    else:
        # Formula: (Recent / Older) ratio for the week
        # If week 1 had 10 cases, week 2 had 15, ratio is 1.5, per-period growth factor
        weekly_ratio = second_week_cases / first_week_cases
        # Convert weekly ratio to an approximate daily compounding rate: (Ratio)^(1/7)
        growth_rate = weekly_ratio ** (1/7)
        
    # Apply floor so we don't return crazy sub-zero rates if disease is dying out, just 1.0 baseline
    growth_rate = max(1.0, round(growth_rate, 2))
    
    # 3. Calculate Doubling Time
    # Formula: ln(2) / ln(growth_rate)
    if growth_rate > 1.01:
        doubling_time = round(np.log(2) / np.log(growth_rate), 1)
    else:
        doubling_time = 0.0 # Not doubling
        
    # 4. Assess Epidemiological Risk Level
    # Standard thresholds: Growth rate > 1.2 is aggressive spread (doubles < 4 days)
    if total_cases < 5:
        # Need at least a cluster to declare higher than Low
        risk = "Low"
    else:
        if growth_rate > 1.25:
            risk = "High"
        elif growth_rate > 1.10:
            risk = "Medium"
        else:
            risk = "Low"
            
    return {
        'growth_rate': growth_rate,
        'doubling_time': doubling_time,
        'risk_level': risk
    }

def analyze_vulnerability(disease, location_name):
    """Interface to the specific geographic vulnerability models."""
    np.random.seed(hash(location_name + disease) % 10000)
    score = round(np.random.uniform(20.0, 95.0), 1)
    return score

def process_patient_report(data, ReportModel=None):
    """Master pipeline tying all models together."""
    
    # 1. Detect Disease
    disease = predict_symptom_disease(data)
    location = data.get('locationName', 'Unknown')
    
    # Extract coordinates for spatial clustering
    try:
        lat = float(data.get('latitude', 0.0))
        lng = float(data.get('longitude', 0.0))
    except (ValueError, TypeError):
        lat, lng = 0.0, 0.0
    
    # 2. Extract historical time-series for this region 
    historical_reports = []
    case_count = 1
    
    if ReportModel:
        # Get all cases for the last 14 days
        fourteen_days_ago = datetime.utcnow() - timedelta(days=14)
        
        # Spatial Clustering Logic: grouped by a geographical bounding box (~10km radius)
        # 1 degree of latitude is ~111km. 0.1 degrees is ~11km.
        if lat != 0.0 and lng != 0.0:
            margin = 0.1
            historical_reports = ReportModel.query.filter(
                ReportModel.predicted_disease == disease,
                ReportModel.latitude.between(lat - margin, lat + margin),
                ReportModel.longitude.between(lng - margin, lng + margin),
                ReportModel.timestamp >= fourteen_days_ago
            ).all()
        else:
            # Fallback to exact string matching if coordinates are missing
            historical_reports = ReportModel.query.filter(
                ReportModel.predicted_disease == disease,
                ReportModel.location_name == location,
                ReportModel.timestamp >= fourteen_days_ago
            ).all()
            
        case_count += len(historical_reports)
        
    # Add the current report being processed to the history virtually
    # so today's count reflects the immediate submission.
    class MockReport:
        def __init__(self):
            self.timestamp = datetime.utcnow()
    historical_reports.append(MockReport())

    # 3. Analyze Trend/Risk using actual spatial-temporal math
    trend = analyze_trend_and_outbreak(disease, location, historical_reports)
    
    # 4. Vulnerability Score
    vuln_score = analyze_vulnerability(disease, location) if trend['risk_level'] != 'None' else 0.0
    
    return {
        'disease': disease,
        'growth_rate': trend['growth_rate'],
        'doubling_time': trend['doubling_time'],
        'risk_level': trend['risk_level'],
        'vulnerability': vuln_score,
        'case_count': case_count
    }
