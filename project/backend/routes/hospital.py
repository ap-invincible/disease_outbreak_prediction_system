from flask import Blueprint, request, jsonify
from models import db, HospitalReport
from auth import token_required
from services.prediction_pipeline import process_patient_report

hospital_bp = Blueprint('hospital', __name__)

@hospital_bp.route('/submit', methods=['POST'])
@token_required
def submit_report(current_user):
    data = request.get_json()
    
    # 1. Run predictions pipeline based on symptoms and location
    # Passing the HospitalReport model allows the pipeline to count historical cases first
    pipeline_results = process_patient_report(data, ReportModel=HospitalReport)
    
    # 2. Save report with predictions attached
    new_report = HospitalReport(
        user_id=current_user.id,
        latitude=data.get('latitude', 0.0),
        longitude=data.get('longitude', 0.0),
        location_name=data.get('locationName', 'Unknown'),
        
        fever=data.get('fever', 98.6),
        body_pain=int(data.get('body_pain', 0)),
        runny_nose=int(data.get('runny_nose', 0)),
        headache=int(data.get('headache', 0)),
        fatigue=int(data.get('fatigue', 0)),
        vomiting_diarrhea=int(data.get('vomiting_diarrhea', 0)),
        
        predicted_disease=pipeline_results['disease'],
        outbreak_risk=pipeline_results['risk_level'],
        vulnerability_score=pipeline_results['vulnerability'],
        growth_rate=pipeline_results['growth_rate'],
        doubling_time=pipeline_results['doubling_time']
    )
    
    db.session.add(new_report)
    db.session.commit()
    
    return jsonify({
        'message': 'Report submitted successfully',
        'pipeline_results': pipeline_results
    })
