from flask import Blueprint, jsonify
from models import db, HospitalReport
from auth import token_required

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/data', methods=['GET'])
@token_required
def get_dashboard_data(current_user):
    # Retrieve all reports
    reports = HospitalReport.query.all()
    
    active_outbreaks = [r for r in reports if r.outbreak_risk == 'High']
    
    # Format for the frontend UI (cards and map markers)
    formatted_reports = []
    for r in reports:
        if r.predicted_disease and r.predicted_disease.lower() == 'healthy':
            continue
        if r.outbreak_risk and r.outbreak_risk.lower() == 'none':
            continue
        formatted_reports.append({
            'id': r.id,
            'timestamp': r.timestamp.isoformat(),
            'lat': r.latitude,
            'lng': r.longitude,
            'locationName': r.location_name,
            'disease': r.predicted_disease,
            'riskLevel': r.outbreak_risk,
            'vulnerabilityScore': r.vulnerability_score,
            'growthRate': r.growth_rate,
            'doublingTime': r.doubling_time
        })
        
    return jsonify({
        'total_cases_reported': len(reports),
        'active_high_risk_outbreaks': len(active_outbreaks),
        'reports': formatted_reports
    })
