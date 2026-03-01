from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True) # Making nullable=True for backward compatibility temporarily
    password = db.Column(db.String(120), nullable=False) # In production use hashed passwords

class EmailVerification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    pin = db.Column(db.String(6), nullable=False)
    verified = db.Column(db.Boolean, default=False)
    expires_at = db.Column(db.DateTime, nullable=False)

class HospitalReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Location
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    location_name = db.Column(db.String(200))
    
    # Symptoms
    fever = db.Column(db.Float, nullable=False)
    body_pain = db.Column(db.Integer, nullable=False)
    runny_nose = db.Column(db.Integer, nullable=False)
    headache = db.Column(db.Integer, nullable=False)
    fatigue = db.Column(db.Integer, nullable=False)
    vomiting_diarrhea = db.Column(db.Integer, nullable=False)
    
    # Predictions (populated by pipeline)
    predicted_disease = db.Column(db.String(50))
    outbreak_risk = db.Column(db.String(20)) # High, Medium, Low
    vulnerability_score = db.Column(db.Float)
    growth_rate = db.Column(db.Float)
    doubling_time = db.Column(db.Float)
