from flask import Blueprint, request, jsonify
from models import db, User, EmailVerification
from services.mail_service import send_verification_pin, send_login_notification
import jwt
from datetime import datetime, timedelta
import random
import os

auth_bp = Blueprint('auth', __name__)
SECRET_KEY = 'your-secret-key-for-jwt'

@auth_bp.route('/request-pin', methods=['POST'])
def request_pin():
    data = request.get_json()
    email = data.get('email')
    
    if not email:
        return jsonify({'error': 'Email is required'}), 400
        
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email is already registered'}), 400
        
    pin = str(random.randint(100000, 999999))
    expires = datetime.utcnow() + timedelta(minutes=10)
    
    verification = EmailVerification.query.filter_by(email=email).first()
    if verification:
        verification.pin = pin
        verification.expires_at = expires
        verification.verified = False
    else:
        verification = EmailVerification(email=email, pin=pin, expires_at=expires)
        db.session.add(verification)
        
    db.session.commit()
    
    send_verification_pin(email, pin)
    return jsonify({'message': 'Verification PIN sent to email'})

@auth_bp.route('/verify-pin', methods=['POST'])
def verify_pin():
    data = request.get_json()
    email = data.get('email')
    pin = data.get('pin')
    
    if not email or not pin:
        return jsonify({'error': 'Email and PIN are required'}), 400
        
    verification = EmailVerification.query.filter_by(email=email).first()
    if not verification:
        return jsonify({'error': 'No pending verification for this email'}), 400
        
    if verification.expires_at < datetime.utcnow():
        return jsonify({'error': 'Verification PIN has expired'}), 400
        
    if verification.pin != pin:
        return jsonify({'error': 'Invalid PIN'}), 400
        
    verification.verified = True
    db.session.commit()
    
    return jsonify({'message': 'Email verified successfully'})

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')
    
    if not username or not password or not email:
        return jsonify({'error': 'Username, email, and password are required'}), 400
    
    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already exists'}), 400
        
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 400
        
    verification = EmailVerification.query.filter_by(email=email, verified=True).first()
    if not verification:
        return jsonify({'error': 'Email not verified. Please verify your email first.'}), 400
        
    new_user = User(username=username, email=email, password=password) # Simple plaintext for demo purposes
    db.session.add(new_user)
    db.session.delete(verification)
    db.session.commit()
    
    return jsonify({'message': 'User registered successfully'})

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data.get('username'), password=data.get('password')).first()
    
    if not user:
        return jsonify({'error': 'Invalid credentials'}), 401
        
    token = jwt.encode({
        'user_id': user.id,
        'exp': datetime.utcnow() + timedelta(hours=24)
    }, SECRET_KEY, algorithm='HS256')
    
    # Send login notification
    if getattr(user, 'email', None):
        time_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        ip_addr = request.remote_addr
        send_login_notification(user.email, user.username, time_str, ip_addr)
    
    return jsonify({'token': token, 'username': user.username, 'email': getattr(user, 'email', None), 'id': user.id})

def token_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        try:
            token = token.split(" ")[1] # Bearer token
            data = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
        except Exception as e:
            return jsonify({'message': 'Token is invalid'}), 401
        return f(current_user, *args, **kwargs)
    return decorated
