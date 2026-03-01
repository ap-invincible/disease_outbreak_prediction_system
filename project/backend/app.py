from flask import Flask
from flask_cors import CORS
from models import db
import os

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Configure SQLite database
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outbreak.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = 'your-secret-key-for-jwt'
    
    db.init_app(app)
    
    with app.app_context():
        db.create_all()
        
    return app

if __name__ == '__main__':
    app = create_app()
    from auth import auth_bp
    from routes.hospital import hospital_bp
    from routes.dashboard import dashboard_bp
    
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(hospital_bp, url_prefix='/api/hospital')
    app.register_blueprint(dashboard_bp, url_prefix='/api/dashboard')
    
    app.run(debug=True, port=5000)
