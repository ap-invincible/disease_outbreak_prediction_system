import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to get credentials from environment variables, fallback to print-only mode for dev
SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', 587))
SMTP_USER = os.environ.get('SMTP_USER', '')
SMTP_PASS = os.environ.get('SMTP_PASS', '')

def send_email(to_email, subject, body):
    if not SMTP_USER or not SMTP_PASS:
        # Dev mode: Print to console instead of actually sending
        logger.info(f"==== MOCK EMAIL TO {to_email} ====")
        logger.info(f"Subject: {subject}")
        logger.info(f"Body: {body}")
        logger.info("====================================")
        return True
        
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_USER
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
            
        logger.info(f"Successfully sent email to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {e}")
        return False

def send_verification_pin(to_email, pin):
    subject = "invncble - Email Verification"
    body = f"Your verification code is: {pin}\n\nThis code will expire in 10 minutes."
    return send_email(to_email, subject, body)

def send_login_notification(to_email, username, time_str, ip_addr=None):
    subject = "invncble - Security Alert"
    body = f"Hello {username},\n\nA new sign-in was detected for your account at {time_str}."
    if ip_addr:
        body += f"\nIP Address: {ip_addr}"
    body += "\n\nIf this was not you, please contact the administrator immediately."
    return send_email(to_email, subject, body)
