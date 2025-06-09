import os
import sqlite3
import random
import string
import time
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory, session
from werkzeug.utils import secure_filename
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from dotenv import load_dotenv
from utils.checker import generate_pdf_report, tokenize_code, get_graphcodebert_embedding, compute_similarity_pair
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tempfile

app = Flask(__name__)
app.secret_key = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
load_dotenv()

# Configuration
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'codecompare_uploads')
REPORT_FOLDER = os.path.join(tempfile.gettempdir(), 'codecompare_reports')
ALLOWED_EXTENSIONS = {'py', 'java', 'cpp', 'c', 'js', 'php'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# SQLite Database Setup
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY, email TEXT UNIQUE, name TEXT, otp TEXT, otp_expiry INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS reports 
                 (id INTEGER PRIMARY KEY, user_id INTEGER, filename TEXT, created_at INTEGER)''')
    conn.commit()
    conn.close()

init_db()

# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

def send_otp_email(email, otp):
    sendgrid_api_key = os.getenv('SENDGRID_API_KEY')
    if not sendgrid_api_key:
        print(f"OTP for {email}: {otp} (No SendGrid API key provided, using console fallback)")
        flash(f"OTP sent to console: {otp}", 'info')
        return True
    message = Mail(
        from_email='no-reply@codecompare.com',
        to_emails=email,
        subject='Your CodeCompare OTP',
        html_content=f'<strong>Your OTP is: {otp}</strong> (Valid for 5 minutes)')
    try:
        sg = SendGridAPIClient(sendgrid_api_key)
        response = sg.send(message)
        if response.status_code == 202:
            print(f"OTP sent successfully to {email}")
            return True
        else:
            print(f"Failed to send OTP: Status {response.status_code}, Body: {response.body}")
            flash(f"Failed to send OTP (Status {response.status_code}). Check console for OTP: {otp}", 'error')
            return False
    except Exception as e:
        print(f"Error sending OTP to {email}: {str(e)}")
        flash(f"Error sending OTP. Check console for OTP: {otp}", 'error')
        return False

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email'].strip()
        name = request.form['name'].strip()
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        try:
            c.execute('INSERT INTO users (email, name) VALUES (?, ?)', (email, name))
            conn.commit()
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already registered.', 'error')
        finally:
            conn.close()
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].strip()
        otp = request.form.get('otp', '').strip()
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        try:
            if otp:  # Verify OTP
                c.execute('SELECT id, otp, otp_expiry FROM users WHERE email = ?', (email,))
                user = c.fetchone()
                if user and user[1] == otp and user[2] > int(time.time()):
                    session['user_id'] = user[0]
                    c.execute('UPDATE users SET otp = NULL, otp_expiry = NULL WHERE email = ?', (email,))
                    conn.commit()
                    flash('Login successful!', 'success')
                    return redirect(url_for('dashboard'))
                else:
                    flash('Invalid or expired OTP.', 'error')
                    c.execute('SELECT otp FROM users WHERE email = ?', (email,))
                    user = c.fetchone()
                    if user and user[0]:
                        return render_template('login.html', email=email, show_otp=True)
                    return redirect(url_for('login'))
            else:  # Generate and send OTP
                c.execute('SELECT id FROM users WHERE email = ?', (email,))
                user = c.fetchone()
                if user:
                    otp = generate_otp()
                    expiry = int(time.time()) + 300
                    c.execute('UPDATE users SET otp = ?, otp_expiry = ? WHERE email = ?', (otp, expiry, email))
                    conn.commit()
                    if send_otp_email(email, otp):
                        flash('OTP sent to your email or console.', 'success')
                    return render_template('login.html', email=email, show_otp=True)
                else:
                    flash('Email not found.', 'error')
        finally:
            conn.close()
    return render_template('login.html', show_otp=False)

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    files = []
    results = []
    latest_report = None  # Track the latest report from this session
    if request.method == 'POST':
        if 'files' not in request.files:
            flash('No files uploaded.', 'error')
            return redirect(request.url)
        
        uploaded_files = request.files.getlist('files')
        file_type = request.form.get('file_type')
        if file_type not in ALLOWED_EXTENSIONS:
            flash('Invalid file type.', 'error')
            return redirect(request.url)
        
        # Save uploaded files
        for file in uploaded_files:
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                files.append({'filename': filename, 'content': content, 'ext': file_type})
            else:
                flash(f'Invalid file: {file.filename if file else "None"}', 'error')
        
        if len(files) < 2:
            flash('At least two files are required for comparison.', 'error')
            return redirect(request.url)
        
        # Compute similarity
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
        model = AutoModel.from_pretrained("microsoft/graphcodebert-base").to(device)
        embeddings = [get_graphcodebert_embedding(file['content'], tokenizer, model, device) for file in files]
        
        scores = []
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                result = compute_similarity_pair(i, j, files, embeddings, file_type)
                scores.append(result)
        
        # Generate PDF
        report_filename = f"report_{int(time.time())}.pdf"
        report_path = os.path.join(app.config['REPORT_FOLDER'], report_filename)
        generate_pdf_report(scores, report_path, len(files))
        
        # Save report to database
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        current_time = int(time.time())
        c.execute('INSERT INTO reports (user_id, filename, created_at) VALUES (?, ?, ?)',
                  (session['user_id'], report_filename, current_time))
        conn.commit()
        conn.close()
        
        results = sorted(scores, key=lambda x: x['score'], reverse=True)
        latest_report = (report_filename, current_time)  # Store latest report
    
    # Get recent reports, filter existing files
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('SELECT filename, created_at FROM reports WHERE user_id = ? ORDER BY created_at DESC LIMIT 5',
              (session['user_id'],))
    reports = c.fetchall()
    conn.close()
    
    # Filter reports to only include existing files, exclude latest if from this session
    valid_reports = []
    for report in reports:
        report_path = os.path.join(app.config['REPORT_FOLDER'], report[0])
        if os.path.exists(report_path):
            if not latest_report or report[0] != latest_report[0]:  # Exclude latest report
                valid_reports.append(report)
            elif not latest_report:  # If no POST, include all valid reports
                valid_reports.append(report)
        else:
            flash(f'Report {report[0]} not found on server.', 'error')
    
    return render_template('dashboard.html', results=results, latest_report=latest_report, reports=valid_reports)

@app.route('/download/<filename>')
def download_report(filename):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return send_from_directory(app.config['REPORT_FOLDER'], filename, as_attachment=True)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Logged out successfully.', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)