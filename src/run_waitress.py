from waitress import serve
from app import app
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

serve(app, host='0.0.0.0', port=5000)