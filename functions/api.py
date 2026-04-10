import serverless_wsgi
from app import app # This links to your app.py

def handler(event, context):
    return serverless_wsgi.handle_request(app, event, context)