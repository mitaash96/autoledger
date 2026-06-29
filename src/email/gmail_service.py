import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build

from .. import config as cfg
from ..logger import get_logger

logger = get_logger(__name__, cfg.logging["log_file"], cfg.logging["level"])

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
if "SSL_CERT_FILE" not in os.environ:
    os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"


def get_gmail_service() -> Resource:

    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("gcs.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


