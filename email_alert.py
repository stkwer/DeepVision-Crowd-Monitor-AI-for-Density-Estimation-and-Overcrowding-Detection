# email_alert.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st
import datetime

def send_email_alert(count, extra_text=None):
    """
    Send an alert email using SMTP credentials stored in st.secrets.
    Returns True on success, False on failure.
    """
    try:
        sender = st.secrets["EMAIL_SENDER"]
        password = st.secrets["EMAIL_PASSWORD"]
        receivers = st.secrets["EMAIL_RECEIVER"]
        smtp_server = st.secrets["SMTP_SERVER"]
        smtp_port = int(st.secrets["SMTP_PORT"])

        # allow multiple recipients comma separated
        if isinstance(receivers, str):
            receiver_list = [r.strip() for r in receivers.split(",")]
        else:
            receiver_list = receivers

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        subject = f"[CrowdVision] Overcrowding Alert at {timestamp}"
        body = f"""ALERT: Overcrowding detected!

Time: {timestamp}
Estimated count: {count:.1f}

{extra_text or ""}

-- CrowdVision AI
"""

        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = ", ".join(receiver_list)
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, receiver_list, msg.as_string())

        print(" Email alert sent.")
        return True
    except Exception as e:
        print(" Failed to send email:", e)
        return False
