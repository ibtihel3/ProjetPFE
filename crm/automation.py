from django.core.mail import send_mail
from django.conf import settings
from django.utils import timezone
from .models import MessageTemplate, ClientNewsletter
import requests


def auto_run_newsletters():
    """
    Automatically send personalized EMAIL messages
    based on active templates and event type.
    """
    today = timezone.now().date()
    templates = MessageTemplate.objects.filter(active=True)

    print("🚀 Starting automated email newsletters...")

    for tpl in templates:
        # --- 1️⃣ Select target clients ---
        if tpl.event_type == "birthday":
            clients = ClientNewsletter.objects.filter(
                date_of_birth__month=today.month,
                date_of_birth__day=today.day,
                is_active=True
            )

        elif tpl.event_type == "risk":
            try:
                res = requests.get("http://127.0.0.1:8000/api/notify/at_risk_clients", timeout=10)
                data = res.json()
                notifications = data.get("notifications", [])
                # Convert each dict into a simple object for compatibility
                clients = [type("obj", (object,), n) for n in notifications]
            except Exception as e:
                print("⚠️ FastAPI error:", e)
                clients = []
        else:
            continue  # skip “custom” templates

        # --- 2️⃣ Send EMAIL messages to each client ---
        for c in clients:
            try:
                name = getattr(c, "name", "") or getattr(c, "Name", "")
                region = getattr(c, "region", "") or getattr(c, "Region", "")
                email = getattr(c, "email", "") or getattr(c, "Email", "")

                if not email:
                    continue  # skip clients with no email

                # Replace placeholders in the message
                message = tpl.message.format(name=name, region=region, discount="20%")
                subject = tpl.subject.format(name=name, region=region)

                send_mail(
                    subject,
                    message,
                    settings.DEFAULT_FROM_EMAIL,
                    [email],
                    fail_silently=False,
                )
                print(f"📧 Sent email to {email}")

            except Exception as e:
                print(f"⚠️ Error sending to {getattr(c, 'email', 'unknown')}: {e}")

    print("✅ Automated emails sent successfully.")
