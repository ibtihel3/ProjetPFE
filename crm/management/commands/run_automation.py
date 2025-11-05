from django.core.management.base import BaseCommand
from crm.automation import auto_run_newsletters


class Command(BaseCommand):
    help = "Runs automatic email newsletters (birthday, risk, sale, etc.)"

    def handle(self, *args, **kwargs):
        self.stdout.write(self.style.NOTICE("🚀 Starting automated email newsletters..."))
        try:
            auto_run_newsletters()
            self.stdout.write(self.style.SUCCESS("✅ Automated emails sent successfully."))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"⚠️ Error during automation: {e}"))
