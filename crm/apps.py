from django.apps import AppConfig
import threading

class ProductsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "crm"
    def ready(self):
        # Import Dash app once Django starts
        try:
            import dash_app
            print("✅ Dash app imported successfully from crm.apps.py")
        except Exception as e:
            print(f"❌ Error importing dash_app: {e}")


        from crm.automation import auto_run_newsletters

        def run_on_start():
            print("⚙️  Running initial automation check...")
            try:
                auto_run_newsletters()
            except Exception as e:
                print("⚠️ Automation failed at startup:", e)

        threading.Thread(target=run_on_start).start()
