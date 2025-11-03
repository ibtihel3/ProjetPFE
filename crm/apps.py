from django.apps import AppConfig


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
