from django.db import models

class Product(models.Model):
    product_id = models.CharField(max_length=100, unique=True)
    product_name = models.CharField(max_length=255)
    category = models.CharField(max_length=100)
    discounted_price = models.FloatField()
    actual_price = models.FloatField()
    discount_percentage = models.FloatField(null=True, blank=True)
    rating = models.FloatField(null=True, blank=True)
    rating_count = models.IntegerField(default=0)
    about_product = models.TextField(blank=True, null=True)
    stock = models.IntegerField(default=0)
    creation_date = models.DateField(auto_now_add=True)
    last_updated_date = models.DateField(auto_now=True)
    status = models.CharField(max_length=50, default="active")

    def __str__(self):
        return f"{self.product_name} ({self.category})"


class Client(models.Model):
    customer_id = models.CharField(max_length=100, unique=True)
    name = models.CharField(max_length=255)
    gender = models.CharField(max_length=20, choices=[('Male', 'Male'), ('Female', 'Female')], blank=True, null=True)
    region = models.CharField(max_length=50, blank=True, null=True)
    age = models.PositiveIntegerField(blank=True, null=True)
    source = models.CharField(max_length=50, blank=True, null=True)
    email = models.EmailField(unique=False, blank=True, null=True)
    phone = models.CharField(max_length=20, blank=True, null=True)
    income_segment = models.CharField(max_length=50, blank=True, null=True)
    loyalty_status = models.CharField(max_length=50, blank=True, null=True)
    is_active = models.BooleanField(default=True)
    registration_date = models.DateField(null=True, blank=True)

    total_spent = models.FloatField(blank=True, null=True, default=0)
    avg_order_value = models.FloatField(blank=True, null=True, default=0)
    total_orders = models.IntegerField(blank=True, null=True, default=0)
    total_items = models.IntegerField(blank=True, null=True, default=0)
    avg_discount = models.FloatField(blank=True, null=True, default=0)
    avg_review_rating = models.FloatField(blank=True, null=True, default=0)
    avg_seller_rating = models.FloatField(blank=True, null=True, default=0)
    avg_delivery_days = models.FloatField(blank=True, null=True, default=0)
    total_returns = models.IntegerField(blank=True, null=True, default=0)
    return_ratio = models.FloatField(blank=True, null=True, default=0)
    total_previous_returns = models.IntegerField(blank=True, null=True, default=0)
    is_prime_member = models.BooleanField(default=False)
    customer_tenure_days = models.FloatField(blank=True, null=True, default=0)
    last_order_date = models.DateField(blank=True, null=True)
    first_order_date = models.DateField(blank=True, null=True)
    recency_days = models.FloatField(blank=True, null=True, default=0)
    tenure_days = models.FloatField(blank=True, null=True, default=0)
    frequency = models.FloatField(blank=True, null=True, default=0)

    predicted_clv = models.FloatField(default=0, null=True, blank=True)

    def __str__(self):
        return f"{self.name} ({self.customer_id})"


class ClientNewsletter(models.Model):
    """Clients subscribed to newsletters and promotional emails."""
    name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=20, blank=True, null=True)
    region = models.CharField(max_length=100, blank=True, null=True)
    date_of_birth = models.DateField(blank=True, null=True)
    is_active = models.BooleanField(default=True)
    subscribed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.email})"

class MessageTemplate(models.Model):
    EVENT_CHOICES = [
        ("birthday", "Birthday"),
        ("risk", "At-Risk Client"),
        ("sale", "Sale Promotion"),
        ("custom", "Custom Campaign"),
    ]
    CHANNEL_CHOICES = [
        ("email", "Email"),
        ("whatsapp", "WhatsApp"),
        ("both", "Both"),
    ]

    event_type = models.CharField(max_length=50, choices=EVENT_CHOICES)
    subject = models.CharField(max_length=255)
    message = models.TextField(help_text="Use placeholders like {name}, {region}, {discount}")
    channel = models.CharField(max_length=20, choices=CHANNEL_CHOICES, default="email")
    active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.get_event_type_display()} ({self.channel})"

