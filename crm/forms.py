from django import forms
from .models import Product
from .models import Client

class ProductForm(forms.ModelForm):
    class Meta:
        model = Product
        fields = [
            "product_id",
            "product_name",
            "category",
            "discounted_price",
            "actual_price",
            "discount_percentage",
            "rating",
            "rating_count",
            "about_product",
            "stock",
            "status",
        ]

class ClientForm(forms.ModelForm):
    class Meta:
        model = Client
        fields = [
            "customer_id",
            "name",
            "gender",
            "region",
            "age",
            "source",
            "email",
            "phone",
            "is_active",
            "income_segment",
            "registration_date",
        ]
