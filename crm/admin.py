from django.contrib import admin
from .models import Product
from .models import Client

@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = (
        "product_name",
        "category",
        "actual_price",
        "discounted_price",
        "discount_percentage",
        "stock",
        "rating",
    )
    search_fields = ("product_name", "category")


@admin.register(Client)
class ClientAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "gender",
        "region",
        "age",
        "registration_date",
        "email",
        "phone",
    )
    search_fields = ("name", "customer_id")
