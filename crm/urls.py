from django.urls import path
from . import views
from .views import predict_for_client
from .views import predict_all_clv
from django.contrib import admin
from django.urls import path, include
urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('', views.home, name='home'),
    path('crm/', views.product_list, name='product_list'),
    path('crm/add/', views.product_create, name='product_add'),
    path('crm/edit/<int:pk>/', views.product_edit, name='product_edit'),
    path('crm/delete/<int:pk>/', views.product_delete, name='product_delete'),
    path('crm/import/', views.import_csv, name='import_csv'),
    path('crm/predict_discount/<int:pk>/', views.predict_for_product, name='predict_for_product'),
    path("chatbot_page/", views.chatbot_page, name="chatbot_page"),  
    path("chatbot/", views.chatbot, name="chatbot"),               
    path("crm/similar/<int:pk>/", views.similar_products, name="similar_products"),
    path("reviews_dashboard/", views.reviews_dashboard, name="reviews_dashboard"),

    path("reviews_export/", views.reviews_export, name="reviews_export"),

    path('crm/client', views.client_list, name='client_list'),
    path('crm/add_client/', views.client_create, name='client_add'),
    path('crm/edit_client/<int:pk>/', views.client_edit, name='client_edit'),
    path('crm/delete_client/<int:pk>/', views.client_delete, name='client_delete'),
    path('crm/predict_clv/<int:pk>/', views.predict_for_client, name='predict_for_client'),
    path('crm/predict_all_clv/', views.predict_all_clv, name='predict_all_clv'),

    path("crm/similar_from_image/", views.similar_from_image, name="similar_from_image"),
    path("newsletter/", views.send_newsletter, name="send_newsletter"),
    path("ajax/search-products/", views.ajax_search_products, name="ajax_search_products"),
    path("notify/at_risk_clients/", views.notifications, name="notifications"),

    path("templates/", views.manage_templates, name="manage_templates"),
    path("templates/view/<int:pk>/", views.view_template, name="view_template"),
    path("templates/edit/<int:pk>/", views.edit_template, name="edit_template"),
    path("templates/delete/<int:pk>/", views.delete_template, name="delete_template"),
    path("templates/toggle/", views.toggle_template_status, name="toggle_template_status"),
]
