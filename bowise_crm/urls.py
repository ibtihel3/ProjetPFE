from django.urls import path, include
from django.contrib import admin

urlpatterns = [
    path('', include('crm.urls')),
    path('crm/admin/', admin.site.urls),
    path('django_plotly_dash/', include('django_plotly_dash.urls')),
]
