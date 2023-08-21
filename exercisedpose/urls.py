from django.urls import path,include
from django.contrib.auth import views as auth_views
from . import views
from exercisedpose import modules
urlpatterns=[
    path("",views.index,name="index"),
    path("process_image",views.main,name="process_img")
]