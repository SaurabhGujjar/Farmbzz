from django.conf.urls import url
from django.urls import path, include
from mailer import views
from .models import Comment

app_name='mailer'
urlpatterns=  [
    path('mailer/', views.form_name_view, name='mailer'),
    path('', views.result, name='result'),
    path('mailer/add/', views.MailerCreate.as_view(model=Comment, success_url=''), name='add')

]
