from django.db import models
from django.utils import timezone
from django.urls import reverse



class formdata(models.Model):

    def __str__(self):
        return self.username

    username = models.CharField(max_length=200)
    subject = models.CharField(max_length=500, blank=False)
    region = models.CharField(max_length=500, blank=False)
    text = models.CharField(max_length=9000, null=False, blank=False)



class Comment(models.Model):


    def __str__(self):
        return self.name

        
    name = models.CharField(max_length=200)
    date_time = models.DateTimeField(default=timezone.now())
    comment = models.CharField(max_length=9000, null=False, blank=False)
    def get_absoulte_url(self):
        return reverse('mailer',kwargs={})

