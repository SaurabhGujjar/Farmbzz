import requests
from django import forms
from django.forms import ModelForm
from .models import City


class CityForm(ModelForm):
    name = forms.CharField()
    class Meta:
        model = City
        fields = ['name']
    def clean_name(self, *args , **kwargs):
        name = self.cleaned_data.get("name")
        print(name)
        url = 'http://api.openweathermap.org/data/2.5/weather?q={}&appid=7e182732b8c643a011c3a25e6818347a&units=metric'
        data = requests.get(url.format(name))
        r = data.json()
        if r['cod'] == '404':
            raise forms.ValidationError("Invalid city entered")
        return name

