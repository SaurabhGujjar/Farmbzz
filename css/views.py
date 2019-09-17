from django.db.models import Q
from django import forms
import requests
from django.shortcuts import render
from .forms import CityForm
from django.http import HttpResponse
from django.template import loader
from .models import City, Crops, InsectModel, Tomato_Bacterial_spot, Tomato_Tomato_YellowLeaf_Curl_Virus, Potato_Early_blight
from django.views.generic import ListView
#from .pdd import *
#from .pdd2 import *



def index(request):

    return render(request, 'css/index.html')

def autodiog(request):
    context={}
    y="No Disease"
    path=request.GET.get('d',None)
    if path is not None: 
        if path[-2]=='p' or path[-2]=='P':
            y=diagnose(path)
            if y == 'Tomato_Bacterial_spot':
                a = Tomato_Bacterial_spot.objects.all()
                context['objects']=a
            elif y == 'Potato_Early_blight':
                a = Potato_Early_blight.objects.all()
                context['objects']=a
            elif y == 'Tomato_Tomato_YellowLeaf_Curl_Virus':
                a = Tomato_Tomato_YellowLeaf_Curl_Virus.objects.all()
                context['objects']=a
            else:
                pass
        else:
            y="No Disease"
        findSimilarImage(path)
    context['disease_name']=y
    return render(request, 'css/diagnose.html',context)

class SearchView(ListView):
    template_name = 'css/search_view.html'
    def get_queryset(self, *args , **kwargs):
        q = self.request.GET.get('q', None)
        if q == ' potato' or q == ' Potato' or q =='potato' or q == 'Potato':
            data=Crops.objects.filter(crop_name='PO')
        elif q == ' wheat' or q == ' Wheat' or q == 'wheat' or q == 'Wheat':
            data = Crops.objects.filter(crop_name='WH')
        elif q == ' sugarcane' or q == ' Sugarcane' or q == 'sugarcane' or q == 'Sugarcane':
            data = Crops.objects.filter(crop_name='SU')
        elif q == ' rice' or q == ' Rice' or q == 'rice' or q == 'Rice':
            data = Crops.objects.filter(crop_name='RI')
        elif q == ' cotton' or q == '  Cotton' or q == 'cotton' or q == 'Cotton':
            data = Crops.objects.filter(crop_name='CO')
        elif q == ' groundnut' or q == ' Groundnut' or q == 'groundnut' or q == 'Groundnut':
            data = Crops.objects.filter(crop_name='GR')
        elif q == ' ':
            data = Crops.objects.filter(disease_name='dsfgrewfrgrfegfdsfdgfd')
        else:
            qs = Crops.objects.all()
            if q is not None:
                q = q[1:]
                data = qs.filter(
                    Q(disease_name__icontains=q)
                )
                print(data)
            else:
                data = None
        return data
    def get_context_data(self, *args, **kwargs):
        context = super(SearchView,self).get_context_data(*args,**kwargs)
        query = self.request.GET.get('q')
        context['query'] = query
        return context


class SearchInsectView(ListView):
    template_name = 'css/insect_view.html'
    def get_queryset(self, *args , **kwargs):
        i = self.request.GET.get('i', None)
        print(i)
        if i == ' potato' or i == ' Potato' or i =='potato' or i == 'Potato':
            data = InsectModel.objects.filter(crop_name='PO')
        elif i == ' wheat' or i == ' Wheat' or i == 'wheat' or i == 'Wheat':
            data = InsectModel.objects.filter(crop_name='WH')
        elif i == ' sugarcane' or i == ' Sugarcane' or i == 'sugarcane' or i == 'Sugarcane':
            data = InsectModel.objects.filter(crop_name='SU')
        elif i == ' rice' or i == ' Rice' or i == 'rice' or i == 'Rice':
            data = InsectModel.objects.filter(crop_name='RI')
        elif i == ' cotton' or i == '  Cotton' or i == 'cotton' or i == 'Cotton':
            data = InsectModel.objects.filter(crop_name='CO')
        elif i == ' groundnut' or i == ' Groundnut' or i == 'groundnut' or i == 'Groundnut':
            data = InsectModel.objects.filter(crop_name='GR')
        elif i == ' ':
            data = InsectModel.objects.filter(insect_name='dsfgrewfrgrfegfdsfdgfd')
        else:
            qs = InsectModel.objects.all()
            if i is not None:
                i = i[1:]
                data = qs.filter(
                    Q(insect_name__icontains=i)
                )
                print(data)
            else:
                data = None
        return data
    def get_context_data(self, *args, **kwargs):
        context = super(SearchInsectView,self).get_context_data(*args,**kwargs)
        query = self.request.GET.get('i')
        context['query'] = query
        return context


        
def weather(request):
    url = 'http://api.openweathermap.org/data/2.5/weather?q={}&appid=7e182732b8c643a011c3a25e6818347a&units=metric'
    if request.method == 'POST':
        form = CityForm(request.POST)
        if form.is_valid():
            form.save()

    form = CityForm()
    cities = City.objects.all()
    weather_data = []

    for city in cities:
        data = requests.get(url.format(city))
        r = data.json()
        city_weather = {
             'city': city.name,
             'temperature': r['main']['temp'],
             'description': r['weather'][0]['description'],
             'icon': r['weather'][0]['icon'],
             'windspeed': r['wind']['speed'],
              'winddir': r['wind']['deg'],
              'cloud': r['clouds']['all'],
        }

    weather_data.append(city_weather)

    context = {'weather_data': weather_data, 'form': form}

    return render(request, 'css/weather.html', context)





def explore(request):

    return render(request, 'css/explore.html')




def wheat(request):

    return render(request, 'css/wheat.html')



def rice(request):

    return render(request, 'css/rice.html')


def sugarcane(request):

    return render(request, 'css/sugarcane.html')


def potato(request):

    return render(request, 'css/potato.html')


def groundnut(request):

    return render(request, 'css/groundnut.html')


def cotton(request):

    return render(request, 'css/cotton.html')


def icotton(request):

    return render(request, 'css/cotton_insect.html')


def isugarcane(request):

    return render(request, 'css/sugarcane_insect.html')


def irice(request):

    return render(request, 'css/rice_insect.html')


def ipotato(request):

    return render(request, 'css/potato_insect.html')


def tomato(request):

    return render(request, 'css/tomato_insect.html')


def mango(request):

    return render(request, 'css/mango_insect.html')









