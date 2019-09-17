from django.conf import settings
from django.views.generic.edit import CreateView
from django.shortcuts import render
from django.core.mail import send_mail
from . import forms
from .models import formdata
from .models import Comment
from django.urls import reverse
from django.http import HttpResponseRedirect





def form_name_view(request):
    form = forms.FormName()
    if request.method == 'POST':
        form = forms.FormName(request.POST)
        print(form)
        if form.is_valid():
            uname = form.cleaned_data['name']
            subject1 = form.cleaned_data['subject']
            comp = form.cleaned_data['complaint']
            phone = form.cleaned_data['phone']
            address = form.cleaned_data['address']
            recover = form.cleaned_data['city']


            list = {'hisar':'sauravpanwar45@gmail.com', 'saharanpur':'sahilnawaab@gmail.com', 'jind':'lihas0709@gmail.com'}
            for key, values in list.items():
                if key == recover:
                    sentid = values
            comp = form.cleaned_data['complaint']

            text = str(comp) + "\n This email is from " + str(uname) +"\n Mobile No- " + str(phone) + "\n Address-" + str(address)
            print(text)
            send_mail(str(subject1), text, 'sahilnawaab@gmail.com', [str(sentid)])
            obj = formdata(username=uname, subject=subject1, region=recover, text=comp)
            obj.save()
            print('mail sent')
            return render(request, 'mailer/result.html')
    return render(request, 'mailer/formpage1.html', {'form': form})


def result(request):
    return render(request, 'mailer/result.html')




class MailerCreate(CreateView):
    model = Comment
    fields = ['name', 'comment']
    url='mailer/'





