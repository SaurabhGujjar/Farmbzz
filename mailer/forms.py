from django import forms

class FormName(forms.Form):
        name = forms.CharField()
        subject = forms.CharField()
        phone = forms.CharField()
        city = forms.CharField()
        address = forms.CharField()
        complaint = forms.CharField(widget=forms.Textarea)

