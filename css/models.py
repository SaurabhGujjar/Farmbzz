from django.db import models



class signup(models.Model):
    first_name = models.CharField(max_length=200)
    last_name = models.CharField(max_length=200)
    state = models.CharField(max_length=200)
    city = models.CharField(max_length=200)
    phone = models.CharField(max_length=10)
    password = models.CharField(max_length=200)
    profile_image = models.ImageField(upload_to='photos', blank=True, null=True)
    def __str__(self):
        return self.first_name + '-' + self.last_name

class City(models.Model):
    name = models.CharField(max_length=50)
    def __str__(self):
        return self.name
    class Meta:
        verbose_name_plural = 'cities'

class Crops(models.Model):

    WHEAT = 'WH'
    RICE = 'RI'
    POTATO = 'PO'
    COTTON = 'CO'
    SUGARCANE = 'SU'
    GROUNDNUT = 'GR'
    MAIZE = 'MA'
    JUTE = 'JU'
    TEA = 'TE'
    COFFEE = 'COF'
    PULSE = 'PU'
    MILLET = 'MI'
    MUSTARD = 'MU'
    TOMATO = 'TO'
    CROP_CHOICES = (
        (WHEAT, 'Wheat'),
        (RICE, 'Rice'),
        (COTTON, 'Cotton'),
        (SUGARCANE, 'Sugarcane'),
        (GROUNDNUT, 'Groundnut'),
        (POTATO, 'Potato'),
        (MUSTARD, 'Mustard'),
        (MILLET, 'Millet'),
        (PULSE, 'Pulse'),
        (COFFEE, 'Coffee'),
        (TEA, 'Tea'),
        (JUTE, 'Jute'),
        (MAIZE, 'Maize'),
        (TOMATO, 'Tomato'),
    )
    crop_name = models.CharField(
        max_length=3,
        choices=CROP_CHOICES,
        default=WHEAT,
    )


    disease_name = models.TextField(max_length=300)
    image1 = models.ImageField(upload_to='photos', blank=True, null=True)
    image2 = models.ImageField(upload_to='photos', blank=True, null=True)
    image3 = models.ImageField(upload_to='photos', blank=True, null=True)
    symptom = models.TextField(max_length=9000)
    chem_sol = models.TextField(max_length=9000)
    cult_sol = models.TextField(max_length=9000)


    def __str__(self):
        return self.disease_name














class InsectModel(models.Model):
    WHEAT = 'WH'
    RICE = 'RI'
    POTATO = 'PO'
    COTTON = 'CO'
    SUGARCANE = 'SU'
    GROUNDNUT = 'GR'
    JUTE = 'JU'
    TEA = 'TE'
    COFFEE = 'COF'
    PULSE = 'PU'
    MILLET = 'MI'
    MUSTARD = 'MU'
    TOMATO = 'TO'
    MANGO = 'MA'
    CROP_CHOICES = (
        (WHEAT, 'Wheat'),
        (RICE, 'Rice'),
        (COTTON, 'Cotton'),
        (SUGARCANE, 'Sugarcane'),
        (GROUNDNUT, 'Groundnut'),
        (POTATO, 'Potato'),
        (MUSTARD, 'Mustard'),
        (MILLET, 'Millet'),
        (PULSE, 'Pulse'),
        (COFFEE, 'Coffee'),
        (TEA, 'Tea'),
        (JUTE, 'Jute'),
        (MANGO, 'Mango'),
        (TOMATO, 'Tomato'),
    )
    crop_name = models.CharField(
        max_length=3,
        choices=CROP_CHOICES,
        default=WHEAT,
    )

    insect_name = models.TextField(max_length=300)
    img1 = models.ImageField(upload_to='photos', blank=True, null=False)
    img2 = models.ImageField(upload_to='photos', blank=True, null=False)
    img3 = models.ImageField(upload_to='photos', blank=True, null=False)
    sympt = models.TextField(max_length=9000, blank=True, null=False)
    trigger = models.TextField(max_length=9000, blank=True, null=False)
    biological_control = models.TextField(max_length=9000, blank=True, null=False)
    chem_control = models.TextField(max_length=9000, blank=True, null=False)
    preventive_measures = models.TextField(max_length=9000, blank=True, null=False)
    
    def __str__(self):
        return self.insect_name


class Tomato_Bacterial_spot(models.Model):
    symptm = models.CharField(max_length=9000)
    mechanical = models.CharField(max_length=9000)
    chemical = models.CharField(max_length=9000)
    physical = models.CharField(max_length=9000)
    pesticide = models.CharField(max_length=9000)
    

class Potato_Early_blight(models.Model):
    symptm = models.CharField(max_length=9000)
    mechanical = models.CharField(max_length=9000)
    chemical = models.CharField(max_length=9000)
    physical = models.CharField(max_length=9000)
    pesticide = models.CharField(max_length=9000)
    


class Tomato_Tomato_YellowLeaf_Curl_Virus(models.Model):
    symptm = models.CharField(max_length=9000)
    mechanical = models.CharField(max_length=9000)
    chemical = models.CharField(max_length=9000)
    physical = models.CharField(max_length=9000)
    pesticide = models.CharField(max_length=9000)
    

