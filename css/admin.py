from django.contrib import admin
from .models import City
from .models import Crops
from .models import InsectModel
from .models import Tomato_Bacterial_spot
from .models import Tomato_Tomato_YellowLeaf_Curl_Virus
from .models import Potato_Early_blight
admin.site.register(Crops)
admin.site.register(City)
admin.site.register(InsectModel)
admin.site.register(Tomato_Bacterial_spot)
admin.site.register(Tomato_Tomato_YellowLeaf_Curl_Virus)
admin.site.register(Potato_Early_blight)



