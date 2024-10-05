from django.shortcuts import render
from django.templatetags.static import static 
import pickle
import os
from django.conf import settings
from django.http import JsonResponse
import numpy as np
from sklearn.tree import DecisionTreeClassifier
def demo(request):
   return render(request, 'demo.html')
def load_DecisionTreeClassifier_model():
    # Define the path to the Ridge model pickle file
    pickle_file_path = os.path.join(settings.MEDIA_ROOT, 'iris_DecisionTreeClassifier.pkl')

    try:
        with open(pickle_file_path, 'rb') as file:
            ridge_model = pickle.load(file)
        return ridge_model
    except FileNotFoundError:
        raise Exception("Iris Decision Tree Classifier model file not found!")
    except pickle.UnpicklingError:
        raise Exception("Error unpickling Ridge model!")

# Create your views here.
def irisClassifier(request):
    if request.method == "POST":
        data = request.POST

        # Get form input values
        sepal_length = float(data.get('Sepal_Length'))
        sepal_width = float(data.get('Sepal_Width'))
        petal_length = float(data.get('Petal_Length'))
        petal_width = float(data.get('Petal_Width'))

        # Collect all inputs into an array for transformation
        
        iris = [[sepal_length,sepal_width,petal_length,petal_width]]
       

        # Load the scaler and model
        try:
            irisDecisionTreeClassifier=load_DecisionTreeClassifier_model()
           
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

      
 # Static image location

    # You can also send dynamic images or media files like uploaded images
    # image_url = 'path/to/your/dynamic/image.jpg'

   
        # Make a prediction using the ridge model
        try:
            result = irisDecisionTreeClassifier.predict(iris)
            if result==0:
             irisClass="Setosa"
             image_url = static('images/setosa.jpg') 
             
             print("Setosa")
            elif result==1:
             irisClass="Versicolor"
             image_url = static('images/versicolor.jpg') 
             print("Versicolor")
            elif result ==2:
               irisClass="Virginica"
               image_url = static('images/virginica.jpg') 
               print("Virginica")
            
        
           
        except Exception as e:
            return JsonResponse({"error": f"Prediction error: {str(e)}"}, status=500)
        

        context = {
        'results': irisClass,
        'image_url': image_url,
    }
        # Render the result in the template
        return render(request, 'irisClassifier.html',context)

    else:
        return render(request, 'irisClassifier.html')
    


