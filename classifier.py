# example of using a pre-trained model as a classifier
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

# load the model
model = VGG16()
def what_is(image_path): 
    # load an image from file
    image = load_img(image_path, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    yhat = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # print the classification
    print('%s (%.2f%%)' % (label[1], label[2]*100))

what_is('tableau1.png')
what_is('mushroom.jpg')
what_is('tableau2.jpg')

## Idée : faire mouliner VGG sur tout un tas de photos de tableaux pour qu'il nous sorte les features correspondantes : et ce, incluant une liste de tableau déformés à la main : e.g. via des morphismes. (Data Augmentation techniques)
## Ensuite : faire une autre base de données non-target : surtout en rapport avec ce qu'on voit dans les musées : fauteuils, banc, sol, couloir etc... 
## Réunir ces deux bases de données et train en dernière layer VGG (transfer learning) pour qu'il sache renvoyer une proba qu'un tableau soit un tableau : faire attention au nombre de faux positifs : objectif : un pourcentage de faux positif très faible. 

## Etape 2 : repérer le tableau en question (draw un paralépipède autour !) et le resize (morphisme) pour normaliser la donnée --> construction d'une IA ? 
## --> J'ai trouvé un truc de ouf qui s'appelle un "Saliency Map Generation" https://deeplearning.cms.waikato.ac.nz/examples/inference/#class-map-lookup-tables
## --> renvoie une heatmap des trucs qui permettent à l'ia de dire que c'est un tableau
## --> sans doute lourd en computation, nécessite surement une technique + simple (cf. semaine 1 avec le masque sur visage)
## --> pourquoi pas utiliser un truc comme "detectMultiscale" ? (cf. semaine 1 facedetection.py)

## Etape 3 : reprendre notre image resized (la plus clean possible !) et tenter de retrouver à quel tableau de la base de données du musée il correspond :
## 3) a) Via méthode du plus proche voisin
## 3) b) Via de la proximité statistique d'histogramme de couleur (solution préférable pour des questions de temps de calcul, plutôt qu'une IA !)

## N.B. : penser au fait que si l'étape 3 galère en accuracy, changement d'objectif : tenter plutôt d'orienter l'aveugle (via bip sonore) jusqu'à ce que le heatmap (via étape 2) du tableau soit à des dimensions acceptables (en size et en torsion !)

## Reading :
"""
https://medium.com/analytics-vidhya/predict-artist-from-art-using-deep-learning-9f465f8879d7
https://betterprogramming.pub/shazam-for-paintings-a-computer-vision-project-513ff2e1b498
https://www.moma.org/calendar/exhibitions/history/identifying-art
https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/
https://stackoverflow.com/questions/36218385/parameters-of-detectmultiscale-in-opencv-using-python
https://deeplearning.cms.waikato.ac.nz/examples/inference/#class-map-lookup-tables 
"""