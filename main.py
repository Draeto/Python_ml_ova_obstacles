# Les différentes étapes du projet : 


# 1 ) Entrainement 

# Créer dataset contenant toutes les images de test et entrainement
# Ou 
# Initialiser le dataset de test et d'entrainement à partir du fichier csv

# Si entrainement pas encore effectué ( dataset vide)
# 	demander en boucle
# 		si l'utilisateur veut encore capturer des images d'entrainement ou de test
# 			capturer image de la caméra
# 			demander la classe à l'utilisateur (obstacle ou pas obstacle)
# 			calculer les variable explicatives
# 			rajouter dans le dataset
# 		Sauvegarde du dataset dans le fichier csv

# Fabrication des données d'entrainement à partir du dataset 
# ( liste de variable explicatives et classes correspondantes déjà connues)

# Fabrication des données d'entrainement à partir du dataset
# ( liste de variable explicatives et classes correspondantes déjà connues)

# Entrainement du classifier à partir des données d'entrainement

# Génération de la matrice de confusion pour vérifier la bonne précision de notre modèle

# 2 ) Prediction ( dans la boucle principale)

# Récupérer la dernière image de la caméra

# Calculer les variables explicatives ( appel de la fonction characterize)

# On demande au classifier de prédire la classe de l'image ( obstacle ou pas obstacle)
# en passant les 2 variables explicatives ( moyenne gradient et ecart-type gradient)

# On affiche la prédiction dans la console




# Liste des librairies nécessaire à la réalisation du projet 
from time import sleep
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mp
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#Imports nécessaires pour se connecter au robot et récupérer ses informations (ici les images envoyées par la caméra frontale)
from j2l.pyrobotx.robot import IRobot
from j2l.pyrobotx.client import OvaClientMqtt, OvaClientHttpV2

#Imports des classes utilisées par notre programme principale
from characterizer.image import ObstacleGradHalfImageCharacterizer
from characterizer.pixel import GradPixelCharacterizer
from characterizer.pixel import IPixelCharacterizer



# #Pour piloter une ova via un broker MQTT
# robot: IRobot = OvaClientMqtt(server="",
#                                  port=,
#                                  useProxy=,
# 								  arena="")


# Pour piloter une ova sur un LAN ou si vous êtes directement connecté à son point d'accès
robot:IRobot = OvaClientHttpV2(url="") 

#variables globales : ici les duex que peuvent avoir nos images ainsi que leur nombre
classesNames = ["pas d'obstacle", "obstacle"]
nClasses = len(classesNames)

# Boucle servant à vérifier la connexion au robot
print("########################")
while (robot.isConnectedToRobot() == False):
	print("Awaiting robot connection...")
	robot.update()
	sleep(1)
robot.enableCamera(True)

# création de la classe qui nous renvoir la luminosité d'un pixel grâce à la méthode characterize
class OvaLuminosityPixelCharacterizer(IPixelCharacterizer):
	def characterize(self, x : int, y:int ) -> float:
		return(robot.getImagePixelLuminosity(x,y))
	


# Boucle servant à récupérer les informations de chaque image (Moyenne du gradient et Ecart-type du gradient) 
# et à les charger dans un fichier csv si celui-ci n'existe pas déjà
if not os.path.exists("data.csv"):
	with open("data.csv", "a")as f:
		f.write("Moyenne,EcartType,Classe\n")
		nCaptures = int(input("Nombre d'images pour chaque classe:"))
		for iclasse in range(nClasses):
			#on marque ici une pause dans notre boucle afin de permettre à l'utilisateur de prendre des images adaptées à la prochaine classe
			#(ici les images où un obstacle est présent)
			input(f"Début capture classe {classesNames[iclasse]}")
			for i in range(nCaptures):
		#Capturer une image
				robot.update()

		#Calcul des variables explicatives
				characterizer = ObstacleGradHalfImageCharacterizer(
				robot.getImageHeight(),
				robot.getImageWidth(),
				GradPixelCharacterizer(
					OvaLuminosityPixelCharacterizer()
			)
			)	# ajout de l'image dans le fichier csv avec l'affichage des informations de cette dernière dans la console
				x= characterizer.characterize()
				print("x=", x)
				f.write(f"{x[0]}, {x[1]}, {classesNames} \n")



# Chargement des données du fichier csv dans un dataset grâce à la librairie pandas
Data = pd.read_csv('data.csv')
X=Data.iloc[:,:-1].values
Y=Data.iloc[:,-1].values

# Séparation des données en jeux d'entrainement et de test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

#Application du modèle SVM sur les données d'entrainement du dataset
classifier = SVC(kernel = 'linear', random_state=0)
classifier.fit(X_train,Y_train)

#Création et affichage des résultats de la prédiction de la classe d'une image grâce aux données de test et création d'une matrice de confusion dans la console
y_pred = classifier.predict(X_test)
confusionMatrix = confusion_matrix(Y_test,y_pred)
print(f"Prédiction sur données de test:{y_pred}")
print(f"Matrice de confusion:{confusionMatrix}")
print(classification_report(Y_test, y_pred))

# Création des datasets qui serviront pour les graphiques matplotlib
histO= Data[Data.Classe == 1].iloc[ : , 0:1].values
histP= Data[Data.Classe == 0].iloc[ : , 0:1].values
scattOM= Data[Data.Classe == 1].iloc[ : , 0:1].values
scattPM= Data[Data.Classe == 0].iloc[  :, 0:1].values
scattOET= Data[Data.Classe == 1].iloc[ : , 1:2].values
scattPET= Data[Data.Classe == 0].iloc[ : , 1:2].values

#Création des graphiques grâce à la librairie matplotlib
bins = np.linspace(0, 10)
LabelClasses = []
colormap = mp.colors.LinearSegmentedColormap.from_list('', ['white', 'black'])

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)

ax0.set_title("Histogramme")
ax0.hist(histO, bins, alpha=0.5, label = 'obstacle' )
ax0.hist(histP, bins, alpha=0.5, label = 'Pas obstacle' )
ax0.legend(classesNames)

ax1.set_title("Nuage de points")
ax1.scatter(scattOM, scattOET)
ax1.scatter(scattPM, scattPET)
ax1.legend(classesNames)

ax2.set_title("Matrice de confusion")
im = ax2.imshow(confusionMatrix, cmap = colormap)
ax2.set_xticks(np.arange(len(classesNames)), labels = classesNames)
ax2.set_yticks(np.arange(len(classesNames)), labels = classesNames)
ax2.set_ylabel("Expected")
ax2.set_ylabel("Predicted")
fig.colorbar( im, orientation='vertical')

plt.show()

#Boucle principale qui détermine le comportement du robot en fonction de la classe de l'image renvoyée par la caméra
temps = int(input("combien de temps voulez-vous que le robot fonctionne ? :"))
while temps >= 0 : 
	robot.update()
	characterizer = ObstacleGradHalfImageCharacterizer(
				robot.getImageHeight(),
				robot.getImageWidth(),
				GradPixelCharacterizer(
					OvaLuminosityPixelCharacterizer()
			)
			)	#rajouter dans le dataset
	X = characterizer.characterize()

	print("fin de characterisation")

	Y = classifier.predict([X])
	print(f"Prédiction sur données de test:{Y}")

	if Y[0] == 0 :
		print("R.A.S")
		robot.setLedColor(0,255,0)
		robot.setMotorSpeed(leftPower=-100, rightPower= 100)
		temps -= 1
	else : 
		print("S.T.O.P")
		robot.playMelody([(440,100)])
		robot.setLedColor(255,0,0)
		robot.setMotorSpeed(leftPower= -50, rightPower= -50)
		temps-= 1
		
robot.stop()
robot.update()