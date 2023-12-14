# Détection d'obstacle automatisée
Automatisation des déplacements d'un robot grâce à un algorythme de Machine learning développé en python

## 🎯 Contexte & cahier des charges
Développé dans le cadre de mon bachelor développeur IA au sein de l'Epsi à Rennes, pour monter en compétence en Machine learning et langage python.

Le but de ce projet est de permettre à un robot d'adapter son comportement en présence ou non d'un objet face à lui. 
Pour cela un algorythme de machine learning à été développé et on souhaite obtenir les résultat de ce modèle prédictif 
via des graphiques et des informations affichées dans la console .

## 🤼‍♀️ Use cases
L'utilisateur aura donc les résultats du modèle prédictif dans la console et sous forme de graphique.
Ensuite il pourra choisir pendant combien de temps le robot devra se déplacer.

L'adminisatrateur pourra quand à lui paramétrer les données d'entrainement et de test 
en choisissant combien d'images seront présentes dans le modèle de données.

## 🖧 Matériel 
Les robots utilisés sont des modèles Ova, développés par jus-de-liens, pouvant réaliser différentes actions comme se déplacer ou envoyer un signal sonore.

Il y a deux manières de se connecter aux robots : 
   - d'abord directement grâce à son adresse ip et via la connexion wifi

   - Ou bien en se connectant à un serveur lui même connecté au robot

## 📂 Arborescence du projet
Dans le dossier characterizer, on trouve tous les modules contenant les classes utilisées par notre programme principale, ce dernier étant dans le module "main.py".

Le fichier data.csv est créer par le programme si il n'existe pas déjà et contien les informations des images récupérer par la caméra de Ova.

Le fichier "img.jpeg" représente la dernière image renvoyée par le robot.

## ✅ Pré-requis
Un ordinateur avec visual studio code ainsi qu'un robot Ova sont nécessaires pour la réalisation de ce projet.

## ⚙️ Installation
les librairies python présentes dans les imports du module "main.py" doivent être installés grâce à la commande "pip install  'nom de la libraire' "
