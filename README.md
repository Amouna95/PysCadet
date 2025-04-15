# Projet de satisfaction

<b>Contexte :</b>
Ce projet a été réalisé dans le cadre de la formation de Data Scientist de l'organisme Datascientest.	
Le sujet concerne la satisfaction des clients : pour de nombreux produits et services, elle peut se mesurer via les commentaires et avis laissés sur des sites dédiés.

<br><b>Objectif :</b>
- un objectif principal : prédire la satisfaction d’un client , c'est-à-dire le nombre d'étoiles (1 à 5) qu’il attribuera	
- un objectif optionnel : extraire des propos clés d’un commentaire (problème de livraison, article défectueux...)

<br><b>Les données à disposition:</b>
- Dans le cadre ce projet, un jeu de données nous a été remis. Celui contient les données de deux sites dédié aux recueil d’avis client :	
	- Trusted shops : les avis sont vérifiés, c’est-à-dire qu’ils font suite à une commande client.
	- Trustpilot : les avis sont directement postés à l’initiative des internautes.
- Le jeu de données est constitué de 10 variables :	
	- Commentaire du client
	- Star : note du client
	- Date : date du commentaire
	- Client : nom du client
	- Réponse : réponse éventuelle de la société au commentaire client
	- Source : Trusted shops ou Truspilot
	- Company : nom de l'entreprise en question, soit Showroom Privé, soit Veepee
	- Ville : ville du client
	- Date_commande : date de la commande
	- Ecart : ecart entre la date du commentaire et la date de la commande

# Contributeurs
Amina ABBI -  Dan FASSI -  Elsa FAUROUX  - Thierry PAYEN


# Notebooks & Traitement des données

<b>Les fichiers suivants sont les bases utilisées, resultats des différents traitements de données : </b>
- Data_Satisfaction_NAN_DUP_FR.csv
- Data_Satisfaction_Original.csv
- Data_Satisfaction_retraitement3.csv
- Data_Satisfaction_retraitement4.csv

<b>Les notebooks suivants ont pour objet le nettoyage et l'exploration des données : </b>
- Retraitement des données.ipynb
- Etape-1_Exploration_des_données.ipynb

<b>Les notebooks suivants ont pour objet les differentes itérations des modelisations : </b>
- Etape-2_Modelisation-1.ipynb
- Update Etape-2_Modelisation-1.ipynb
- Final Modélisation 3D.ipynb
- Final Modélisation 5D.ipynb
- Modélisation - CountVectorizer(150)_ màj.ipynb
- Modélisation - tfidf(150)_ màj.ipynb
- Modélisation - WordEmbedding - Neural Network.ipynb
- Modélisation - WordEmbedding(200)_ màj.ipynb
- Optimisations modeles ML.ipynb

<b>Les notebooks suivants ont pour objet les modelisations optimales retenues : </b>
- Final Modélisation 3D.ipynb
- Final Modélisation 5D.ipynb
- Extraction d'information.ipynb

# Démonstration des résultats : Streamlit

Lancer le fichier "streamlit" pour voir la présentation des résultats
