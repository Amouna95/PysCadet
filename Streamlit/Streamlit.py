

import streamlit as st
import pandas as pd

df1= pd.read_csv("Data_Satisfaction_Original.csv")
df2= pd.read_csv("Data_Satisfaction_retraitement4.csv")


models= ["Gradient Boosting","Random Forest","Support Vector Machine", "Régression logistique", " Réseaux de neurones denses"]
techniques= ["Countvectorizer", "TF-IDF","Word embedding"]

st.image("new-logo.png", width=150)
html_temp = f"""
    <h2 style = "color:#2dcdd9; text_align:center; font-weight: bold;"> Analyse de la Satisfaction Client </h2>
    <h3 style = "color:#2dcdd9; text_align:center; font-weight: bold;"> Projet de groupe Datascientest </p>
    </div>
"""
st.markdown(html_temp, unsafe_allow_html = True)

st.sidebar.title("Navigation")

pages = ["Présentation du projet","Exploration des données","Modélisation","Extraction d'information","Conclusion"]
page = st.sidebar.radio("Choisir une page",pages)


if page == pages[0]:
    st.header("Présentation du projet")
    st.markdown("**ABBI Amina**")
    st.markdown("**FASSI Dan**")
    st.markdown("**FAUROUX Elsa**")
    st.markdown("**PAYEN Thierry**")
    st.image("Photo_presentation_projet.png")
    st.subheader("Contexte:")
    st.write("Ce projet a été réalisé dans le cadre de la formation de Data Scientist de l'organisme Datascientest.")
    st.write("Le sujet concerne la satisfaction des clients. Pour de nombreux produits et services, elle peut se mesurer via les commentaires et avis laissés sur des sites dédiés.")
    st.subheader("Objectifs:")
    st.write("- Un objectif principal : prédire la satisfaction d’un client , c'est-à-dire le nombre d'étoiles (1 à 5) qu’il attribuera.")
    st.write("- Un objectif optionnel : extraire des propos clés d’un commentaire (problème de livraison, article défectueux...)")
    st.subheader("Données à disposition:")
    st.write("Dans le cadre de ce projet, un jeu de données nous a été remis. Celui contient les données de deux sites dédiés aux recueils d’avis client :")
    st.write("-Trusted shops : les avis sont vérifiés, c’est-à-dire qu’ils font suite à une commande client.")
    st.write("-Trustpilot : les avis sont directement postés à l’initiative des internautes.")
    st.write("Ci-dessous l'aperçu des données sous format Dataframe Pandas composé de 19863 lignes et 11 colonnes en chargeant le fichier “TrustReview.csv”:")
    #st.dataframe(df1.head())
    st.image("Etape-1_Exploration_des_données_0.png")
    st.write("Le jeu de données est constitué de 11 variables :")
    st.write("-Commentaire du client")
    st.write("-Star: note du client")
    st.write("-Date: date du commentaire")
    st.write("-Client : nom du client")
    st.write("-Réponse : réponse éventuelle de la société au commentaire client")
    st.write("-Source : Trusted shops ou Truspilot")
    st.write("-Company : nom de l'entreprise en question, soit Showroom Privé, soit Veepee")
    st.write("-Ville : ville du client")
    st.write("-Date_commande : date de la commande")
    st.write("-Ecart : écart entre la date du commentaire et la date de la commande")
    
     
if page == pages[1]:
    st.header("Exploration des données")
    st.subheader("Nettoyage de données:")
    st.image("Etape-1_Exploration_des_données_1.png")
    st.write("La colonne maj ainsi que les lignes sans commentaires ont été supprimés.")
    st.write("De plus, dans notre dataset, nous avons identifié 543 lignes doublons que nous avons supprimés. Et les lignes des commentaires en langue autre que français ont été supprimées de notre Dataset.")
    st.subheader("Analyse des données:")
    st.markdown("**Répartition de la valeur numérique star:**")
    st.image("Etape-1_Exploration_des_données_3.png")        
    st.write("On observe que les modalités extrêmes (les notes 5 et 1) sont les plus représentées alors que les modalités intermédiaires sont plus en retrait.")
    st.markdown("**Répartition des notes en croisant les entreprises et les sources**")
    st.image("Etape-1_Exploration_des_données_9.png")
    st.write("Au delà de voir que seul ShowRoom est sur TrustedShop, ce qui est le plus étonnant dans ces deux graphiques et le fait que sur TrustPilot, ShowRoom obtient à une grande majorité une note de 1, alors que sur TrustedShop, la majorité des avis sont entre 4 et 5.")
    st.markdown("**Présence de réponse en fonction de la note:**")
    st.image("Etape-1_Exploration_des_données_10.png")
    st.write("On remarque que dans la majorité des cas, lorsqu'une réponse est présente dans la base de données, cela correspond à une bonne note.")
    st.markdown("**Évolution temporelle du nombre de commentaires par source:**")
    st.image("Etape-1_Exploration_des_données_13.png")
    st.write("En voyant ce graphique, il devient clair que l'arrivée massive de commentaires ainsi que de bonnes notes est lié à l'arrivée de la source TrustedShop à partir de mi-2020.")
    st.markdown("**Répartition des notes en fonction de la présence d'informations client:**")
    st.image("Etape-1_Exploration_des_données_18.png")
    st.write("Poster un commentaire sur TrustedShop semble ne pas nécessiter de renseigner obligatoirement un nom client. Il semble donc plus aisé de poster un avis sur TrustedShop.")
    st.image("Etape-1_Exploration_des_données_19.png")
    st.write("On peut déduire de ce graphique qu’il n’est pas demandé d’indiquer le nom de la ville sur TrustPilot, et qu’il s’agit là aussi pour TrustedShop d’un renseignement optionnel.")
    st.markdown("**Analyse Verbatims:**")
    st.write("Une première étape dans la compréhension du contenu des commentaires a été la création de plusieurs nuage de mots. Chaque itération est le résultat de l’ajout successif de mots considérés comme vides à la liste de stop words. Ci-dessous le dernier nuage de mots obtenu:")
    st.image("Etape-1_Exploration_des_données_22.png")
    st.write("En analysant la taille des commentaires après retraitement stop words,on a constaté que la taille du commentaire soit liée à la note laissée. De manière générale, plus la note laissée est bonne, plus le commentaire est court.")
    st.image("Etape-1_Exploration_des_données_23.png")
    st.write("On observe une nette différence de taille dans les commentaires entre TrustPilot et TrustedShop concernant l'entreprise ShowRoom:")    
    st.image("Etape-1_Exploration_des_données_24.png")
    st.write("Ci-dessous la répartition de la taille des commentaires sur TrustedShop pour l'entreprise ShowRoom. Plus de 75% des avis ShowRoom sur TrustedShop laissant 5 étoiles comportent moins de 20 mots. 50% même en comprennent moins de 10:") 
    st.image("Etape-1_Exploration_des_données_26.png")        
    st.write("En affichant la fréquence des notions les plus fréquemment utilisées dans les commentaires, on a sans surprise retrouvé les thèmes (service client, qualité du produit, livraison, retours & remboursements):")
    st.image("Etape-1_Exploration_des_données_27.png")        
    st.markdown("**Synthese:**")
    st.write(" - La comparaison stars / source puis stars /company nous a permis d'identifier un écart anormal en faveur de ShowRoom. \n ") 
    st.write(" - La présence d'une réponse peut nous permettre de catégoriser le sentiment du commentaire.\n ") 
    st.write(" - La présence de données clients tel que le nom ou la ville est optionnelle sur TrustedShop, facilitant ainsi la publication rapide d'avis. \n ")
    st.write(" - Une relation existe entre la taille d'un commentaire et la note attribuée. \n")
    st.write(" - Les métadonnées peuvent participer à l'estimation de la note. (par exemple la Source : la majorité des notes négatives sont sur TrustPilot tandis que la majorité des notes positives sont sur TrustedShop).\n ")
    st.write(" - Il est possible de réduire la taille du dictionnaire utilisé pour interpréter les commentaires.\n")

    
if page == pages[2]:
    st.header("Retraitement de données")
    st.write("**Métadonnées:**")
    st.write("   - Colonnes **source et compagny:** remplacement par des valeurs 0 ou 1")
    st.write("   - Encodage de la colonnes **réponse** (absence, positive, négative)")
    st.write("   - Séparation et normalisation des données dates à partir de la colonne **date_commande**")
    st.write("   - Remplissage de la colonne **écart** par son mode puis normalisation")
    st.write("**Commentaires:**")
    st.write("   - Applications et optimisations de la fonction stop word")
    st.write("   - Suppression ponctuations")
    st.write("   - Suppression nombres et numéros")
    st.markdown("Ci-dessous un aperçu de notre Dataframe suite au dernier retraitement du fichier de données:")    
    st.dataframe(df2.head()) 
    st.header("Modélisation 5 classes")
    st.subheader("Modèles de Machines Learning classiques")
    st.markdown("Ci-dessous les résultats obtenus en combinant les différents Modèles de Machnine Learning classiques/Techniques de vectorisations sur 5 classes:")
    st.image("Modelisation_5classes_1.png")
    st.markdown("La meilleure précision obtenue sur l'ensemble de test provient du modèle GradientBoosting quelque soit la technique de vectorisation.")
    st.markdown("Ci-dessous le rapport de classification et la matrice de confusion pour le modèle Gradient Boosting:")
    st.image("Modelisation_5_DNN - Confusion.png")
    st.image("Modelisation_5_GB - Classification.png")
    st.subheader("Réseau de neurones Dense")
    st.markdown("Nous avons considéré un modèle simple: 1 couche Dense de 6 neurones avec fonction d’activation tanh puis 1 couche Dense de 5 neurones avec fonction d’activation softmax.")
    st.markdown("L'évaluation du modèle nous fournit le plot et les résultats suivants:")
    st.image("Modelisation_5_plot.png")
    st.image("Modelisation_5_DNN - Classification.png")
    st.image("Modelisation_5_DNN - Confusion.png")
    st.header("Modélisation 3 classes")
    st.markdown("Au lieu des classes 1,2,3,4,5 on retient les nouvelles classes 1,2,2,2,3 respectivement. En utilisant les mêmes méthodes que précédemment, le meilleur résultat est obtenu par le réseau de neurones dont voici les résultats:")
    st.image("Modelisation_3classes_Resultas_RN.png")
    st.image("Modelisation_3classes_ClassificationReport_RN.png")
    st.markdown("""
    <style>
    /* The input itself */
    div[data-baseweb="select"] > div {
        background-color: #2dcdd9 !important;
        font-size: 18px !important;
        }

    /* The list of choices */
    li>span {
        color: white !important;
        font-size: 18px;
        background-color: white !important;
        }

    li {
        background-color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)  


if page == pages[3]:
    st.markdown("Pour une entreprise, il peut être long et  fastidieux de lire et d’analyser les commentaires qui sont pourtant essentiels à la compréhension de la satisfaction client. Nous avons exploré deux méthodologies, pouvant mener à une possible automatisation de cette tâche.")
    st.subheader("Méthode manuelle:")
    st.write("- Identifier les mots clés  les plus fréquents ")
    st.write("- Construire des catégories métier liées à ces mots clés")
    st.image("Extraction_information_methode_manuelle.png")
    st.write("Comme exemple, nous avons créé la catégorie 'produit' à partir des mots clés: produit, qualité, prix et taille. ")
    st.write("Cela reste cependant une méthode limitée.")
    st.subheader("Méthode TF-IDF:")
    st.write("- Utilisation de l'algorithme tf-idf")
    st.write("- Pour chaque commentaire extractions des 10 mots ayant les plus grand poids")
    st.image("Extraction_information_methode_tfidf.png")
    st.write("Cette methode necessite d’autres étapes afin d’être correctement exploitable.")
    
    
if page == pages[4]:
    st.subheader("Conclusion:")
    st.write("En utilisant les meilleurs paramètres sur une même vectorization, le meilleur modèle est le GradientBoosting avec un résultat 70,7%.")
    st.write("Seul le réseau de neurones est plus performant mais toujours avec la difficulté de prédire 5 classes.")
    st.write("Les classes intermédiaires sont difficiles à prédire.")
    st.write("En test sur une modélisation sur 3 classes, le meilleur résultat est supérieur à 80%.")
    st.write("L’utilisation de 5 classes avec nos modèles fait mieux que les 20 % théoriques mais les performances sont limitées à 70% quand l’utilisation de moins de classes nous amène à plus de 80%.")
    st.write("En complément nous avons investigué deux méthodes d’extraction de l’information: une empirique et l’autre plus formelle qui donne des résultats intéressants mais qui nécessiteraient plus d’approfondissement pour être exploitable pour une entreprise.")
    st.subheader("Regard critique:")
    st.write("On peut se poser la question de la limitation de la prédiction pour un usage professionnel. Le jeu de données était- il trop petit et mal distribué: pourquoi c’est mieux d’avoir 3 classes plutôt que 5 dans ce contexte. Les émoticônes ont été considérés comme du bruit: peuvent-ils contribuer pour renseigner le sentiment?")
    st.write("Avec plus de temps on aurait developper:")
    st.write("- Optimiser / compléter la modelisation sur 2 ou 3 classes (on est resté sur l'optimum 5 classes).")
    st.write("- La partie Word Embedding en utilisant les réseaux de neurones: recherche de modèle performant (préentrainé), construire son modèle.")
    st.write("- La partie retraitement des données plus poussée: utilisation des smiley ...")
    st.write("- Aborder la composante métier dans l'extraction d'information.")
    st.subheader("Retour d'expérience:")
    st.write("Tout au long de ce projet nous avons été amenés à explorer le jeu de données en utilisant différentes méthodes de visualisation et les techniques d’intelligence artificielle.")
    st.write("Nous avons manipulé voir découvert de nouveau outils et acquis de nouvelles compétences: github, streamlit, googledoc, etc")
    st.write("Nous remercions Antoine, notre mentor projet pour son écoute, ses commentaires lors de nos points projet ainsi que les informations fournies.")
