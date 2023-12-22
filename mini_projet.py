import pandas as pd 
import statistics as stat
import numpy as np
import matplotlib.pyplot as plt
import csv
import math 
from statistics import mean
from scipy.stats import * 
import seaborn as sns

'''Exercice 1'''

# 1 (a,b et c)
data = pd.read_csv('dataMP.csv', sep = ';', encoding = 'latin_1')
X = data.iloc[ :, 1]
Y = data.iloc[ :, 2]

# 2)
def nuage_point(x, y):
    plt.scatter(x, y)
    plt.title("nuage de points de Y en fonction de X")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# 3)
m_X = np.mean(X)
m_Y = np.mean(Y)
ec_type_X = np.std(X)
ec_type_Y = np.std(Y)

# 4)
def cov(x, y):
    covariance = np.cov(x, y, ddof = 1)[0, 1]
    return covariance

# 5)
def correlation(cov, ec_type_1, ec_type_2):
    r = cov / (ec_type_1 * ec_type_2)
    return r

# 6)
def droite_regression(x, y):
    var_X = np.var(x, ddof = 1)
    
    a = cov(x, y) / var_X
    b = m_Y - a * m_X
    
    plt.scatter(x, y)

    x_range = np.linspace(min(x), max(y), 100)
    y_regression = a * x_range + b
    plt.plot(x_range, y_regression, color='blue', label='Régression linéaire')
    
    # Ajouter des étiquettes et une légende
    plt.title('Nuage de points avec la droite de régression linéaire et le point G')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

# 7)
def point_G(moyenne_x, moyenne_y):
    x = moyenne_x
    y = moyenne_y
    G = plt.scatter(x, y, color = 'black', label = "Point G moyen")
 
# 8 (a, b et c)
def residuel(x, y):
   
    var_X = np.var(x, ddof = 1)
    a = cov(x, y) / var_X
    b = m_Y - a * m_X

    residus = y - (a * x + b)
    Y_hat = a * x + b
   
    
    # Méthode 1 : On calcule la variance des résidus 
    variance_residuelle_1 = np.var(residus, ddof = 1)
    # Méthode 2 : On calcule la somme des carrés des résidus divisée par (n-2)
    variance_residuelle_2 = np.sum(residus ** 2) / (len(data) - 2)
    print(f"Variance résiduelle (méthode 1) : {variance_residuelle_1}")
    print(f"Variance résiduelle (méthode 2) : {variance_residuelle_2}")
    
    # Méthode 1 : On calcule la variance des valeurs prédites directement
    variance_expliquee_1 = np.var(Y_hat, ddof = 1)
    # Méthode 2 : On calcule la somme des carrés des valeurs prédites divisée par (n-1)
    variance_expliquee_2 = np.sum((Y_hat - np.mean(y)) ** 2) / len(data)
    print(f"Variance expliquée (méthode 1) : {variance_expliquee_1}")
    print(f"Variance expliquée (méthode 2) : {variance_expliquee_2}")
    
    
    # Vérifier l'équation de la variance
    variance_Y = np.var(y, ddof=1)
    equation_variance = variance_expliquee_1 + variance_residuelle_1
    print(f"Variance totale (var(Y)) : {variance_Y}")
    print(f"Variance expliquée + Variance résiduelle : {equation_variance}")
      


# Affichage de toutes les questions 
def affichage():
    nuage_point(X, Y)
    point_G(m_X, m_Y)
    droite_regression(X, Y)
    print(f"Moyenne X : {m_X}")
    print(f"Moyenne Y : {m_Y}") 
    print(f"Ecart type X : {ec_type_X}")
    print(f"Ecart type Y : {ec_type_Y}")
    print(f"Covariance XY : {cov(X, Y)}") 
    print(f"Corrélation linéaire XY :  r = {correlation(cov(X, Y), ec_type_X, ec_type_Y)}\n")
    residuel(X, Y)
    

# affichage()

"""EXERCICE 2"""

# Données
T = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
V = np.array([5.098, 3.618, 2.581, 2.011, 1.486, 1.028, 0.845, 0.573, 0.429, 0.29, 0.2])

# Calcul des valeurs de x, T (ms)
X_ln = T

# Calcul des valeurs de y
Y_ln = np.log(V)

#print(f" valeurs de X_ln : { X_ln} \n ")
#print(f" valeurs de Y_ln : {Y_ln} \n ")

# Régression linéaire
pente, intercept, r_value, p_value, erreur_std = linregress(X_ln, Y_ln)

# Calcul de V0 et tau
V0 = np.exp(intercept)
tau = -1 / pente

#print(f" V0 = {V0} \n tau = {tau}")

Y_pred =  pente * X_ln + intercept

# Affichage des résultats
plt.scatter(X_ln, Y_ln, label='Données')
plt.plot(X_ln, Y_pred, color='red', label='Régression linéaire')
plt.xlabel('X_ln (T en ms)')
plt.ylabel('Y_ln')
plt.legend()
plt.title('Régression Linéaire')
#plt.show()


t = 53  # temps en ms 

# Calcul de V pour t = 53 ms
V = V0 * np.exp(-t / tau)

#print(f"La valeur de V pour t = {t} ms est : {V}")



"""EXERCICE 3"""

XPA = np.array([100, 61, 76, 74, 90, 93, 102, 98, 103, 110, 117, 118, 112, 115, 116, 121, 134, 130])
YPI = np.array([10, 50, 84, 99, 113, 122, 128, 143, 145, 159, 172, 188, 204, 213, 220, 242, 254, 273])

# 8.(a) Régression linéaire de y en x
a, b = np.polyfit(XPA, YPI, 1)

# 9.(a) Régression linéaire de x en y
a_prime, b_prime = np.polyfit(YPI, XPA, 1)

# 8.(b) Droite d'ajustement
regression_line = a * XPA + b

# 9.(b) Droite d'ajustement
regression_line_prime = a_prime * YPI + b_prime

# 5. covariance entre x et y.
covariance = np.cov(XPA, YPI)[0, 1]
#print(f'Covariance entre x et y : {covariance}')

# 6. coefficient de corrélation r.
correlation_coefficient = np.corrcoef(XPA, YPI)[0, 1]
#print(f'Coefficient de corrélation r : {correlation_coefficient}')

# 8.(a) droite de régression linéaire de y en x
#print(f"Droite de régression linéaire : y = {a:.2f}x + {b:.2f}")

# 9.(a) droite de régression linéaire de x en y
#print(f"Droite de régression linéaire : x = {a_prime:.2f}y + {b_prime:.2f}")

# 8.(b) valeurs ajustées ŷi
y_pred = a * XPA + b
# distances de chaque point par rapport à la droite d'ajustement
residuals = YPI - y_pred

# 9.(b) valeurs ajustées x̂i 
x_pred = a_prime * YPI + b_prime
# distances de chaque point par rapport à la droite d'ajustement
residuals_prime = XPA - x_pred

#8.(b) valeurs ajustées ŷi distances résiduelles
# for i in range(len(XPA)):
    #print(f"Valeur ajustée ŷ{XPA[i]} : {y_pred[i]:.2f}, Distance résiduelle : {residuals[i]:.2f}")

# 9.(b) valeurs ajustées x̂i et distances résiduelles
# for i in range(len(YPI)):
    #print(f"Valeur ajustée x̂{YPI[i]} : {x_pred[i]:.2f}, Distance résiduelle : {residuals_prime[i]:.2f}")
    
# 8.(c) variance résiduelle variance expliquée
residual_variance = np.var(residuals)
explained_variance = np.var(y_pred)
#print(f"Variance résiduelle de y en x: {residual_variance:.2f}")
#print(f"Variance expliquée de y en x: {explained_variance:.2f}")

# 9.(c) variance résiduelle et variance expliquée
residual_variance_prime = np.var(residuals_prime)
explained_variance_prime = np.var(x_pred)
#print(f"Variance résiduelle de x en y : {residual_variance_prime:.2f}")
#print(f"Variance expliquée de x en y: {explained_variance_prime:.2f}")
    

G = (np.mean(XPA), np.mean(YPI))
plt.scatter(XPA, YPI,color='red', label='Nuage de points y en x')
plt.scatter(YPI, XPA,color='blue', label='Nuage de points x en y')
plt.plot(XPA, regression_line, color='red', label='Droite de régression linéaire y en x')
plt.plot(YPI, regression_line_prime, color='blue', label='Droite de régression linéaire x en y')
plt.scatter(*G, color='black', marker='x', label='Point moyen G')
plt.xlabel('Production Agricole')
plt.ylabel('Production Industrielle')
plt.title('Nuage de points avec le point moyen G')
plt.legend()
#plt.show()

# 10(b) Coefficients de régression et corrélation
r = np.corrcoef(XPA, YPI)[0, 1]  # coefficient de corrélation entre x et y

# écarts-types
sigma_x = np.std(XPA)
sigma_y = np.std(YPI)

# relations
result_a = r * sigma_y / sigma_x
result_a_prime = r * sigma_x / sigma_y

#print(f' a\': {a_prime:.2f}, Résultat numérique : {result_a_prime:.2f}')
#print(f' a : {a:.2f}, Résultat numérique : {result_a:.2f}')

# 11
# valeur prédite de la production agricole en 1962
prediction_1962 = a_prime * 273 + b_prime
#print(f"La valeur prédite de la production agricole en 1962 est de : {prediction_1962:.2f}")




"""EXERCICE4"""


def boite_dispersion(serie, titre="Boîte de Dispersion"):

    fig, ax = plt.subplots()
    ax.boxplot(serie)
    ax.set_title(titre)
    ax.set_ylabel("Valeurs")
    plt.show()

def  tracer_histogramme(liste_longueurs, titre, abcisse, n):
    plt.hist(liste_longueurs, bins = n, alpha = 0.5, color = 'blue', edgecolor = 'black')

    plt.title(titre)
    plt.xlabel(abcisse)
    plt.ylabel('Effectif')

    

BonasaUmbellus = [153,165,160,150,159,151,163,160,158,150,154,153,163,150,158,150,158,155,163,156,157,162,160,152,164,158,153,162,166,162,165,157,174,158,171,162,155,156,159,162,152,158,164,164,162,158,156,171,164,158]


min_BonasaUmbellus = min(BonasaUmbellus)
max_BonasaUmbellus = max(BonasaUmbellus)
etendue_BonasaUmbellus = max_BonasaUmbellus-min_BonasaUmbellus
moyenne_BonasaUmbellus = stat.mean(BonasaUmbellus)
mediane_BonasaUmbellus = stat.median(BonasaUmbellus)
quartiles_BonasaUmbellus = np.quantile(BonasaUmbellus, [0, 0.25, 0.5, 0.75, 1])
ecart_quartiles_BonasaUmbellus = quartiles_BonasaUmbellus[3]-quartiles_BonasaUmbellus[1]


variance_BonasaUmbellus = np.var(BonasaUmbellus)
ecart_type_BonasaUmbellus = np.std(BonasaUmbellus)

# print("EXERCICE 4:")
# print("longueur minimun = ", min_BonasaUmbellus, "cm")
# print("longueur maximun = ", max_BonasaUmbellus, "cm")
# print("L'étendue = ", etendue_BonasaUmbellus, "cm")
# print("moyenne = ", moyenne_BonasaUmbellus, "cm")
# print("médiane = ", mediane_BonasaUmbellus, "cm")
# print("Voici les quartiles de l'échantillon: \n -Q1 =", quartiles_BonasaUmbellus[0], "\n -Q2 =", quartiles_BonasaUmbellus[1], "\n -Q3 =", quartiles_BonasaUmbellus[2], "\n -Q4 =", quartiles_BonasaUmbellus[3])
# print("Variance = ", variance_BonasaUmbellus)
# print("Ecart type = ", ecart_type_BonasaUmbellus)

# boite_dispersion(BonasaUmbellus)
# tracer_histogramme(BonasaUmbellus, "Histogramme des longueurs de la rectrice centrale", 'Longuer(cm)', etendue_BonasaUmbellus)
# plt.show()




"""EXERCICE 5"""

def importer_csv(nom_fichier):    
    data = []
    with open(nom_fichier, 'r') as fichier_csv:
        lecteur_csv = csv.reader(fichier_csv)
        for ligne in lecteur_csv:
            for i in ligne:
                data.append(float(i))
    return data



S1 = importer_csv('data.csv')
min_S1 = min(S1)
max_S1 = max(S1)
etendue_S1 = max(S1)-min(S1)

POIDS1 = S1[0:15]
m1 = stat.mean(POIDS1)

POIDS2 = S1[15:-1]
m2 = stat.mean(POIDS2)

mtot = (m1+m2)/2



moyenne_S1 = stat.mean(S1)
ecart_type_S1 = np.std(S1)


# print("EXERCICE 5:")
# print("La moyenne de l'échantillon est de ", moyenne_S1, "kg")
# print("Ecart type = ", ecart_type_S1)
# print("m1 = ", m1)
# print("m2 = ", m2)
# print("moyenne totale = ", mtot)

# boite_dispersion(S1)
# tracer_histogramme(S1, "Diagramme en bâtons des pesées des 30 bébés", "Poids en kg", [1.8,1.9, 2., 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.,  3.1, 3.2, 3.3, 3.4, 3.5,3.6, 3.7, 3.8])
# plt.xticks([1.8, 2., 2.2, 2.4, 2.6, 2.8, 3., 3.2, 3.4,3.6, 3.8])
# plt.show()



"""
EXERCICE 6
"""

def comb(n,k):
    if k <=n:
        return math.factorial(n)/(math.factorial(n-k)*math.factorial(k))
    else: 
        return 0

n = 500
p = 0.031175

moyenne = n * p
ecart_type = math.sqrt(n*p*(1-p))

prob_zero_personne =((1-p)**n) 
prob_au_moins_un = 1 - prob_zero_personne

prob_max_3 = sum(comb(n, k) * (p**k) * ((1-p)**(n-k)) for k in range(4))

prob_X_250 = sum(comb(n, k) * (p**k) * ((1-p)**(n-k)) for k in range(250))
prob_X_50 = sum(comb(n, k) * (p**k) * ((1-p)**(n-k)) for k in range(50))
prob_X_sup_250 = 1 - prob_X_250
prob_X_sup_50 = 1 - prob_X_50

# print("moyenne = ", moyenne)
# print("ecart type = ",ecart_type)

# print("La probabilité qu'au moins une personne fasse sonner le portique est :", round(prob_au_moins_un, 4), "en arrondissant au millième près")
# print("La probabilité qu'au maximum 3 personnes fassent sonner le portique est :", round(prob_max_3, 4))
# print("La probabilité que X > 250 est :", round(prob_X_sup_250, 4))
# print("La probabilité que X > 50 est :", round(prob_X_sup_50, 4))




"""
EXERCICE 7
"""



mu = 360
sigma = 6  # La racine carrée de la variance, car la variance est σ^2


def fonction_densite_proba(y):
    return 1 / (np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(y - mu)**2 / (2 * sigma**2))


y_values = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
pdf_values = fonction_densite_proba(y_values)

Y = norm(mu, sigma)

P_au_moins_345 = Y.sf(345)  # P(Y >= 345)
P_plus_de_28_rates = Y.cdf(371)  # P(Y <= 372)
op_rate_max = Y.ppf(0.99)

# print("Probabilité de réussir au moins 345 opérations:", P_au_moins_345)
# print("Probabilité de rater plus de 28 opérations:", P_plus_de_28_rates)
# print("Nombre d'opération ratées que l'assurance accepte de couvrir: ", 400-op_rate_max)

# plt.plot(y_values, pdf_values)
# plt.xlabel('Y')
# plt.ylabel('Densité de probabilité')
# plt.title('Courbe de densité de probabilité de Y')
# plt.show()




"""    Exercice 8    """

donnees = [54.8, 55.4, 57.7, 59.6, 60.1, 61.2, 62.0, 63.1, 63.5, 64.2, 65.2, 65.4, 65.9, 66.0, 67.6, 68.1, 69.5, 70.6, 71.5, 73.4, 75.0, 75.2]

# 1. moyenne et écart-type
moyenne = mean(donnees)
ecart_type = np.std(donnees)  

#print("La moyenne de l'échantillon est de :", moyenne)
#print("L'écart-type de l'échantillon est de :", ecart_type)

# 2. histogramme
plt.hist(donnees, color='orange', edgecolor='black')  
plt.title('Histogramme des niveaux de bruit')
plt.xlabel('Niveau de bruit en dB')
plt.ylabel('Fréquence')
plt.show()

# 4.(a) Estimation de la moyenne
moy_est = moyenne

# 4.(a) Estimation de l'écart-type sans ajustement sans biais
ecart_type_est = ecart_type

#print(" estimation moyenne :", moy_est)
#print(" estimation écart-type :", ecart_type_est)

# 4.(c) - Intervalle de confiance pour la moyenne avec 95%

# Niveau de confiance
confiance = 0.95

# Degrés de liberté
df = len(donnees) - 1

# Valeur critique de la distribution de Student
t_critique = t.ppf(1 - (1 - confiance) / 2, df)

# Intervalle de confiance
borne_inferieure = moy_est - t_critique * (ecart_type_est / np.sqrt(len(donnees)))
borne_superieure = moy_est + t_critique * (ecart_type_est / np.sqrt(len(donnees)))

#print("Intervalle de confiance à", confiance * 100, "% pour la moyenne :", (borne_inferieure, borne_superieure))


# 4.(d) densité de proba
val_x = np.linspace(min(donnees), max(donnees))
densite_proba = norm.pdf(val_x, moy_est, ecart_type_est)

# nouveau histograme avec courbe de densité de proba
plt.hist(donnees, color='orange',edgecolor='black',density=True)
plt.plot(val_x, densite_proba, label='Densité de probabilité')
plt.title('Histogramme et Densité de probabilité de X')
plt.xlabel('Niveau de bruit (dB)')
plt.ylabel('Fréquence / Densité de proba')
plt.legend()
plt.show()

# 4.(e) Estimation de la probabilité dépasse 70 dB
sigma = np.std(donnees, ddof=1) # ddof ajuste les degrés de liberté
# Calcul de la probabilité P(X > 70)
probabilite_depasse_70 = 1 - norm.cdf(70, loc=moy_est, scale= sigma)

#print("Probabilité que le niveau de bruit dépasse 70 dB :", probabilite_depasse_70)


# 4.(f) Estimation de la probabilité que le niveau de bruit est entre 60 db et 75 db

# Calcul de la probabilité P(60 <= X <= 75)
probabilite_intervalle = norm.cdf(75, loc=moy_est, scale= sigma) - norm.cdf(60, loc=moy_est, scale= sigma)

#print("Probabilité que le niveau de bruit soit entre 60 dB et 75 dB :", probabilite_intervalle)


# 4.(g) Déterminer t1 tel que P(X < t1) = 0.95

t1 = norm.ppf(0.95, moy_est, sigma)
#print("t1 tel que P(X < t1) = 0.95 :", t1)

# 4.(g) Déterminer t2 tel que P(X ≥ t2) = 0.25
t2 = norm.ppf(1 - 0.25, moy_est, sigma)
#print("t2 tel que P(X ≥ t2) = 0.25 :", t2)



'''Exercice 9'''

poids = ["< 55", "55-57", "57-59", "59-61", "61-63", "> 63"]
nb_oeufs = [12, 12, 15, 18, 20, 23]

# 1)
def histogramme(x, y):
    plt.bar(x, y, color='blue', alpha=0.7)
    plt.title("Distribution du poids des œufs")
    plt.xlabel("Poids en grammes")
    plt.ylabel("Nombre d'œufs")
    plt.show()

# 2)
poids_centre = np.array([55, 56, 58, 60, 62, 63.5])
nombre_oeufs = np.array([12, 12, 15, 18, 20, 23])

moyenne = np.sum(poids_centre * nombre_oeufs) / np.sum(nombre_oeufs)
variance = np.sum(nombre_oeufs * (poids_centre - moyenne)**2) / np.sum(nombre_oeufs)
ecart_type = np.sqrt(variance)

# 3)
def test_normalité():

    # Effectuer le test de normalité
    stat, p_value = shapiro(poids_centre)
    
    
    print(f"Statistique de test : {stat}")
    
    # Interprétation des résultats
    if p_value > 0.05:
        print("La distribution peut être modélisée par une loi normale.")
    else:
        print("La distribution ne suit pas une loi normale.")


# histogramme(poids, nb_oeufs)
# print(f"Moyenne : {moyenne}")
# print(f"Écart-type : {ecart_type}")  
# test_normalité()
    


"""
EXERCICE 10
"""

televisions = pd.read_csv('televisions.dat', delimiter='\t')
 

resumé_numerique = televisions.describe()
 

def dispersion_tv_phys(data):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['teleratio'], data['physratio'])
    plt.title('Diagramme de dispersion entre teleratio et physratio')
    plt.xlabel('Nombre de personnes par télévision')
    plt.ylabel('Nombre de personnes par physicien')
    plt.show()

def histogrammes_tv_phys(data):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.hist(data['teleratio'], bins=15)
    plt.title('Distribution de teleratio')
    plt.xlabel('Nombre de personnes par téléviseur')
    plt.ylabel('Fréquence')

    plt.subplot(122)
    plt.hist(data['physratio'], bins=15)
    plt.title('Distribution de physratio')
    plt.xlabel('Nombre de personnes par physicien')
    plt.ylabel('Fréquence')

    plt.tight_layout()
    plt.show()

def boites_tv_phys(data):
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    sns.boxplot(x=data['teleratio'])
    plt.title('Diagramme en boîte de teleratio')
    plt.xlabel('Nombre de personnes par téléviseur')

    plt.subplot(122)
    sns.boxplot(x=data['physratio'])
    plt.title('Diagramme en boîte de physratio')
    plt.xlabel('Nombre de personnes par physicien')
    
    plt.tight_layout()
    plt.show()


def histogrammes_esperances_vie(data):
    plt.figure(figsize=(18, 6))
    plt.subplot(131)
    sns.histplot(data['espvie'], kde=True) 
    plt.title("Histogramme de l'Espérance de Vie")

    plt.subplot(132)
    sns.histplot(data['espvieF'], kde=True) 
    plt.title("Histogramme de l'Espérance de Vie des Femmes")

    plt.subplot(133)
    sns.histplot(data['espvieH'], kde=True) 
    plt.title("Histogramme de l'Espérance de Vie des Hommes")

    plt.tight_layout()
    plt.show()

def boites_esperances_vie(data):    
    plt.figure(figsize=(18, 6))
    plt.subplot(131)
    sns.boxplot(x=data['espvie'])
    plt.title('Boîte - Espérance de Vie')

    plt.subplot(132)
    sns.boxplot(x=data['espvieF'])
    plt.title('Boîte - Espérance de Vie des Femmes')

    plt.subplot(133)
    sns.boxplot(x=data['espvieH'])
    plt.title('Boîte - Espérance de Vie des Hommes')

    plt.tight_layout()
    plt.show()

def relations(data, obj1, obj2):
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.scatter(data[obj1], data[obj2])
    plt.title(f'Représentation entre {obj1} et {obj2}')
    plt.xlabel(obj1)
    plt.ylabel(obj2)

    log_obj1 = np.log(data[obj1])

    plt.subplot(122)
    plt.scatter(log_obj1, data[obj2])
    plt.title(f'Représentation entre {obj2} et log({obj1})')
    plt.xlabel(f'Log({obj1})')
    plt.ylabel(obj2)

    plt.tight_layout()
    plt.show()
 


print(resumé_numerique)

dispersion_tv_phys(televisions)
histogrammes_tv_phys(televisions)
boites_tv_phys(televisions)

histogrammes_esperances_vie(televisions)
boites_esperances_vie(televisions)

relations(televisions, 'teleratio', 'espvie')
relations(televisions, 'physratio', 'espvie')








