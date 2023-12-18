import pandas as pd 
import statistics as stat
import numpy as np
import matplotlib.pyplot as plt
import csv


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

# Données d'exemple
T = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
V = np.array([5.098, 3.618, 2.581, 2.011, 1.486, 1.028, 0.845, 0.573, 0.429, 0.29, 0.2])

# Calcul de la covariance avec la fonction cov de NumPy
covariance_matrix = np.cov(T, V)

# La covariance entre x et y est le coefficient situé à la position (0, 1) ou (1, 0) dans la matrice de covariance
covariance_TV = covariance_matrix[0, 1]
covariance_T = np.cov(T)

a = covariance_TV /covariance_T
b = log(5.098)
tau = (-1)/ a
# print("a = :", a)
# print("b = :", b)
# print("tau = :", tau)

# Calcul des valeurs de x
X_ln = -T / tau

# Calcul des valeurs de y
Y_ln = np.log(V)

# Affichage des résultats
# print("Valeurs de X_ln :", X_ln)
# print("Valeurs de Y_ln :", Y_ln)


# Générer les valeurs prédites à partir de la régression linéaire
Y_pred = a * X_ln + b

# Affichage des résultats
plt.scatter(X_ln, Y_ln, label='Données')
plt.plot(X_ln, Y_pred, color='red', label='Régression linéaire')
plt.xlabel('X_ln')
plt.ylabel('Y_ln')
plt.legend()
plt.title('Régression Linéaire')
#plt.show()


t = 53  # temps en ms
V0 = 5.098  # Remplacez cette valeur par la tension initiale obtenue

# Calcul de V pour t = 53 ms
V = V0 * np.exp(-t / tau)

# Affichage du résultat
# print(f"La valeur de V pour t = {t} ms est : {V}")



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
for i in range(len(XPA)):
    #print(f"Valeur ajustée ŷ{XPA[i]} : {y_pred[i]:.2f}, Distance résiduelle : {residuals[i]:.2f}")

# 9.(b) valeurs ajustées x̂i et distances résiduelles
for i in range(len(YPI)):
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

print("EXERCICE 4:")
print("La valeur longueur minimun de l'échantillon est de ", min_BonasaUmbellus, "cm")
print("La valeur longueur maximun de l'échantillon est de ", max_BonasaUmbellus, "cm")
print("L'étendue de l'échantillon est de ", etendue_BonasaUmbellus, "cm")
print("La moyenne de l'échantillon est de ", moyenne_BonasaUmbellus, "cm")
print("La médiane de l'échantillon est de ", mediane_BonasaUmbellus, "cm")
print("Voici les quartiles de l'échantillon: \n -Q1 =", quartiles_BonasaUmbellus[0], "\n -Q2 =", quartiles_BonasaUmbellus[1], "\n -Q3 =", quartiles_BonasaUmbellus[2], "\n -Q4 =", quartiles_BonasaUmbellus[3])
print("Variance = ", variance_BonasaUmbellus)
print("Ecart type = ", ecart_type_BonasaUmbellus)

boite_dispersion(BonasaUmbellus)
tracer_histogramme(BonasaUmbellus, "Histogramme des longueurs de la rectrice centrale", 'Longuer(cm)', etendue_BonasaUmbellus)
plt.show()




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


print("EXERCICE 5:")
print("La moyenne de l'échantillon est de ", moyenne_S1, "kg")
print("Ecart type = ", ecart_type_S1)
print("m1 = ", m1)
print("m2 = ", m2)
print("mtot = ", mtot)

boite_dispersion(S1)
tracer_histogramme(S1, "Diagramme en bâtons des pesées des 30 bébés", "Poids en kg", [1.8,1.9, 2., 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.,  3.1, 3.2, 3.3, 3.4, 3.5,3.6, 3.7, 3.8])
plt.xticks([1.8, 2., 2.2, 2.4, 2.6, 2.8, 3., 3.2, 3.4,3.6, 3.8])
plt.show()















