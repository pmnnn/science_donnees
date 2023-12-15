import statistics as stat
import numpy as np
import matplotlib.pyplot as plt
import csv


'''PARTIE 1 EX2'''
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


"""
EXERCICE4
"""


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




"""
EXERCICE 5
"""


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















