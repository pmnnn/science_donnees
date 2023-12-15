import statistics as stat
import numpy as np
import matplotlib.pyplot as plt
import csv

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















