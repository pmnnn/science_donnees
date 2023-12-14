import statistics as stat
import numpy as np
import matplotlib.pyplot as plt


def boite_dispersion(serie, titre="Boîte de Dispersion"):

    fig, ax = plt.subplots()
    ax.boxplot(serie)
    ax.set_title(titre)
    ax.set_ylabel("Valeurs")
    plt.show()

def  tracer_histogramme(liste_longueurs, titre="Histogramme des longueurs de la rectrice centrale"):
    min_BonasaUmbellus = min(liste_longueurs)
    max_BonasaUmbellus = max(liste_longueurs)
    plt.hist(liste_longueurs, bins = max_BonasaUmbellus-min_BonasaUmbellus, alpha = 0.5, color = 'blue', edgecolor = 'black')

    plt.title(titre)
    plt.xlabel('Longueur (cm)')
    plt.ylabel('Effectif')

    plt.show()
    

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


print("La valeur longueur minimun de l'échantillon est de ", min_BonasaUmbellus, "cm")
print("La valeur longueur maximun de l'échantillon est de ", max_BonasaUmbellus, "cm")
print("L'étendue de l'échantillon est de ", etendue_BonasaUmbellus, "cm")
print("La moyenne de l'échantillon est de ", moyenne_BonasaUmbellus, "cm")
print("La médiane de l'échantillon est de ", mediane_BonasaUmbellus, "cm")
print("Voici les quartiles de l'échantillon: \n -Q1 =", quartiles_BonasaUmbellus[0], "\n -Q2 =", quartiles_BonasaUmbellus[1], "\n -Q3 =", quartiles_BonasaUmbellus[2], "\n -Q4 =", quartiles_BonasaUmbellus[3])
print("Variance = ", variance_BonasaUmbellus)
print("Ecart type = ", ecart_type_BonasaUmbellus)

boite_dispersion(BonasaUmbellus)
tracer_histogramme(BonasaUmbellus)









