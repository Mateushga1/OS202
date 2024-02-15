# TD n°3 - parallélisation du Bucket Sort

*Ce TD peut être réalisé au choix, en C++ ou en Python*

Implémenter l'algorithme "bucket sort" tel que décrit sur les deux dernières planches du cours n°3 :

- le process 0 génère un tableau de nombres arbitraires,
- il les dispatch aux autres process,
- tous les process participent au tri en parallèle,
- le tableau trié est rassemblé sur le process 0.


# Analyse des résultats obtenus

Temps de calcul et speedup pour différents nombres de processus pour une tableau de 1.000.000 d'éléments:
- 1 tâche (sans parallélisation): 2.624. Speedup: 1
- 2 tâches: 1.435. Speedup: 1.83
- 3 tâches: 1.114. Speedup: 2.35
- 4 tâches: 0.9664. Speedup: 2.71
