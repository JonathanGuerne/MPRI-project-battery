# MPRI-project-battery
final MPRI project 

## HMM classifier 

analyse de la variance des features pour voir leur pertiance :
```
Feature : charge_nb variance of 3380.037504869084
Feature : voltage_measured variance of 0.08091295447782859
Feature : current_measured variance of 1.2593335117769673
Feature : temperature_measured variance of 22.893801758680418
Feature : current_charge variance of 0.8663792149241187
Feature : voltage_charge variance of 2.2496041828303306
Feature : ambiant_temp variance of 0.0
```
On remarque que ambiant_temp a une variance de 0.0 donc on peut l'enlever.


### cération de modèles 

On va créer un modèle par qualité de batterie dans l'ensemble d'entraînement. Pour extraire ces données on utilise pandas 
``` python
quality_1 = df.loc[df['quality' == 1]]
```

mais il faut ensuite faire attention a enlever la colonne quality des données qu'on va donner au HMM.

On regroupe ensuite les données par batterie pour par nombre de charge.

#### problèmes 

lors de l'extraction des données on rencontre très souvent des NaN qui vont empêcher le bon fonctionnement du HMM il faut trouver une solution pour s'en débarasser.

### Sampleing des données 
il y a beaucoup de données, il faut trouver une façon intelligente d'en extraire un sous ensemble 

Une approche décevante a été d'utiliser une fonction logarithmique pour donner plus d'importance aux premiers échantillons de la liste.

```python 
def log_sampeling(arr, nb_samples):
    _max = len(arr)-1 
    _exposant = 2.0 
    ids = np.round(np.power(np.linspace(0,1,nb_samples),_exposant) * _max)
    return arr[ids.astype(int)]
```

il est devenu clair (trop tard) que cette approce, bien que sur le papier intéressant fasse à la forme des données, perdait tout son sens quand les données étaient randomisées