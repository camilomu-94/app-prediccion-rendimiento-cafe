# clr_model.py
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score

class Fitness():
  def __init__(self, k, random_state, max_depth, min_samples_split, criterion,
  alpha, l1_ratio):
    self.k = k
    self.random_state = random_state
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.criterion = criterion
    self.alpha = alpha
    self.l1_ratio = l1_ratio

  def fit_elastic_net(self, X, y):
      elastic_net = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                               random_state=self.random_state, max_iter=10000)
      elastic_net.fit(X, y)
      return elastic_net

  def fit_decision_tree(self, X, y):
      decision_tree = DecisionTreeClassifier(random_state=self.random_state,
                      max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                      criterion=self.criterion)
      decision_tree.fit(X, y)
      return decision_tree

  def calculate(self, X, y, X_scaled):
    kmeans = KMeans(n_clusters=self.k, init='k-means++', random_state=self.random_state)
    kmeans.fit(X_scaled)
    y_clusters = kmeans.labels_
    clusters = np.unique(y_clusters)
    k_real = len(clusters)

    reg_models = [None] * k_real  # Create a list of size k with None values

    X2 = np.hstack((X, y.reshape(-1, 1)))           # Concatenar los datos originales con
    X2 = np.hstack((X2, y_clusters.reshape(-1, 1))) # el target y los clusters asignadas

    # Ajustar el modelo de Elastic Net para cada cluster
    for this_cluster in clusters:
        # Filtrar los datos por el cluster actual
        temp = X2[X2[:, -1] == this_cluster]
        temp = temp[:, :-1] # Quito la última columna, la del cluster
        X_this_cluster = temp[:, :-1] # Quito la columna de y_real = target
        y_this_cluster = temp[:, -1] # target

        # Ajustar el modelo de Elastic Net a los datos de este cluster
        reg_models [this_cluster] = self.fit_elastic_net(X_this_cluster, y_this_cluster) # Guardar el modelo entrenado para este cluster

    # Árbol de Decisión
    dt_model = self.fit_decision_tree(X, y_clusters)
    y_classified = dt_model.predict(X)
    classes = np.unique(y_classified)

    X2 = np.hstack((X, y.reshape(-1, 1)))             # Concatenar los datos originales con
    X2 = np.hstack((X2, y_classified.reshape(-1, 1))) # el target y las clases asignadas
    y_target_real = []
    y_target_predicted = []

    # Recorrer los clusters del dataset de entrenamiento train ----------- cambiar todo lo que diga test a train!!!!!!!! en fit
    for this_class in classes:
        # Filtrar los datos por la clase actual
        temp = X2[X2[:, -1] == this_class]
        y_this_class = temp[:, -1]
        temp = temp[:, :-1]
        X_this_class = temp[:, :-1]
        y_this_target_real = temp[:, -1]

        y_this_target_predicted = reg_models[this_class].predict(X_this_class)
        y_target_real = np.concatenate([y_target_real, y_this_target_real])
        y_target_predicted = np.concatenate([y_target_predicted, y_this_target_predicted])

    r2 = r2_score(y_target_real, y_target_predicted) # metrica de calidad, AQUI UNE SIN IMPORTAR EL ORDEN INCIAL LOS VALORES REALES DE yield por la etiqueta de grupo
                                                     # se asegura que tanto reales como predichos esten correctamente concatenados y seán comparables
    #return [r2, dt_model, reg_models]
    return r2, dt_model, reg_models

class CLR(BaseEstimator, ClassifierMixin):
  def __init__(self, max_clusters=8, random_state=42, max_depth=5,
               min_samples_split=2, criterion="gini", alpha=1.0, l1_ratio=0.5, repetitions=10):
    self.max_clusters = max_clusters
    self.random_state = random_state
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.criterion = criterion
    self.alpha = alpha
    self.l1_ratio = l1_ratio
    self.repetitions = repetitions

  def fit (self, X, y):
    standard_scaler = StandardScaler()
    X_scaled = standard_scaler.fit_transform(X)

    k_max = min(int(np.sqrt(len(y) / 2)), self.max_clusters)
    best_r2 = -float('inf')

    for k in range(2, k_max + 1):
        for r in range (0, self.repetitions):
          ##### paralelizar
            fitness = Fitness(k=k, random_state= self.random_state+r,
                              max_depth=self.max_depth,
                              min_samples_split=self.min_samples_split,
                              criterion = self.criterion,
                              alpha = self.alpha, l1_ratio=self.l1_ratio)
            r2, dt_model, reg_models  = fitness.calculate(X, y, X_scaled)
            if r2 >= best_r2:
              best_r2 = r2
              best_dt_model = dt_model
              best_reg_models = reg_models
              best_k = k
              best_r = r

    #print(f"Best R�: {mejor_r2} for k={best_K} and seed r={best_R} and m={best_m}")
    self.best_r2 = best_r2
    self.best_dt_model = best_dt_model
    self.best_reg_models = best_reg_models
    self.best_k = best_k
    self.best_r = best_r

  def predict(self, X):
    y_classes_predicted = self.best_dt_model.predict(X) # predice la clase con el arbol de decision
    results = []  # Crear una lista para almacenar los resultados
    num_rows = X.shape[0]
    for i in range(num_rows):  # Iterar sobre cada registro de X
        y_class_predicted = int(y_classes_predicted[i]) # Obtener la etiqueta de la clase
        model = self.best_reg_models[y_class_predicted]  # Seleccionar el modelo correspondiente
        y_target_predicted = model.predict(X[i].reshape(1, -1))  # Realizar la predicción
        results.append(y_target_predicted)  # Adicionar el resultado a la lista
    return np.array(results)

  def score(self, X, y):
    y_predicted = self.predict(X)
    r2 = r2_score(y, y_predicted)
    return r2