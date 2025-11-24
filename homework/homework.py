import gzip
import json
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns="ID")
    df = df.dropna()
    df = df.rename(columns={"default payment next month": "default"})
    df = df[(df["SEX"] != 0) & (df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]
    df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4
    return df


def definir_transformador():
    columnas_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
    columnas_numericas = [
        "LIMIT_BAL", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]

    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(dtype='int', handle_unknown='ignore'), columnas_categoricas),
            ('num', StandardScaler(), columnas_numericas)
        ],
        remainder="passthrough"
    )


def construir_pipeline():
    pasos_pipeline = [
        ('preprocesamiento', definir_transformador()),
        ('seleccion_caracteristicas', SelectKBest(score_func=f_classif)),
        ('reduccion_dim', PCA()),
        ('clasificador', MLPClassifier(max_iter=15000, random_state=42))
    ]
    return Pipeline(pasos_pipeline)


def definir_busqueda(pipeline, cv=10):
    parametros = {
        'reduccion_dim__n_components': [None],
        'seleccion_caracteristicas__k': [20],
        'clasificador__hidden_layer_sizes': [(50, 30, 40, 60)],
        'clasificador__alpha': [0.28],
        'clasificador__learning_rate_init': [0.001]
    }

    return GridSearchCV(
        estimator=pipeline,
        param_grid=parametros,
        scoring='balanced_accuracy',
        cv=cv,
        refit=True
    )


def guardar_modelo(modelo):
    ruta = "files/models"
    os.makedirs(ruta, exist_ok=True)
    with gzip.open(os.path.join(ruta, "model.pkl.gz"), "wb") as archivo:
        pickle.dump(modelo, archivo)


def calcular_metricas(y_verdad, y_pred, tipo):
    return {
        "type": "metrics",
        "dataset": tipo,
        "precision": precision_score(y_verdad, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_verdad, y_pred),
        "recall": recall_score(y_verdad, y_pred),
        "f1_score": f1_score(y_verdad, y_pred)
    }


def matriz_confusion_personalizada(y_real, y_estimado, tipo):
    tn, fp, fn, tp = confusion_matrix(y_real, y_estimado).ravel()
    return {
        "type": "cm_matrix",
        "dataset": tipo,
        "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
        "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)}
    }


def guardar_metricas(lista_metricas):
    ruta = "files/output"
    os.makedirs(ruta, exist_ok=True)
    with open(os.path.join(ruta, "metrics.json"), "w") as f:
        for m in lista_metricas:
            f.write(json.dumps(m) + "\n")


# Carga y procesamiento de datos
df_entrenamiento = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
df_prueba = pd.read_csv("files/input/test_data.csv.zip", compression="zip")

x_train = limpiar_datos(df_entrenamiento)
x_test = limpiar_datos(df_prueba)

X_train, y_train = x_train.drop(columns="default"), x_train["default"]
X_test, y_test = x_test.drop(columns="default"), x_test["default"]

# Construcción y entrenamiento del modelo
modelo = construir_pipeline()
buscador = definir_busqueda(modelo)
buscador.fit(X_train, y_train)

# Guardar modelo
guardar_modelo(buscador)

# Evaluación
train_pred = buscador.predict(X_train)
test_pred = buscador.predict(X_test)

metricas_train = calcular_metricas(y_train, train_pred, "train")
metricas_test = calcular_metricas(y_test, test_pred, "test")
matriz_train = matriz_confusion_personalizada(y_train, train_pred, "train")
matriz_test = matriz_confusion_personalizada(y_test, test_pred, "test")

# Guardar métricas
guardar_metricas([metricas_train, metricas_test, matriz_train, matriz_test])