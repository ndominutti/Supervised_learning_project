import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import recall_score, fbeta_score, precision_score, confusion_matrix
from typing import Union
import matplotlib.pyplot as plt
import seaborn as sns


class AuxTargetEncoding(BaseEstimator, TransformerMixin):
    def __init__(self):
        """Clase utilizada para hacer Target Encoding dentro de GridSearch/RandomizedSearch. La idea principal radica en que el Target Encoding
        debe hacerce en cada fold de train (fit_transform) para luego ser traspasado a test (transform), para ello sklearn necesita el formato
        de esta clase.
        En cuanto a la lógica de encodeo, se utiliza un Mean Encoding que se podría representar como E(y|Y=y) para todo j perteneciente a J, donde
        J es el conjunto de variables e "y" es el valor que toma el target.
        Luego de realizar el proceso se eliminan todas las columnas, salvo las del encoding.
        """
        ...

    def fit_transform(self, X: pd.DataFrame, Y: np.array) -> pd.DataFrame:
        """Se encarga de realizar el proceso de Target Encoding para las folds de train.

        Args:
            X (pd.DataFrame): dataframe que se quiere encodear.
            Y (np.array): vector de respuesta

        Returns:
            pd.DataFrame: dataframe encodeado
        """
        df = pd.concat([X, Y], axis=1)
        df = df.rename(columns={0: "label"})
        te = df.groupby("label").mean().reset_index()
        self.te = te
        dataframe = X.copy()

        for col in te.columns[1:]:
            dataframe[f"TE_{col}_label_1"] = te.loc[te["label"] == 1, col].values[0]
            dataframe[f"TE_{col}_label_0"] = te.loc[te["label"] == 0, col].values[0]

            dataframe[f"{col}_vs_TE_label1"] = np.round(
                dataframe[col] / dataframe[f"TE_{col}_label_1"], 3
            )
            dataframe[f"{col}_vs_TE_label0"] = np.round(
                dataframe[col] / dataframe[f"TE_{col}_label_0"], 3
            )

            dataframe[f"{col}_vs_TE_label1"] = np.where(
                dataframe[f"{col}_vs_TE_label1"] == np.inf,
                0,
                dataframe[f"{col}_vs_TE_label1"],
            )
            dataframe[f"{col}_vs_TE_label0"] = np.where(
                dataframe[f"{col}_vs_TE_label0"] == np.inf,
                0,
                dataframe[f"{col}_vs_TE_label0"],
            )

        dataframe = dataframe.drop(columns=[f"TE_{col}_label_1", f"TE_{col}_label_0"])
        dataframe = dataframe[[col for col in dataframe if "TE" in col]]
        self.columns = dataframe.columns
        return dataframe

    def transform(self, X: pd.DataFrame, Y: np.array = None) -> pd.DataFrame:
        """Se encarga de realizar el proceso de Target Encoding para las folds de test.

        Args:
            X (pd.DataFrame): dataframe que se quiere encodear.
            Y (np.array, optional): vector de respuesta. Defaults to None.

        Returns:
            pd.DataFrame: dataframe encodeado
        """
        te = self.te
        dataframe = X.copy()
        for col in te.columns[1:]:
            dataframe[f"TE_{col}_label_1"] = te.loc[te["label"] == 1, col].values[0]
            dataframe[f"TE_{col}_label_0"] = te.loc[te["label"] == 0, col].values[0]

            dataframe[f"{col}_vs_TE_label1"] = np.round(
                dataframe[col] / dataframe[f"TE_{col}_label_1"], 3
            )
            dataframe[f"{col}_vs_TE_label0"] = np.round(
                dataframe[col] / dataframe[f"TE_{col}_label_0"], 3
            )

            dataframe[f"{col}_vs_TE_label1"] = np.where(
                dataframe[f"{col}_vs_TE_label1"] == np.inf,
                0,
                dataframe[f"{col}_vs_TE_label1"],
            )
            dataframe[f"{col}_vs_TE_label0"] = np.where(
                dataframe[f"{col}_vs_TE_label0"] == np.inf,
                0,
                dataframe[f"{col}_vs_TE_label0"],
            )

        dataframe = dataframe.drop(columns=[f"TE_{col}_label_1", f"TE_{col}_label_0"])
        return dataframe[[col for col in dataframe if "TE" in col]]


class AuxTargetEncodingNoDrop(BaseEstimator, TransformerMixin):
    def __init__(self):
        """Clase utilizada para hacer Target Encoding dentro de GridSearch/RandomizedSearch. La idea principal radica en que el Target Encoding
        debe hacerce en cada fold de train (fit_transform) para luego ser traspasado a test (transform), para ello sklearn necesita el formato
        de esta clase.
        En cuanto a la lógica de encodeo, se utiliza un Mean Encoding que se podría representar como E(y|Y=y) para todo j perteneciente a J, donde
        J es el conjunto de variables e "y" es el valor que toma el target.
        Luego de realizar el proceso NO se elimina columna alguna.
        """
        ...

    def fit_transform(self, X: pd.DataFrame, Y: np.array) -> pd.DataFrame:
        """Se encarga de realizar el proceso de Target Encoding para las folds de train.

        Args:
            X (pd.DataFrame): dataframe que se quiere encodear.
            Y (np.array): vector de respuesta

        Returns:
            pd.DataFrame: dataframe encodeado
        """
        df = pd.concat([X, Y], axis=1)
        df = df.rename(columns={0: "label"})
        te = df.groupby("label").mean().reset_index()
        self.te = te
        dataframe = X.copy()

        for col in te.columns[1:]:
            dataframe[f"TE_{col}_label_1"] = te.loc[te["label"] == 1, col].values[0]
            dataframe[f"TE_{col}_label_0"] = te.loc[te["label"] == 0, col].values[0]

            dataframe[f"{col}_vs_TE_label1"] = np.round(
                dataframe[col] / dataframe[f"TE_{col}_label_1"], 3
            )
            dataframe[f"{col}_vs_TE_label0"] = np.round(
                dataframe[col] / dataframe[f"TE_{col}_label_0"], 3
            )

            dataframe[f"{col}_vs_TE_label1"] = np.where(
                dataframe[f"{col}_vs_TE_label1"] == np.inf,
                0,
                dataframe[f"{col}_vs_TE_label1"],
            )
            dataframe[f"{col}_vs_TE_label0"] = np.where(
                dataframe[f"{col}_vs_TE_label0"] == np.inf,
                0,
                dataframe[f"{col}_vs_TE_label0"],
            )

        dataframe = dataframe.drop(columns=[f"TE_{col}_label_1", f"TE_{col}_label_0"])
        self.columns = dataframe.columns
        return dataframe

    def transform(self, X: pd.DataFrame, Y: np.array = None) -> pd.DataFrame:
        """Se encarga de realizar el proceso de Target Encoding para las folds de test.

        Args:
            X (pd.DataFrame): dataframe que se quiere encodear.
            Y (np.array, optional): vector de respuesta. Defaults to None.

        Returns:
            pd.DataFrame: dataframe encodeado
        """
        te = self.te
        dataframe = X.copy()
        for col in te.columns[1:]:
            dataframe[f"TE_{col}_label_1"] = te.loc[te["label"] == 1, col].values[0]
            dataframe[f"TE_{col}_label_0"] = te.loc[te["label"] == 0, col].values[0]

            dataframe[f"{col}_vs_TE_label1"] = np.round(
                dataframe[col] / dataframe[f"TE_{col}_label_1"], 3
            )
            dataframe[f"{col}_vs_TE_label0"] = np.round(
                dataframe[col] / dataframe[f"TE_{col}_label_0"], 3
            )

            dataframe[f"{col}_vs_TE_label1"] = np.where(
                dataframe[f"{col}_vs_TE_label1"] == np.inf,
                0,
                dataframe[f"{col}_vs_TE_label1"],
            )
            dataframe[f"{col}_vs_TE_label0"] = np.where(
                dataframe[f"{col}_vs_TE_label0"] == np.inf,
                0,
                dataframe[f"{col}_vs_TE_label0"],
            )

        dataframe = dataframe.drop(columns=[f"TE_{col}_label_1", f"TE_{col}_label_0"])
        return dataframe


def threshold_optimization(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model: object,
    skf: object,
    min_aceptable_recall: int = 0.7,
) -> Union[pd.DataFrame, float]:

    """Funcion encargada de seleccionar un threshold que boostee el fbeta score con un beta=.5. La idea es realizar este proceso
    solo sobre la data de train para no overfittear test. Además, para no dejar de lado el recall, se acepta solo aquel threshold cuyo
    score de recall supere el valor del argumento  min_aceptable_recall


    Args:
        X_train (pd.DataFrame): dataframe de train sin el target
        y_train (pd.Series): target del dataframe de train
        model (object): modelo pre-entrenado que permita realizar predict_proba
        skf (object): instancia de StratifiedKFold
        min_aceptable_recall (int, optional): valor mínimo de recall aceptado para el threshold. Defaults to 0.7.

    Returns:
        Union[pd.DataFrame, float]: devuelve el dataframe de mejores scores de fbeta_score (que hayan superado el min_aceptable_recall),
        ordenados en orden descendientes. Además devuelve el número óptimo para el threshold según la data de train.
    """
    # probamos valores entre .45 y .8
    threshold = np.arange(0.45, 0.8, 0.01)
    results = {
        "trheshold": [],
        "fbeta_score": [],
        "recall_score": [],
        "precision_score": [],
    }
    for i in range(len(threshold)):
        aux_fbeta = []
        aux_recall = []
        aux_precision = []
        for train_index, test_index in skf.split(X_train, y_train):
            train = X_train.iloc[train_index, :]
            train_label = y_train.reindex(train.index)

            test = X_train.iloc[test_index, :]
            test_label = y_train.reindex(test.index)

            y_hat = model.predict_proba(test)
            # si la probabilidad de que sea positivo es mayor al threshold lo marcamos
            # como positivo (buscamos disminuír la cantidad de FP)
            y_hat = np.where(y_hat[:, 1] > threshold[i], 1, 0)
            aux_fbeta.append(fbeta_score(test_label, y_hat, beta=0.5))
            aux_recall.append(recall_score(test_label, y_hat))
            aux_precision.append(precision_score(test_label, y_hat, zero_division=0))

        # Si el recall es menor al min_aceptable_recall, descartamos estre threshold
        if (np.mean(aux_recall) > min_aceptable_recall) & (np.mean(aux_precision) > 0):
            results["trheshold"].append(threshold[i])
            results["fbeta_score"].append(np.mean(aux_fbeta))
            results["precision_score"].append(np.mean(aux_precision))
            results["recall_score"].append(np.mean(aux_recall))
    results = pd.DataFrame(results).sort_values(by="fbeta_score", ascending=False)
    top_threshold = results.iloc[0, 0]

    return results, top_threshold


def guardar_errores(nombre_modelo: str, errores: np.array,path="../data/error_analysis.csv") -> None:
    """Función para generar .csv que permita trackear los errores de los modelos.

    Args:
        nombre_modelo (str): nombre a asignarle al modelo
        errores (np.array): array con los ID en los que el modelo falló
    """
    temp = pd.DataFrame({"modelo": [nombre_modelo], "errores": [errores.tolist()]})
    df = pd.read_csv(path)
    df = pd.concat([df, temp])
    try:
        df = pd.read_csv(path)
        if nombre_modelo in df.modelo.unique():
            df.loc[df[df["modelo"] == nombre_modelo].index, "errores"] = [
                errores.tolist()
            ]
        else:
            df = pd.concat([df, temp])
    except:
        print("No se encontró el archivo, creando...")
        df = temp

    df.to_csv(path, index=False)


def guardar_aciertos(nombre_modelo: str, aciertos: np.array, path="../data/aciertos_analysis.csv") -> None:
    """Función para generar .csv que permita trackear los aciertos de los modelos.

    Args:
        nombre_modelo (str): nombre a asignarle al modelo
        aciertos (np.array): array con los ID que el modelo clasificó bin
    """
    temp = pd.DataFrame({"modelo": [nombre_modelo], "aciertos": [aciertos.tolist()]})
    df = pd.read_csv(path)
    df = pd.concat([df, temp])
    try:
        df = pd.read_csv(path)
        if nombre_modelo in df.modelo.unique():
            df.loc[df[df["modelo"] == nombre_modelo].index, "aciertos"] = [
                aciertos.tolist()
            ]
        else:
            df = pd.concat([df, temp])
    except:
        print("No se encontró el archivo, creando...")
        df = temp

    df.to_csv(path, index=False)


def guardar_metricas(nombre_modelo: str, y_test, y_hat, path="../data/comparacion_modelos.csv") -> None:
    """Función para generar .csv que permita trackear las métricas de los modelos.

    Args:
        nombre_modelo (str): nombre a asignarle al modelo
    """
    cm = confusion_matrix(y_test, y_hat)
    temp = pd.DataFrame(
        {
            "modelo": [nombre_modelo],
            "fbeta(beta=.4)": [fbeta_score(y_test, y_hat, beta=0.4)],
            "Precision": [precision_score(y_test, y_hat)],
            "Recall": [recall_score(y_test, y_hat)],
            "TN": [cm[0][0]],
            "FP": [cm[0][1]],
            "FN": [cm[1][0]],
            "TP": [cm[1][1]],
        }
    )
    df = pd.read_csv(path)
    df = pd.concat([df, temp])
    try:
        df = pd.read_csv(path)
        if nombre_modelo in df.modelo.unique():
            df.loc[df[df["modelo"] == nombre_modelo].index, :] = temp.iloc[
                0, :
            ].values.tolist()
        else:
            df = pd.concat([df, temp])
    except:
        print("No se encontró el archivo, creando...")
        df = temp

    df.to_csv(path, index=False)


def discretizacion_recomendacion(probabilidad: float, top_threshold: float) -> str:
    """Discretiza en cuatro categorias que tan probable es que la recomendacion sea mas confiable segun su probabilidad

    Args:
        row (float): probabilidad de que esa cancion le guste al usuario
        top_threshold(float): controla a partir de dónde el modelo decreta una canción como NO RECOMENDABLE

    Returns:
        str: discretizacion del nivel de recomendacion de la cancion evaluada, segun los gustos del usuario
    """
    if probabilidad > 0.9:
        return "HOT SPOT"
    if probabilidad > 0.7 and probabilidad <= 0.9:
        return "MUY RECOMENDABLE"
    if probabilidad > 0.6 and probabilidad <= 0.7:
        return "RECOMENDABLE"
    if probabilidad > top_threshold and probabilidad <= 0.6:
        return "POSIBLE"
    else:
        return "NO RECOMENDABLE"


def plot_matriz_confusion(y_test, y_hat, title):
    plt.figure(figsize=(8, 5))
    sns.heatmap(
        confusion_matrix(y_test, y_hat),
        annot=True,
        fmt="4d",
        cmap="Blues",
    )
    plt.xlabel("VALORES PREDICHOS")
    plt.ylabel("VALORES REALES")
    plt.title(title)
    plt.show()
