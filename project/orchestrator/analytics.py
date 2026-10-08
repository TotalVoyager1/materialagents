from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class AnalyticsResult:
    summary: str
    profile: pd.DataFrame
    clusters: pd.DataFrame | None
    chart: object | None
    classification: str | None
    forecast: pd.DataFrame | None


def analyze_csv(
    data: bytes,
    clusters: int = 3,
    target_column: str | None = None,
    forecast_column: str | None = None,
) -> AnalyticsResult:
    from io import BytesIO

    import matplotlib.pyplot as plt

    frame = pd.read_csv(BytesIO(data))
    profile = pd.DataFrame(
        {
            "Тип": frame.dtypes.astype(str),
            "Пропуски": frame.isna().sum(),
            "Уникальные": frame.nunique(),
        }
    )
    summary = (
        f"Строк: {len(frame)}; столбцов: {len(frame.columns)}; "
        f"полных дубликатов: {int(frame.duplicated().sum())}."
    )
    numeric = frame.select_dtypes(include="number").dropna()
    clustered = None
    chart = None
    if len(numeric) >= 2 and numeric.shape[1] >= 2:
        count = min(max(2, clusters), len(numeric))
        values = StandardScaler().fit_transform(numeric)
        labels = KMeans(n_clusters=count, random_state=42, n_init=10).fit_predict(values)
        clustered = numeric.assign(Кластер=labels)
        figure, axis = plt.subplots()
        axis.scatter(numeric.iloc[:, 0], numeric.iloc[:, 1], c=labels, cmap="viridis")
        axis.set_xlabel(str(numeric.columns[0]))
        axis.set_ylabel(str(numeric.columns[1]))
        axis.set_title("Кластеризация числовых данных (K-means)")
        chart = figure
    classification = None
    if target_column and target_column in frame.columns:
        labeled = frame.dropna(subset=[target_column])
        features = labeled.select_dtypes(include="number").drop(
            columns=[target_column], errors="ignore"
        ).dropna()
        labels = labeled.loc[features.index, target_column]
        if features.shape[1] and labels.nunique() > 1 and len(features) >= 10:
            try:
                x_train, x_test, y_train, y_test = train_test_split(
                    features,
                    labels,
                    test_size=0.25,
                    random_state=42,
                    stratify=labels,
                )
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(x_train, y_train)
                score = accuracy_score(y_test, model.predict(x_test))
                classification = (
                    f"Random Forest по числовым признакам; целевой столбец "
                    f"«{target_column}»; accuracy={score:.3f} "
                    f"(тестовая выборка, случайное разбиение 75/25)."
                )
            except ValueError as error:
                classification = f"Классификация пропущена: {error}"
        else:
            classification = (
                "Для классификации нужны числовые признаки, минимум два класса "
                "и не менее 10 строк."
            )
    forecast = None
    if forecast_column and forecast_column in frame.select_dtypes(include="number").columns:
        values = frame[forecast_column].dropna().reset_index(drop=True)
        if len(values) >= 3:
            positions = values.index.to_numpy().reshape(-1, 1)
            model = LinearRegression().fit(positions, values.to_numpy())
            future_positions = range(len(values), len(values) + 5)
            predictions = model.predict(
                pd.DataFrame({"position": future_positions}).to_numpy()
            )
            forecast = pd.DataFrame(
                {
                    "Шаг (индекс строки)": list(future_positions),
                    "Линейный прогноз": predictions,
                }
            )
    return AnalyticsResult(summary, profile, clustered, chart, classification, forecast)
