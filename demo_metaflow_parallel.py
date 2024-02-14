import os

from metaflow import FlowSpec, card, current, project, step
from metaflow.cards import Image

METAFLOW_IMAGE = (
    "us-east1-docker.pkg.dev/prolaio-data-testing/docker/metaflow-image-job"
)


def install_dependencies() -> None:
    os.system('pip install keyrings.google-artifactregistry-auth~="1.1.2"')
    os.system('pip install pandas~="2.0.1"')
    os.system('pip install neurokit2~="0.2.4"')
    os.system('pip install label-studio-sdk=="0.0.32"')
    os.system('pip install scikit-learn~="1.3.0"')
    os.system('pip install google-cloud~="0.34.0"')
    os.system('pip install google-cloud-storage~="2.14.0"')
    os.system('pip install google-cloud-secret-manager~="2.17.0"')
    os.system('pip install google-cloud-bigquery[all]~="3.14.1"')
    os.system('pip install db-dtypes~="1.2.0"')
    os.system('pip install fsspec~="2023.12.2"')
    os.system('pip install gcsfs~="2023.12.2.post1"')
    os.system('pip install google-api-python-client~="2.83.0"')
    os.system('pip install aws-secretsmanager-caching~="1.1.1.5"')
    os.system('pip install awswrangler~="3.3.0"')
    os.system('pip install JSON-log-formatter~="0.5.2"')
    os.system('pip install orjson~="3.8.10"')
    os.system('pip install pandera~="0.15.1"')
    os.system("pip install google-cloud-bigquery")
    os.system("pip install google-cloud-storage")
    os.system("pip install metaflow-card-html")
    os.system("pip install numpy")
    os.system("pip install reportlab")
    os.system("pip install matplotlib")
    os.system(
        "pip install -i https://northamerica-northeast1-python.pkg.dev/prolaio-data-testing/pypi/simple/ prolaiotoolkit"
    )


@project(name="demo")
class DemoMetaflowParallel(FlowSpec):

    @step
    def start(self):
        import pandas as pd
        from google.cloud import bigquery
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        # fetch from bigquery
        bq_client = bigquery.Client()
        query = "SELECT * FROM `prolaio-data-testing.vpasquierdemo.heart`"
        dataset = bq_client.query(query).to_dataframe()
        # Prepare data
        dataset_dummies = pd.get_dummies(
            dataset,
            columns=["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
        )
        dataset_dummies.head()
        standard_scaler = StandardScaler()
        columns_to_scale = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        dataset_dummies[columns_to_scale] = standard_scaler.fit_transform(
            dataset_dummies[columns_to_scale]
        )
        # Split dataset
        y = dataset_dummies["target"]
        X = dataset_dummies.drop(["target"], axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.next(self.random_forest_1)

    @card()
    @step
    def random_forest_1(self):
        self.nb_iter = 100
        self.score, self.model = self._train_model()
        self.next(self.end)

    @step
    def end(self):
        print("End of execution")

    def _train_model(self):
        # Training on Random Forest model
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import RandomizedSearchCV

        rf_param_grid = {
            "n_estimators": range(1, 100, 10),
        }
        rf = RandomForestClassifier()
        rf_random = RandomizedSearchCV(
            param_distributions=rf_param_grid,
            estimator=rf,
            scoring="accuracy",
            verbose=0,
            n_iter=self.nb_iter,
            cv=4,
        )
        rf_random.fit(self.X_train, self.y_train)
        best_params = rf_random.best_params_
        print(f"Best parameters: {best_params}")
        score = rf_random.score(self.X_test, self.y_test)
        print(f"Score: {score}")

        # Plot results
        def feature_imp(df, model):
            fi = pd.DataFrame(columns=["feature", "importance"])
            fi["feature"] = df.columns
            fi["importance"] = model.best_estimator_.feature_importances_
            return fi.sort_values(by="importance", ascending=False)

        features_importance = feature_imp(self.X_train, rf_random)
        current.card.append(
            Image.from_matplotlib(
                features_importance.plot(
                    "feature", "importance", "barh", figsize=(10, 7), legend=False
                )
            )
        )
        return score, rf_random


if __name__ == "__main__":
    DemoMetaflowParallel()
