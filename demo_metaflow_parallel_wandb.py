import os

from metaflow import FlowSpec, project, step

import wandb
from wandb.integration.metaflow import wandb_log


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
    os.system("pip install fastcore wandb pytorch")
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

    @wandb_log(
        datasets=True,
        models=True,
        others=True,
        settings=wandb.Settings(project="demo_random_forest"),
    )
    @step
    def random_forest_1(self):
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
        )
        import wandb

        rf_param_grid = {
            "n_estimators": range(1, 100, 10),
        }
        rf = RandomForestClassifier()
        rf_random = RandomizedSearchCV(
            param_distributions=rf_param_grid,
            estimator=rf,
            scoring="accuracy",
            verbose=0,
            n_iter=100,
            cv=4,
        )
        rf_random.fit(self.X_train, self.y_train)
        rf_random.score(self.X_test, self.y_test)

        def feature_imp(df, model):
            fi = pd.DataFrame(columns=["feature", "importance"])
            fi["feature"] = df.columns
            fi["importance"] = model.best_estimator_.feature_importances_
            return fi.sort_values(by="importance", ascending=False)

        self.features_importance = feature_imp(self.X_train, rf_random)
        wandb.sklearn.plot_learning_curve(rf_random, self.X_train, self.y_train)
        wandb.sklearn.plot_feature_importances(rf_random)

        # Get best parameters

        # Evaluate model on test data
        y_pred = rf_random.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)

        # Log confusion matrix
        confusion_matrix(self.y_test, y_pred)
        # wandb.log(
        #     {
        #         "confusion_matrix": wandb.plot.confusion_matrix(
        #             probs=None, y_true=self.y_test, preds=y_pred
        #         )
        #     }
        # )

        # Log accuracy
        wandb.log({"accuracy": accuracy})

        # Log loss
        loss = 1 - accuracy
        wandb.log({"loss": loss})

        # Log feature importance
        features_importance = feature_imp(self.X_train, rf_random)
        wandb.log({"feature_importance": wandb.Table(data=features_importance)})

        # Plot learning curve
        wandb.sklearn.plot_learning_curve(rf_random, self.X_train, self.y_train)

        # Log training loss
        train_loss = 1 - rf_random.best_score_
        wandb.log({"train_loss": train_loss})

        # Plot feature importances
        wandb.sklearn.plot_feature_importances(rf_random)

        self.next(self.end)

    @step
    def end(self):
        print("End of execution")


if __name__ == "__main__":
    DemoMetaflowParallel()
