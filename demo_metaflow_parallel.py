from metaflow import FlowSpec, card, current, kubernetes, project, step
from metaflow.cards import Image

METAFLOW_IMAGE = (
    "us-east1-docker.pkg.dev/prolaio-data-testing/docker/metaflow-image-job"
)


@project(name="demo")
class DemoMetaflowParallel(FlowSpec):

    # nb_iter = Parameter("nb_iter", help="Number of iterations", default="[100000, 2000000]", type=JSONType)

    @kubernetes(image=METAFLOW_IMAGE, cpu=4, memory=16384)
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

        self.next(self.random_forest_1, self.random_forest_2)

    @kubernetes(image=METAFLOW_IMAGE, cpu=4, memory=16384)
    @card()
    @step
    def random_forest_1(self):
        self.nb_iter = 100
        self.score = self._train_model()
        self.next(self.join)

    @kubernetes(image=METAFLOW_IMAGE, cpu=4, memory=16384)
    @card()
    @step
    def random_forest_2(self):
        self.nb_iter = 200
        self.score = self._train_model()
        self.next(self.join)

    @kubernetes(image=METAFLOW_IMAGE, cpu=4, memory=16384)
    @step
    def join(self, inputs):
        for input in inputs:
            print(f"Score: {input.score} for {input.nb_iter} iterations")
        self.next(self.end)

    @kubernetes(image=METAFLOW_IMAGE, cpu=4, memory=16384)
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
        return score


if __name__ == "__main__":
    DemoMetaflowParallel()
