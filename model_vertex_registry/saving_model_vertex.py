# Saving the model
import joblib
from google.cloud import aiplatform, storage

# Save via joblib
joblib.dump(rf_random, "saved_model.joblib")

# Set up GCP credentials
storage_client = storage.Client()

# Specify your GCP bucket and object paths
bucket_name = "configuration-pilot"
object_name = "saved_model.joblib"

# Upload the model to GCP Storage
bucket = storage_client.get_bucket(bucket_name)
blob = bucket.blob(f"models/{object_name}")
blob.upload_from_filename(object_name)

# Set up GCP credentials
aiplatform.init(project="prolaio-data-testing")

# Specify the model display name and description
model_display_name = "random-forest-v1"
model_description = "The best one"

# Specify the model URI in GCP Storage
model_uri = f"gs://{bucket_name}/models"

# Register the model in the Model Registry
model = aiplatform.Model(model_name=model_display_name)
model.update()
