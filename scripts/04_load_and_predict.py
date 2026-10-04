import mlflow
from sklearn.datasets import load_breast_cancer

 
def load_and_predict():
    """
    Simulates a production scenario by loading a model using an alias
    from the MLflow Model Registry and using it for prediction.
    """
    MODEL_NAME = "cancer-classifier-prod"
    MODEL_ALIAS = "staging"  # MLflow 3 ใช้ Alias แทน Stage เดิม (เช่น staging, champion)
 
    print(f"Loading model '{MODEL_NAME}' with alias '@{MODEL_ALIAS}'...")
 
    # Load the model from the Model Registry ด้วย Alias URI
    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    except mlflow.exceptions.MlflowException as e:
        print(f"\nError loading model: {e}")
        print(f"Please make sure a model version has the alias '@{MODEL_ALIAS}' in the MLflow UI.")
        return
 
    # Prepare new sample data (as_frame=True เพื่อให้ชื่อคอลัมน์ตรงกับ signature ของโมเดล)
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    sample_data1 = X.iloc[0:1]  # Using the first row as a sample
    sample_data2 = X.iloc[19:20]  
    actual_label1 = y.iloc[0]
    actual_label2 = y.iloc[19]

    # กำหนด Mapping คำอธิบายคลาส
    label_map = {
        0: "Malignant",
        1: "Benign"
    }
 
    # Use the loaded model to make a prediction
    # No manual preprocessing is needed because we logged the entire pipeline
    prediction1 = model.predict(sample_data1)
    prediction2 = model.predict(sample_data2)
    pred_label1 = int(prediction1[0])
    pred_label2 = int(prediction2[0])
 
    print("-" * 30)
    # print(f"Sample Data Features:\n{sample_data.iloc[0]}")
    print(f"Actual Label1: {label_map[actual_label1]}")
    print(f"Predicted Label1: {label_map[pred_label1]}")
    print("-" * 30)
    print(f"Actual Label2: {label_map[actual_label2]}")
    print(f"Predicted Label2: {label_map[pred_label2]}")
    print("-" * 30)
 
if __name__ == "__main__":
    load_and_predict()
