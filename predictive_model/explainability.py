import shap
import pandas as pd
import numpy as np

def calculate_shap_values(model_artifact, X_train, X_test):
    """
    Calculates SHAP values for the test set, handling both Pipeline and GAM dictionary.
    
    Args:
        model_artifact: The trained pipeline or GAM dictionary artifact.
        X_train: Training data (needed for background distribution).
        X_test: Test data to explain.
        
    Returns:
        explainer: The SHAP explainer object.
        shap_values: The calculated SHAP values.
        X_test_transformed: Transformed test data.
        feature_names: Names of features after transformation.
    """
    
    # Check if it's our GAM dictionary artifact
    if isinstance(model_artifact, dict) and "model" in model_artifact and "preprocess" in model_artifact:
        model = model_artifact["model"]
        preprocessor = model_artifact["preprocess"]
        # label_encoder = model_artifact["label_encoder"] # Not needed for SHAP calculation directly
        
        # Transform data
        # Ensure we cast to float for pygam
        X_train_transformed = preprocessor.transform(X_train).astype(float)
        X_test_transformed = preprocessor.transform(X_test).astype(float)
        
        # Get feature names
        try:
            feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
             # Fallback if get_feature_names_out isn't available or fails
             feature_names = [f"feat_{i}" for i in range(X_train_transformed.shape[1])]

        # GAMs are additive, but pygam's implementation with sklearn API usually requires KernelExplainer
        # for general probability outputs.
        # Summarize background to speed up (GAMs can be slow with KernelExplainer)
        # Using 10-20 k-means centroids as background is standard for speed/approximation
        background = shap.kmeans(X_train_transformed, 10) 
        
        # We explain the probability of the positive class (usually index 1)
        # gam.predict_proba returns (N, 2) or (N,). We need a function that returns the prob.
        f = lambda x: model.predict_proba(x)
        
        explainer = shap.KernelExplainer(f, background)
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_test_transformed)
        
        return explainer, shap_values, X_test_transformed, feature_names

    # Legacy support for sklearn Pipeline (Baseline Model)
    elif hasattr(model_artifact, 'named_steps'):
        model = model_artifact
        classifier = model.named_steps['clf']
        preprocessor = model.named_steps['pre']
        
        # Transform data
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Get feature names
        try:
            feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
            feature_names = None

        if feature_names is None or len(feature_names) != X_train_transformed.shape[1]:
             feature_names = [f"feature_{i}" for i in range(X_train_transformed.shape[1])]

        if "Forest" in str(type(classifier)) or "Tree" in str(type(classifier)):
             explainer = shap.TreeExplainer(classifier)
             shap_values = explainer.shap_values(X_test_transformed)
        else:
            background = shap.kmeans(X_train_transformed, 10) 
            explainer = shap.KernelExplainer(classifier.predict_proba, background)
            shap_values = explainer.shap_values(X_test_transformed)
            
        return explainer, shap_values, X_test_transformed, feature_names

    else:
        raise ValueError("Model format not recognized. Expected GAM dictionary or sklearn Pipeline.")

def generate_genai_explanation(student_id, risk_prob, top_features):
    """
    Generates a natural language explanation for a student's risk.
    
    Args:
        student_id: ID of the student.
        risk_prob: Probability of dropout (0.0 to 1.0).
        top_features: List of tuples (feature_name, shap_value) representing key drivers.
        
    Returns:
        str: A natural language summary.
    """
    # MOCK GENAI RESPONSE
    # In a real scenario, this would call OpenAI/Gemini API
    
    risk_level = "High" if risk_prob > 0.6 else "Medium" if risk_prob > 0.3 else "Low"
    
    explanation = f"""
    **Student Analysis (ID: {student_id})**
    
    **Risk Level:** {risk_level} ({risk_prob:.1%})
    
    **Key Drivers:**
    """
    
    for feature, impact in top_features[:3]:
        direction = "increases risk" if impact > 0 else "decreases risk"
        explanation += f"- **{feature}**: This factor {direction} (Impact: {impact:.2f}).\n"
        
    explanation += "\n**GenAI Summary:**\n"
    explanation += f"Based on the predictive model, this student shows a {risk_level.lower()} likelihood of dropping out. "
    
    if risk_level == "High":
        explanation += "The primary concerns are related to their academic performance and financial factors. " \
                       "Immediate intervention is recommended to discuss support options."
    elif risk_level == "Medium":
        explanation += "While not critical, there are some warning signs. A check-in email would be beneficial " \
                       "to ensure they are on the right track."
    else:
        explanation += "The student appears to be progressing well. No immediate action is required, " \
                       "but positive reinforcement is always helpful."
                       
    return explanation
