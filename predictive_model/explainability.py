import shap
import pandas as pd
import numpy as np

def calculate_shap_values(model, X_train, X_test):
    """
    Calculates SHAP values for the test set.
    
    Args:
        model: The trained pipeline or model.
        X_train: Training data (needed for some SHAP explainers).
        X_test: Test data to explain.
        
    Returns:
        explainer: The SHAP explainer object.
        shap_values: The calculated SHAP values.
    """
    # Extract the classifier step from the pipeline
    # Assuming the model is a Pipeline with 'pre' (preprocessor) and 'clf' (classifier)
    if hasattr(model, 'named_steps'):
        classifier = model.named_steps['clf']
        preprocessor = model.named_steps['pre']
        
        # We need to transform the data first because SHAP works on the features *seen* by the model
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Get feature names from the preprocessor
        try:
            feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
            feature_names = None

        # Ensure feature names match the transformed data shape
        if feature_names is None or len(feature_names) != X_train_transformed.shape[1]:
             feature_names = [f"feature_{i}" for i in range(X_train_transformed.shape[1])]

        # For Tree-based models (RandomForest, XGBoost, etc.)
        # If it's LogisticRegression, we might need a different explainer (LinearExplainer)
        if "Forest" in str(type(classifier)) or "Tree" in str(type(classifier)):
             explainer = shap.TreeExplainer(classifier)
             shap_values = explainer.shap_values(X_test_transformed)
        else:
            # Fallback for generic models (like Logistic Regression) using KernelExplainer or LinearExplainer
            # Using LinearExplainer for speed if it's linear, otherwise Kernel (slow)
            # For safety in this demo, let's use a generic approach or Linear if applicable
            # But since we saw LogisticRegression in the baseline, let's try LinearExplainer with a summary of train data
            # Note: LinearExplainer expects independent features, might be an approximation
            
            # Summarizing background data to speed up SHAP
            background = shap.kmeans(X_train_transformed, 10) 
            explainer = shap.KernelExplainer(classifier.predict_proba, background)
            shap_values = explainer.shap_values(X_test_transformed)
            
        return explainer, shap_values, X_test_transformed, feature_names

    else:
        raise ValueError("Model format not recognized. Expected a sklearn Pipeline.")

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
