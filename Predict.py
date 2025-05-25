import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Define encoding maps
eth_map = {"White": 0, "Black": 1, "Asian": 2, "Others": 3}
cont_map = {"USA": 0, "UK": 1, "India": 2, "Others": 3}
rel_map = {"Self": 0, "Parent": 1, "Relative": 2, "Others": 3}

# Load the trained models
models = {}
models['KNearestNeighbours'] = pickle.load(open("knn_model.pkl", "rb"))
models['DecisionTree'] = pickle.load(open("dt_model.pkl", "rb"))
models['LightGBM'] = pickle.load(open("lgb_model.pkl", "rb"))
models['XGBoostRF'] = pickle.load(open("xgb_model.pkl", "rb"))
models['CatBoost'] = pickle.load(open("cat_model.pkl", "rb"))
models['RandomForest'] = pickle.load(open("rf_model.pkl", "rb"))
models['LogisticRegression'] = pickle.load(open("logistic_model.pkl", "rb"))
models['SVC'] = pickle.load(open("svc_model.pkl", "rb"))

# Load the StandardScaler used for training
scaler = pickle.load(open("scaler.pkl", "rb"))

# Define functions for data preprocessing
def value_count(val):
    return 1 if val == "yes" else 0

def preprocess_data(input_data):
    input_data_dict = {
        'age': input_data[0],
        'ethnicity': input_data[1],
        'jaundice': input_data[2],
        'austim': input_data[3],
        'contry_of_res': input_data[4],
        'result': input_data[5],
        'A1_Score': input_data[6],
        'A2_Score': input_data[7],
        'A3_Score': input_data[8],
        'A4_Score': input_data[9],
        'A5_Score': input_data[10],
        'A6_Score': input_data[11],
        'A7_Score': input_data[12],
        'A8_Score': input_data[13],
        'A9_Score': input_data[14],
        'A10_Score': input_data[15],
        'relation': input_data[16]
    }

    # Group age
    age = input_data_dict['age']
    if age <= 14:
        input_data_dict['age'] = 1
    elif 14 < age <= 24:
        input_data_dict['age'] = 2
    elif 24 < age <= 64:
        input_data_dict['age'] = 3
    else:
        input_data_dict['age'] = 4

    # Encode categorical variables directly
    input_data_dict['ethnicity'] = eth_map.get(input_data_dict['ethnicity'], len(eth_map))
    input_data_dict['jaundice'] = value_count(input_data_dict['jaundice'])
    input_data_dict['austim'] = value_count(input_data_dict['austim'])
    input_data_dict['contry_of_res'] = cont_map.get(input_data_dict['contry_of_res'], len(cont_map))
    input_data_dict['relation'] = rel_map.get(input_data_dict.get('relation', 'Self'), len(rel_map))

    input_data_array = [input_data_dict[col] for col in [
        'age', 'ethnicity', 'jaundice', 'austim', 'contry_of_res', 'result',
        'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
        'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'relation']]

    scaled_data = scaler.transform([input_data_array])
    return scaled_data

# Prediction
def predict(model, data):
    prediction = model.predict(data)
    return prediction

# Form layout
st.title(":bookmark_tabs: :blue[Autism Data Assessment]")
st.write("---")
st.write("Fill the form below to check if you or your child is suffering from ASD")

# Form fields
form_values = {}
form_values['age'] = st.number_input("Age", min_value=1, max_value=120, step=1)
form_values['ethnicity'] = st.selectbox("Ethnicity", options=["White", "Black", "Asian", "Others"])
form_values['jaundice'] = st.selectbox("Jaundice", options=["no", "yes"])
form_values['austim'] = st.selectbox("Family member with ASD", options=["no", "yes"])
form_values['contry_of_res'] = st.selectbox("Country of Residence", options=["USA", "UK", "India", "Others"])
form_values['result'] = st.number_input("Screening Test Result", min_value=0, max_value=10, step=1)

# Adding A1 to A10 questions
questions = {
    'A1_Score': "Does your child look at you when you call his/her name?",
    'A2_Score': "How easy is it for you to get eye contact with your child?",
    'A3_Score': "Does your child point to indicate that s/he wants something?",
    'A4_Score': "Does your child point to share interest with you?",
    'A5_Score': "Does your child pretend?",
    'A6_Score': "Does your child follow where you're looking?",
    'A7_Score': "If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them?",
    'A8_Score': "Would you describe your child’s first words as:",
    'A9_Score': "Does your child use simple gestures?",
    'A10_Score': "Does your child stare at nothing with no apparent purpose"
}

# Note for answering questions A1 to A10
st.info("""
For the following questions, respond with 1 or 0:
- For questions A1-A9, respond with 1 for 'Sometimes', 'Rarely', or 'Never', otherwise respond with 0.
- For question A10, respond with 1 for 'Always', 'Usually', or 'Sometimes', otherwise respond with 0.
""")

for key, question in questions.items():
    form_values[key] = st.selectbox(question, options=[0, 1])
form_values['relation'] = st.selectbox("Relation", options=["Self", "Parent", "Relative", "Others"])

input_data = [form_values[col] for col in [
    'age', 'ethnicity', 'jaundice', 'austim', 'contry_of_res', 'result'] + list(questions.keys()) + ['relation']]

# Preprocess and predict
std_data = preprocess_data(input_data)

results = {}
for model_name, model in models.items():
    prediction = predict(model, std_data)
    results[model_name] = prediction

# Display results
st.subheader("Results:")
for model_name, prediction in results.items():
    st.write(f"Model: {model_name}, Prediction: {'Autism' if prediction[0] == 1 else 'No Autism'}")

# Determine and display the best model
accuracies = {
    'KNearestNeighbours': 0.88,  
    'DecisionTree': 0.79,
    'LightGBM': 0.92,
    'XGBoostRF': 0.92,
    'CatBoost': 0.93,
    'RandomForest': 0.92,
    'LogisticRegression': 0.93,
    'SVC': 0.51
}

# Best model approach
best_model_name = max(accuracies, key=accuracies.get)
st.subheader(f"Best Model: {best_model_name}")
best_prediction = results[best_model_name]
st.write(f"Overall Prediction (Best Model): {'Autism' if best_prediction[0] == 1 else 'No Autism'}")

# Ensemble approach
ensemble_prediction = 1 if sum(pred[0] for pred in results.values()) > len(results) / 2 else 0
st.write(f"Overall Prediction (Ensemble): {'Autism' if ensemble_prediction == 1 else 'No Autism'}")

# Plot model accuracies
st.subheader("Model Accuracies:")
model_list = list(accuracies.keys())
score = list(accuracies.values())

plt.rcParams['figure.figsize'] = 20, 8
sns.set_style('darkgrid')
ax = sns.barplot(x=model_list, y=score, palette="husl", saturation=2.0)
plt.xlabel('Classifier Models', fontsize=20)
plt.ylabel('% of Accuracy', fontsize=20)
plt.title('Accuracy of different Classifier Models', fontsize=20)
plt.xticks(fontsize=12, horizontalalignment='center', rotation=8)
plt.yticks(fontsize=12)
for i in ax.patches:
    width, height = i.get_width(), i.get_height()
    x, y = i.get_xy()
    ax.annotate(f'{round(height * 100, 2)}%', (x + width / 2, y + height * 1.02), ha='center', fontsize='x-large')
st.pyplot(plt)
