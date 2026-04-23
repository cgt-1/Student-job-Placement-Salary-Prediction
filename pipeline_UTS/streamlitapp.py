import streamlit as st
import pickle
import pandas as pd


clf_model = pickle.load(open("models/classifier.pkl", "rb"))
reg_model = pickle.load(open("models/regressor.pkl", "rb"))


st.set_page_config(page_title="Student Placement Predictor", layout="wide")
st.title("Student job Placement & Salary Prediction")


st.sidebar.header("Input Student Data")

def user_input():

    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    branch = st.sidebar.selectbox("Branch", ["CSE", "ECE", "ME", "CE", "EE"])

    cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, 8.5)
    study_hours = st.sidebar.slider("Study Hours per Day", 0.0, 12.0, 6.0)
    attendance = st.sidebar.slider("Attendance %", 0.0, 100.0, 85.0)

    projects = st.sidebar.slider("Projects Completed", 0, 10, 5)
    internships = st.sidebar.slider("Internships Completed", 0, 5, 2)

    coding = st.sidebar.slider("Coding Skill (1-10)", 1, 10, 7)
    aptitude = st.sidebar.slider("Aptitude Skill (1-10)", 1, 10, 7)
    communication = st.sidebar.slider("Communication Skill (1-10)", 1, 10, 7)

    data = {
        'gender': gender,
        'branch': branch,
        'cgpa': cgpa,
        'study_hours_per_day': study_hours,
        'attendance_percentage': attendance,
        'projects_completed': projects,
        'internships_completed': internships,
        'coding_skill_rating': coding,
        'aptitude_skill_rating': aptitude,
        'communication_skill_rating': communication
    }

    return pd.DataFrame([data])


input_df = user_input()


st.subheader("Input Data")
st.write(input_df)


default_values = {
    'tenth_percentage': 70,
    'twelfth_percentage': 70,
    'backlogs': 0,
    'hackathons_participated': 2,
    'certifications_count': 3,
    'sleep_hours': 7,
    'stress_level': 4,
    'part_time_job': 'No',
    'family_income_level': 'Medium',
    'city_tier': 'Tier 1',
    'internet_access': 'Yes',
    'extracurricular_involvement': 'High'
}



for col, val in default_values.items():
    if col not in input_df.columns:
        input_df[col] = val



input_df['academic_score'] = (
    input_df['cgpa'] +
    input_df['tenth_percentage'] / 10 +
    input_df['twelfth_percentage'] / 10
) / 3

input_df['technical_score'] = (
    input_df['coding_skill_rating'] +
    input_df['aptitude_skill_rating'] +
    input_df['projects_completed'] +
    input_df['internships_completed']
) / 4




expected_cols = clf_model.feature_names_in_

for col in expected_cols:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[expected_cols]



st.subheader("Model Input")
st.write(input_df)



if st.button("Predict"):

    placement_pred = clf_model.predict(input_df)[0]
    salary_pred = reg_model.predict(input_df)[0]

    # probability (for debugging)
    proba = clf_model.predict_proba(input_df)[0]

    st.subheader("Prediction Result")

    st.write("Raw Prediction:", placement_pred)
    st.write("Probabilities [Not Placed, Placed]:", proba)


    if placement_pred == "Placed":
        st.success("Student is likely to be PLACED")
    else:
        st.error("Student is NOT likely to be placed")

    st.info(f"Estimated Salary: {salary_pred:.2f} LPA")