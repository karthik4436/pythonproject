#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('writefile', 'app.py', 'import streamlit as st\nimport pandas as pd\nimport numpy as np\nimport joblib\n\nmodel = joblib.load("salary_predictor_model.pkl")\nmodel_columns = joblib.load("model_columns.pkl")\n\n\nst.set_page_config(page_title="Salary Predictor", layout="centered")\nst.title("💼 Employee Salary Predictor")\nst.markdown("Predict employee salary based on **education**, **experience**, **job role**, and **gender**.")\n\n\nst.sidebar.header("📝 Enter Details")\nexperience = st.sidebar.slider("Years of Experience", 0, 40)\neducation = st.sidebar.selectbox("Education Level", ["Bachelor\'s", "Master\'s", "PhD"])\ngender = st.sidebar.selectbox("Gender", ["Male", "Female"])\njob_title = st.sidebar.selectbox("Job Title", ["Software Engineer", "Data Analyst", "Senior Manager", "Sales Associate", "Director"])\n\n\ndef encode_input(exp, edu, gender, title):\n    data = {\n        \'Years of Experience\': exp,\n        \'Gender_Male\': 1 if gender == "Male" else 0,\n        \'Education Level_Master\\\'s\': 1 if edu == "Master\'s" else 0,\n        \'Education Level_PhD\': 1 if edu == "PhD" else 0,\n        \'Job Title_Data Analyst\': 1 if title == "Data Analyst" else 0,\n        \'Job Title_Director\': 1 if title == "Director" else 0,\n        \'Job Title_Sales Associate\': 1 if title == "Sales Associate" else 0,\n        \'Job Title_Senior Manager\': 1 if title == "Senior Manager" else 0,\n        \'Job Title_Software Engineer\': 1 if title == "Software Engineer" else 0,\n    }\n    return pd.DataFrame([data])\n\n\ninput_df = encode_input(experience, education, gender, job_title)\ninput_df = input_df.reindex(columns=model_columns, fill_value=0)\n\n\nif st.button("Predict Salary"):\n    salary = model.predict(input_df)[0]\n    st.success(f"💰 Predicted Salary: ₹{int(salary):,}")\n\n')


# In[ ]:


get_ipython().system('streamlit run app.py')


# In[ ]:




