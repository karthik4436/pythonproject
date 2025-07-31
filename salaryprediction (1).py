#!/usr/bin/env python
# coding: utf-8

# In[1]:

import streamlit as st
import pandas as pd
data=pd.read_csv("C:/Users/karth/Downloads/Salary Data.csv")
data



# In[2]:


data.head(15)


# In[5]:


data.shape


# In[6]:


data.isna().sum()



# In[7]:


print(data['Education Level'].value_counts())


# In[8]:


print(data['Job Title'].value_counts())


# In[2]:


print(data['Gender'].value_counts())


# In[10]:


data['Education Level'].replace({'?':'Others'},inplace=True)
print('Education Level')
print(data['Education Level'].value_counts())


# In[11]:


print(data['Education Level'].value_counts())


# In[14]:


data['Education Level'].replace({'?':'Others'},inplace=True)


# In[40]:


data.dropna(subset=['Age'], inplace=True)


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

Q1 = data['Salary'].quantile(0.25)
Q3 = data['Salary'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[data['Salary'].between(lower_bound, upper_bound)]
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(x=data['Salary'])
plt.title("Before Outlier Removal")

plt.subplot(1, 2, 2)
sns.boxplot(x=data['Salary'])
plt.title("After Outlier Removal")

plt.tight_layout()
plt.show()


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.boxplot(data['Age'])
data=data[(data['Age']<=55) & (data['Age']>=27)]
plt.title('Boxplot of age')
plt.show()


# In[7]:


data_encoded = pd.get_dummies(data,columns=['Gender', 'Education Level', 'Job Title'], drop_first=True)
X = data_encoded[[
    'Years of Experience',
    'Gender_Male',
    'Education Level_Master\'s',
    'Education Level_PhD',
    'Job Title_Data Analyst',
    'Job Title_Director',
    'Job Title_Sales Associate',
    'Job Title_Senior Manager',
    'Job Title_Software Engineer'
]]
y = data_encoded['Salary']


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
data=pd.read_csv("C:/Users/karth/Downloads/Salary Data.csv")

# Remove outliers (example using IQR method)
Q1 = data['Salary'].quantile(0.25)
Q3 = data['Salary'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

data = data[data['Salary'].between(lower_bound, upper_bound)]
import matplotlib.pyplot as plt
plt.boxplot(data['Salary'])
plt.show()


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
data=pd.read_csv("C:/Users/karth/Downloads/Salary Data.csv")

# Remove outliers (example using IQR method)
Q1 = data['Years of Experience'].quantile(0.25)
Q3 = data['Years of Experience'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

data = data[data['Years of Experience'].between(lower_bound, upper_bound)]
import matplotlib.pyplot as plt
plt.boxplot(data['Years of Experience'])
plt.show()


# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = data_encoded[[
    'Years of Experience',
    'Gender_Male',
    'Education Level_Master\'s',
    'Education Level_PhD',
    'Job Title_Data Analyst',
    'Job Title_Director',
    'Job Title_Sales Associate',
    'Job Title_Senior Manager',
    'Job Title_Software Engineer'
]]
y = data_encoded['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared Score:", r2)

get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(8, 5))
sns.regplot(x=data['Years of Experience'], y=data['Salary'])
plt.title("Linear Regression: Years of Experience vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.grid(True)
plt.tight_layout()
plt.savefig("linear_regression_plot.png")
plt.show()
joblib.dump(model, "salary_predictor_model.pkl")
joblib.dump(X_train.columns.tolist(), "model_columns.pkl")



# In[9]:


get_ipython().run_cell_magic('writefile', 'app.py', 'import streamlit as st\nimport pandas as pd\nimport numpy as np\nimport joblib\n\nmodel = joblib.load("salary_predictor_model.pkl")\nmodel_columns = joblib.load("model_columns.pkl")\n\n\nst.set_page_config(page_title="Salary Predictor", layout="centered")\nst.title("💼 Employee Salary Predictor")\nst.markdown("Predict employee salary based on **education**, **experience**, **job role**, and **gender**.")\n\n\nst.sidebar.header("📝 Enter Details")\nexperience = st.sidebar.slider("Years of Experience", 0, 40)\neducation = st.sidebar.selectbox("Education Level", ["Bachelor\'s", "Master\'s", "PhD"])\ngender = st.sidebar.selectbox("Gender", ["Male", "Female"])\njob_title = st.sidebar.selectbox("Job Title", ["Software Engineer", "Data Analyst", "Senior Manager", "Sales Associate", "Director"])\n\n\ndef encode_input(exp, edu, gender, title):\n    data = {\n        \'Years of Experience\': exp,\n        \'Gender_Male\': 1 if gender == "Male" else 0,\n        \'Education Level_Master\\\'s\': 1 if edu == "Master\'s" else 0,\n        \'Education Level_PhD\': 1 if edu == "PhD" else 0,\n        \'Job Title_Data Analyst\': 1 if title == "Data Analyst" else 0,\n        \'Job Title_Director\': 1 if title == "Director" else 0,\n        \'Job Title_Sales Associate\': 1 if title == "Sales Associate" else 0,\n        \'Job Title_Senior Manager\': 1 if title == "Senior Manager" else 0,\n        \'Job Title_Software Engineer\': 1 if title == "Software Engineer" else 0,\n    }\n    return pd.DataFrame([data])\n\n\ninput_df = encode_input(experience, education, gender, job_title)\ninput_df = input_df.reindex(columns=model_columns, fill_value=0)\n\n\nif st.button("Predict Salary"):\n    salary = model.predict(input_df)[0]\n    st.success(f"💰 Predicted Salary: ₹{int(salary):,}")\n\n')


# In[ ]:


get_ipython().system('streamlit run app.py')


# In[ ]:


get_ipython().system('streamlit run app.py')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'app.py', 'import streamlit as st\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport joblib\n\n# Load the trained model and the column order used during training\nmodel = joblib.load("salary_predictor_model.pkl")\nmodel_columns = joblib.load("model_columns.pkl")\n\n# App title\nst.title("💼 Salary Prediction Comparison: Male vs Female")\n\n# Sidebar input section\nst.sidebar.header("Input for Comparison")\neducation = st.sidebar.selectbox("Education Level", ["Bachelor\'s", "Master\'s", "PhD"])\njob_title = st.sidebar.selectbox("Job Title", [\n    "Software Engineer", "Data Analyst", "Senior Manager", "Sales Associate", "Director"\n])\n\n# Define the input encoding function\ndef encode_input(exp, edu, gender, title):\n    return pd.DataFrame([{\n        \'Years of Experience\': exp,\n        \'Gender_Male\': 1 if gender == "Male" else 0,\n        \'Education Level_Master\\\'s\': 1 if edu == "Master\'s" else 0,\n        \'Education Level_PhD\': 1 if edu == "PhD" else 0,\n        \'Job Title_Data Analyst\': 1 if title == "Data Analyst" else 0,\n        \'Job Title_Director\': 1 if title == "Director" else 0,\n        \'Job Title_Sales Associate\': 1 if title == "Sales Associate" else 0,\n        \'Job Title_Senior Manager\': 1 if title == "Senior Manager" else 0,\n        \'Job Title_Software Engineer\': 1 if title == "Software Engineer" else 0,\n    }])\n\n# Range of experience values\nexp_range = list(range(0, 41))\nmale_salaries = []\nfemale_salaries = []\n\n# Predict salary for each experience value for both genders\nfor exp in exp_range:\n    male_input = encode_input(exp, education, "Male", job_title).reindex(columns=model_columns, fill_value=0)\n    female_input = encode_input(exp, education, "Female", job_title).reindex(columns=model_columns, fill_value=0)\n\n    male_salary = model.predict(male_input)[0]\n    female_salary = model.predict(female_input)[0]\n\n    male_salaries.append(male_salary)\n    female_salaries.append(female_salary)\n\n# Plot the comparison chart\nfig, ax = plt.subplots()\nax.plot(exp_range, male_salaries, label="Male", color="blue", linewidth=2)\nax.plot(exp_range, female_salaries, label="Female", color="pink", linewidth=2)\nax.set_xlabel("Years of Experience")\nax.set_ylabel("Predicted Salary (₹)")\nax.set_title(f"Predicted Salary vs Experience\\\\n({job_title}, {education})")\nax.legend()\nax.grid(True)\n\n# Show plot in Streamlit\nst.pyplot(fig)\n')


# In[ ]:


get_ipython().system('streamlit run app.py')


# In[ ]:





# In[ ]:




