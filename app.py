import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle

# disable warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# load trained model
lgbm_base = pickle.load(open('lgbm_base.pkl', 'rb'))
lgbm_opt = pickle.load(open('lgbm_opt.pkl', 'rb'))

# Explicitly set the num_classes attribute for the models
lgbm_base._n_classes = 1  # Assuming it's a regression model (single output)
lgbm_opt._n_classes = 1   # Assuming it's a regression model (single output)



np.random.seed(123)

# Set font color and background color using HTML tags
st.markdown("<body style='color: #00D9DC; background-color: #FFFFFF;'> </body>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: #000000;'>Welcome to My HuggingFace Streamlit App!</h2>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; color: #800080;'>This app generates house sale price predictions . </h3>", unsafe_allow_html=True)

# Add more HTML styling to the instructions
st.markdown("<h5 style='text-align: center; color: #00008B; background-color: #F0F0F0;'>Instructions:</h5>", unsafe_allow_html=True)

st.markdown("<h5 style='text-align: center; color: #00008B;'>On the sidebar to the left, you will find a series of questions about the house specifics that require your input. Once you have finished answering the questions, click the button below to generate the predicted house sale price, as well as the respective SHAP summary plot and SHAP interaction plot.</h5>", unsafe_allow_html=True)

st.markdown("<h5 style='text-align: center; color: #00008B;'>Note: Please allow the app some time (a few seconds) to generate predictions and plots.</h5>", unsafe_allow_html=True)

name_list = ['OverallQual', 'OverallCond', 'YrSold', 'YearBuilt', 'YearRemodAdd', 
                   'LotArea', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GrLivArea', 
                   'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 
                   'GarageYrBlt', 'GarageArea', 'PoolArea'
]

name_list_train = ['OverallQual', 'OverallCond', 'YrSold', 'YearBuilt', 'YearRemodAdd', 
                   'LotArea', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GrLivArea', 
                   'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 
                   'GarageYrBlt', 'GarageArea', 'PoolArea'
]

data = pd.read_csv('train.csv')

data = data[name_list_train].values

description_list = [
    'What is the overall material and quality finish rating?',
    'What is the overall condition rating?',
    'In which year was the house sold?',
    'In which year was the house built?',
    'In which year was the house remodeled?',
    'What is the lot size (in square feet)?',
    'What is the total basement area (in square feet)?',
    'How many full bathrooms in the basement area?',
    'How many half bathrooms in the basement area?',
    'What is the total ground area (in square feet)?',
    'How many full bathrooms in the ground area?',
    'How many half bathrooms in the ground area?',
    'How many bedrooms in the ground area?',
    'How many kitchens in the ground area?',
    'How many fireplaces in the ground area?',
    'In which year was the garage built?',
    'What is the garage size (in square feet)?',
    'What is the pool area (in square feet)?'
 ]

min_list = [1, 1, 2006, 1872, 1950, 1300, 0, 0, 
            0, 334, 0, 0, 0, 0, 0, 1900, 0, 0
]

max_list = [10, 9, 2010, 2010, 2010, 215245, 6110, 3, 
            2, 5642, 3, 2, 8, 3, 3, 2010, 1418, 738
]

count = 0

with st.sidebar:

    for i in range(len(name_list)):

        variable_name = name_list[i]
        globals()[variable_name] = st.slider(description_list[i] ,min_value=int(min_list[i]), max_value =int(max_list[i]),step=1)
        
data_df = {'OverallQual': [OverallQual], 'OverallCond': [OverallCond], 'YrSold': [YrSold],'YearBuilt': [YearBuilt], 
           'YearRemodAdd': [YearRemodAdd], 'LotArea': [LotArea], 'TotalBsmtSF':[TotalBsmtSF], 'BsmtFullBath': [BsmtFullBath], 
           'BsmtHalfBath': [BsmtHalfBath], 'GrLivArea':[GrLivArea], 'FullBath': [FullBath], 'HalfBath': [HalfBath], 
           'BedroomAbvGr':[BedroomAbvGr], 'KitchenAbvGr': [KitchenAbvGr], 'Fireplaces' : [Fireplaces], 
           'GarageYrBlt' : [GarageYrBlt], 'GarageArea' : [GarageArea], 'PoolArea' : [PoolArea]
}

data_df = pd.DataFrame.from_dict(data_df)

y_pred_base = lgbm_base.predict(data_df)
y_pred_opt = lgbm_opt.predict(data_df)

col1, col2, col3 , col4, col5 = st.columns(5)

with col1:
    pass
with col2:
    pass
with col4:
    pass
with col5:
    pass
with col3 :
    center_button = st.button('Generate House Price')

if center_button:
    import time

    with st.spinner('Calculating....'):
        time.sleep(2)
        st.markdown("<h5 style='text-align: center; color: #1B9E91;'>The price range of your house is between:</h5>", unsafe_allow_html=True)


    col1, col2, col3 = st.columns([3, 3, 3])

    estimated_price_base = "{:,.0f}".format(float(y_pred_base.mean()))
    estimated_price_opt = "{:,.0f}".format(float(y_pred_opt.mean()))


    with col1:
        st.write("")


        st.markdown("<h3 style='text-align: center;'>LightGBM Baseline Model Prediction:</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>USD {estimated_price_base}</h3>", unsafe_allow_html=True)
        st.write("")

        st.markdown("<h5 style='text-align: center; color: #00D9DC;'>The SHAP summary and interaction plots are:</h5>", unsafe_allow_html=True)
        explainer_base = shap.TreeExplainer(lgbm_base)
        shap_explainer_base = explainer_base(data_df)
        shap_values_base = shap.TreeExplainer(lgbm_base).shap_values(data_df)
        shap_interaction_values_base = shap.TreeExplainer(lgbm_base).shap_interaction_values(data_df)

        st.subheader("SHAP Summary Plot")
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        shap.plots.beeswarm(shap_explainer_base, max_display=10)
        st.markdown("</div>", unsafe_allow_html=True)
        st.pyplot()

        st.subheader("SHAP Interaction Plot")
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        shap.summary_plot(shap_interaction_values_base, data_df)
        st.markdown("</div>", unsafe_allow_html=True)
        st.pyplot()
        
        st.markdown("<h3 style='text-align: center;'>LightGBM Model Optimized (via Optuna) Prediction:</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>USD {estimated_price_opt}</h3>", unsafe_allow_html=True)
        st.write("")

        st.markdown("<h5 style='text-align: center; color: #00D9DC;'>The SHAP summary and interaction plots are:</h5>", unsafe_allow_html=True)
        explainer_opt = shap.TreeExplainer(lgbm_opt)
        shap_explainer_opt = explainer_opt(data_df)
        shap_values_opt = shap.TreeExplainer(lgbm_opt).shap_values(data_df)
        shap_interaction_values_opt = shap.TreeExplainer(lgbm_opt).shap_interaction_values(data_df)

        st.subheader("SHAP Summary Plot")
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        shap.plots.beeswarm(shap_explainer_opt, max_display=10)
        st.markdown("</div>", unsafe_allow_html=True)
        st.pyplot()

        st.subheader("SHAP Interaction Plot")
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        shap.summary_plot(shap_interaction_values_opt, data_df)
        st.markdown("</div>", unsafe_allow_html=True)
        st.pyplot()
