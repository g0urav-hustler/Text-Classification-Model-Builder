import shutil
import streamlit as st
import os
import time
import yaml
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_processing import DataProcessing
from src.components.train_model import TrainModel
from src.components.evaluate_model import EvaluateModel
from src.utils.common import read_yaml
from pathlib import Path


@st.cache_resource(show_spinner="Loading model tokenizer...")
def load_tokenizer(tokenizer_path):
    print("Tokenizer path == ", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer


@st.cache_resource(show_spinner="Loading model for testing..")
def load_model( model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_name= model_path,)
    return model


def compressing_model(model_path, zip_file_name):
    shutil.make_archive(zip_file_name, "zip", model_path)

model_name = None
model_result = None

# page config
st.set_page_config(page_title="Text Classification Model Builder", layout="wide",)

# title
st.title("Text Classification Model Application")

#subtitle
st.markdown("This application helps you to fine hugging face model on your custom dataset. ")

uploaded_file = st.file_uploader("Upload your data set file in csv format")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    df_columns = list(df.columns)
    
    df_columns.insert(0, "None")

    with st.form("my_form"):

        #subtitle
        st.markdown("Enter the data from dataset")
        col1, col2 = st.columns(2)
        
        # context
        with col1:
            text_column = st.selectbox(label = "Select the text column", options= df_columns)

        # question column input
        with col2:
            label_column = st.selectbox(label = "Select the labels column", options= df_columns)

        train_data_range = list(range(10,100,10))
        data_size_col1, data_size_col2, data_size_col3 = st.columns(3)

        # train_data_size input
        # train_data_range = 80 
        with data_size_col1:
            train_data_size = st.selectbox("Enter training data size ", options=train_data_range)

        # val_data_size input
        validation_data_range = list(range(10,100,10))
        with data_size_col2:
            val_data_size = st.selectbox("Enter validation data size ", options= validation_data_range)

        test_data_range = list(range(0,100,10))
        with data_size_col3:
            test_data_size = st.selectbox("Enter test data size ", options= test_data_range)


        st.markdown("Select the model parameter")
        
        par_col1, par_col2, par_col3, par_col4 = st.columns(4)

        # model type input
        with par_col1:
            model_name = st.text_input(label="Enter model name ", value = None)

        # epochs input
        with par_col2:
            no_of_epochs = st.number_input("Enter the no of epochs", value=0)

        # train batch size
        with par_col3:
            train_batch_size = st.number_input("Enter train batch size ", value = 4)

        # val batch size input
        with par_col4:
            val_batch_size = st.number_input("Enter val batch size", value = 4)

        sb_col1, sb_col2, sb_col3, sb_col4, sb_col5 = st.columns(5)
        with sb_col3:
            submitted = st.form_submit_button(label = "Submit Parameters")


    if submitted:

        labels = list(df[label_column].unique())
        labels.sort()
        num_labels = df[label_column].nunique()
         
        params_data = {
        "data_processing":
            {
                "text_col": text_column,
                "label_col": label_column,
                
                "train_data_size": train_data_size / 100,
                "val_data_size": val_data_size /100 ,
                "test_data_size": test_data_size /100

            },

            "num_labels": num_labels,
            "labels": labels,

           "model_params":
            {
                "model_name": model_name,
                "epochs": no_of_epochs,
                "train_batch_size": train_batch_size,
                "val_batch_size": val_batch_size
            }

        }

        print("After puting ", params_data)

        folder = 'web_files'
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, "data_file.csv")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with open('params.yaml', 'w') as outfile:
            yaml.dump(params_data, outfile, default_flow_style=False)

    button_col1, button_col2, button_col3, button_col4, button_col5 = st.columns(5)
    if os.path.isfile("params.yaml"):
        with button_col3:
            train_button = st.button("Train The Model")
        if train_button and not submitted:
            with st.status("Training Status", expanded=True) as status:
                config = ConfigurationManager()
                st.write("Data Loading Stage")
                data_ingestion_process = DataIngestion(config.get_data_ingestion_config())
                data_ingestion_process.get_raw_data()
                st.write("Data Processing Stage")
                data_processing_process = DataProcessing(config.get_data_processing_config())
                data_processing_process.get_processed_data()
                data_processing_process.get_split_data()
                data_processing_process.save_tokenizer()
                st.write("Model Training Stage")
                model_training_process = TrainModel(config.get_train_model_config())
                model_training_process.train_model()
                st.write("Model Evaluation Stage")
                model_evaluation_process = EvaluateModel(config.get_evaluate_model_config())
                model_result = model_evaluation_process.get_model_evaluation()
                
                status.update(label="Model Trained Succesfully", state="complete", expanded=False)
 
            st.success('Done!')


    if os.path.isdir(f"./artifacts/models/{model_name}/model"):
        if model_result != None:

            result_col1, result_col2 = st.columns(2)
                
            with result_col1:
                st.write("Model Parameters")
                params_data = read_yaml(Path("params.yaml"))
                print("last putting === ", params_data)
                st.write(params_data["model_params"])

            with result_col2:
                st.write("Model Result")
                st.write(model_result)
                    
        st.write("Try Trained Model Output")

        tokenizer_path = os.path.join("./artifacts","models",model_name, "tokenizer")
        model_path = os.path.join("./artifacts","models",model_name, "model")

        pretrained_tokenizer = load_tokenizer(tokenizer_path)
        pretrained_model = load_model(model_path)

        input_text = st.text_input("Try any sentence.. ",None, placeholder= "Write text here..")
        if input_text:
            tokenized_text = pretrained_tokenizer(input_text,
                             truncation=True,
                             is_split_into_words=False,
                             return_tensors='pt')
            
            outputs = pretrained_model(tokenized_text["input_ids"])
            predicted_label = labels[outputs.logits.argmax(-1)]

            st.write("Prediction")
           
            st.write("The prediction is", predicted_label)


        with open(f"{model_name}.zip", "rb") as fp:
            btn = st.download_button(
                label="Download Trained Model",
                data=fp,
                file_name=f"{model_name}.zip",
                mime="application/octet-stream"
                )
            
footer = """<style>.footer {position: fixed;left: 0;bottom: 0;width: 100%;background-color: #000;color: white;text-align: center;}
</style><div class='footer'><p>Made By Gourav Chouhan</p></div>"""
st.markdown(footer, unsafe_allow_html=True)
            


# st.caption("Made by Gourav Chouhan ")
