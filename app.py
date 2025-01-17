import os  
import streamlit as st
import numpy as np
import pickle
from datasets import load_from_disk
import matplotlib.pyplot as plt
import random
from PIL import Image
import pandas as pd
import base64

st.set_page_config(page_title="SAE Demo", layout="wide")

def plot_images(images_list, captions_list):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))  
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx < len(images_list):
            image = images_list[idx]
            if isinstance(image, Image.Image):
                image = np.array(image)

            ax.imshow(image)
            ax.set_title(captions_list[idx], fontsize=12, pad=10)
        ax.axis('off')

    plt.tight_layout(pad=3.0)
    st.pyplot(fig)

def load_images_for_neuron(option):
    dataset = load_from_disk("mini-imagenet")
    dataset = dataset.shuffle(seed=35)

    with open(f"neuron_data/{option}.pkl", "rb") as f:
        neurons_topk = pickle.load(f)

    neurons_topk = {k: v for k, v in neurons_topk.items() if len(v) >= 16}

    neuron_idx = random.choice(list(neurons_topk.keys()))

    topk = sorted(neurons_topk[neuron_idx], key=lambda x: -x[2])[:16]

    images_list = []
    captions_list = []
    for idx, label, act in topk:
        idx -= 1  
        if idx < len(dataset):
            images_list.append(dataset[idx]['image'])
            captions_list.append(f"Label: {label}, Act: {round(act, 2)}")
        else:
            st.warning(f"Index {idx} out of range for dataset.")

    st.write(f"Showing images for neuron: {neuron_idx}")
    plot_images(images_list, captions_list)
    return neuron_idx

def load_markdown(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"Markdown file not found: {file_path}")
        return ""

def main_page():
    st.markdown("<h1 style='text-align: center;'>SAE Trained on CLIP Demo</h1>", unsafe_allow_html=True)

    markdown_content = load_markdown("docs/main.md")
    st.markdown(markdown_content, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    flag = False
    with col1:
        option = st.selectbox(
            "Choose model iteration",
            ("model_1", "model_2", "model_3"),
        )        
    with col2:        
        if st.button("Top 16 activating images for random neuron"):
            flag=True
    if flag:
        load_images_for_neuron(option)
    

    
def formulation_page():
    st.markdown("<h1 style='text-align: center;'>Formulation</h1>", unsafe_allow_html=True)
    markdown_content = load_markdown("docs/formulation.md")
    st.markdown(markdown_content, unsafe_allow_html=True)

def implementation_page():
    st.markdown("<h1 style='text-align: center;'>Implementation</h1>", unsafe_allow_html=True)
    markdown_content = load_markdown("docs/implementation.md")
    st.markdown(markdown_content, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("docs/first_loss.png", caption="model 1 loss (FVU+alpha*auxk)")
    with col2:
        st.image("docs/second_loss.png", caption="model 2 loss (FVU+alpha*auxk)")    
    with col3:
        st.image("docs/third_loss.png", caption="model 3 loss (FVU+alpha*auxk)")    



def evaluation_page():
    st.markdown("<h1 style='text-align: center;'>Evaluation</h1>", unsafe_allow_html=True)
    markdown_content = load_markdown("docs/evaluation/evaluation1.md")
    st.markdown(markdown_content, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image("docs/oriental_designs.png", caption="Feature generalizing beyond class labels, in this case to oriental designs or intricate patterns")
    with col2:
        st.image("docs/green_thing.png", caption="Feature locking onto class label")
    
    markdown_content = load_markdown("docs/evaluation/evaluation2.md")
    st.markdown(markdown_content, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image("docs/histogram.png")
    with col2:
        st.image("docs/highest_avg_act.png", width=400, caption="Images associated with highest avg. activation neuron in imagenet-mini")    

    
    markdown_content = load_markdown("docs/evaluation/evaluation3.md")
    st.markdown(markdown_content, unsafe_allow_html=True)
    df = pd.DataFrame(
    [["29.09%", "4.6%", "780 (25.3%)"]], columns=("FVU", "auxilliary FVU of K dead neurons", "# neurons featured in top 16 activations of image in imagenet-mini"))
    st.table(df)

    markdown_content = load_markdown("docs/evaluation/evaluation4.md")
    st.markdown(markdown_content, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image("docs/without_sae.png", caption="out of the box CLIP performance")
    with col2:
        st.image("docs/with_sae.png", caption="with SAE inserted")    


def future_work_page():
    st.markdown("<h1 style='text-align: center;'>Future Work</h1>", unsafe_allow_html=True)
    markdown_content = load_markdown("docs/future_work.md")
    st.markdown(markdown_content, unsafe_allow_html=True)

def hitchhiker_page():
    st.markdown("<h1 style='text-align: center;'>Hitchhiker Test</h1>", unsafe_allow_html=True)
    markdown_content = load_markdown("docs/hitchhiker.md")
    st.markdown(markdown_content, unsafe_allow_html=True)
    pdf_path = "hitchhiker.pdf"
    
    with open(pdf_path, "rb") as pdf_file:
        base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
    
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)



st.sidebar.title("Sections")
    
page = st.sidebar.radio(" ", ["Main", "Formulation", "Implementation", "Evaluation", "Future Work", "Hitchhiker Analysis"])

if page == "Main":
    main_page()
elif page == "Formulation":
    formulation_page()
elif page == "Implementation":
    implementation_page()
elif page == "Evaluation":
    evaluation_page()
elif page == "Future Work":
    future_work_page()
elif page=="Hitchhiker Analysis":
    hitchhiker_page()

