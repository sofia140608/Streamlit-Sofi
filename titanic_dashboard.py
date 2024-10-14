import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px

# Set up Streamlit page configuration
st.set_page_config(page_title="Titanic Data Dashboard", layout="wide")

# Load the Titanic dataset
@st.cache_data
def load_data():
    return sns.load_dataset("titanic")

titanic = load_data()

# Sidebar navigation
page = st.sidebar.selectbox(
    "Select a Page", 
    ["Overview", "Survival Analysis", "Passenger Distribution", "About"]
)

# Page 1: Overview
if page == "Overview":
    st.title("Titanic Dataset Overview")
    st.write("This page provides a high-level overview of the Titanic dataset.")
    
    # Show the first few rows of the dataset
    st.subheader("Data Preview")
    st.dataframe(titanic.head())

    # Show some basic statistics
    st.subheader("Key Metrics")
    st.metric("Total Passengers", value=titanic.shape[0])
    st.metric("Survived Passengers", value=titanic["survived"].sum())
    st.metric("Survival Rate", value=f"{titanic['survived'].mean() * 100:.2f}%")

    # Display survival counts by class
    st.subheader("Passenger Class Distribution")
    class_counts = titanic["class"].value_counts()
    st.bar_chart(class_counts)

# Page 2: Survival Analysis
elif page == "Survival Analysis":
    st.title("Survival Analysis")
    st.write("This page explores the survival rate based on various factors.")

    # Create columns
    col1, col2 = st.columns(2)

    with col1:
        # Survival rate by gender
        st.subheader("Survival Rate by Gender")
        fig = px.bar(
            titanic.groupby("sex")["survived"].mean().reset_index(),
            x="sex", y="survived", color="sex", barmode="group",
            labels={"survived": "Survival Rate"}, title="Survival Rate by Gender"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Survival rate by class
        st.subheader("Survival Rate by Passenger Class")
        fig = px.bar(
            titanic.groupby("class")["survived"].mean().reset_index(),
            x="class", y="survived", color="class",
            labels={"survived": "Survival Rate"}, title="Survival Rate by Class"
        )
        st.plotly_chart(fig, use_container_width=True)

# Page 3: Passenger Distribution
elif page == "Passenger Distribution":
    st.title("Passenger Distribution")
    st.write("This page visualizes the distribution of various passenger attributes.")

    # Use columns to split space
    col1, col2 = st.columns(2)

    with col1:
        # Age distribution
        st.subheader("Distribution of Age")
        fig = px.histogram(
            titanic, x="age", nbins=30, title="Passenger Age Distribution",
            labels={"age": "Age"}, marginal="box"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Fare distribution
        st.subheader("Distribution of Fare")
        fig = px.histogram(
            titanic, x="fare", nbins=30, title="Passenger Fare Distribution",
            labels={"fare": "Fare ($)"}, marginal="box"
        )
        st.plotly_chart(fig, use_container_width=True)

# Page 4: About
elif page == "About":
    st.title("About this Dashboard")
    st.write("""
        This dashboard was created to analyze the Titanic dataset, providing insights into passenger survival rates and attributes.
        
        **Dataset Source:** Seaborn's Titanic dataset.
        
        **Pages Overview:**
        - **Overview:** Summary of the dataset and key metrics.
        - **Survival Analysis:** Explore survival rates based on gender and class.
        - **Passenger Distribution:** Visualize distributions of passenger age and fare.
        - **About:** Information about this dashboard and its purpose.
        
        **Created by:** Sofía Álvarez
    """)
