import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

def load_model_and_data():
    """Load the trained model and dataset"""
    model = joblib.load('polynomialRegModel.pkl')
    data = pd.read_csv('sustainability_data.csv')
    return model, data

def create_prediction_plot(y_true, y_pred):
    """Create scatter plot of predictions vs actual values"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot scatter points
    ax.scatter(y_true, y_pred, color='blue', alpha=0.5, label='Predictions')
    
    # Plot perfect prediction line
    line_range = np.linspace(min(y_true), max(y_true), 100)
    ax.plot(line_range, line_range, color='red', label='Perfect Prediction')
    
    ax.set_xlabel('Actual CO2 Emissions')
    ax.set_ylabel('Predicted CO2 Emissions')
    ax.set_title('Predicted vs Actual CO2 Emissions')
    ax.legend()
    
    return fig

def main():
    st.set_page_config(page_title="CO2 Emissions Predictor", layout="wide")
    
    st.title("🌍 Sustainability Metrics Analyzer")
    st.write("Predict CO2 emissions based on energy consumption, renewable percentage, and GDP")
    
    try:
        # Load model and data
        model, data = load_model_and_data()
        poly = PolynomialFeatures(degree=1)  # Same degree as training
        
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["Predictor", "Model Performance", "Data Analysis"])
        
        with tab1:
            st.subheader("Make Predictions")
            
            # Input columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                energy = st.number_input(
                    "Energy Consumption",
                    min_value=float(data['Energy_Consumption'].min()),
                    max_value=float(data['Energy_Consumption'].max()),
                    value=float(data['Energy_Consumption'].mean())
                )
                
            with col2:
                renewable = st.number_input(
                    "Renewable Percentage",
                    min_value=float(data['Renewable_Percentage'].min()),
                    max_value=float(data['Renewable_Percentage'].max()),
                    value=float(data['Renewable_Percentage'].mean())
                )
                
            with col3:
                gdp = st.number_input(
                    "GDP",
                    min_value=float(data['GDP'].min()),
                    max_value=float(data['GDP'].max()),
                    value=float(data['GDP'].mean())
                )
            
            if st.button("Predict CO2 Emissions"):
                # Prepare input data
                input_data = np.array([[energy, renewable, gdp]])
                input_poly = poly.fit_transform(input_data)
                
                # Make prediction
                prediction = model.predict(input_poly)[0]
                
                st.success(f"Predicted CO2 Emissions: {prediction:.2f}")
                
                # Add sustainability recommendations
                if prediction > data['CO2_Emissions'].mean():
                    st.warning("⚠️ High CO2 emissions predicted! Consider these recommendations:")
                    st.markdown("""
                    - Increase renewable energy usage
                    - Implement energy efficiency measures
                    - Review and optimize energy consumption patterns
                    - Consider carbon offset programs
                    """)
        
        with tab2:
            st.subheader("Model Performance")
            
            # Calculate predictions for all data
            X_all = data[['Energy_Consumption', 'Renewable_Percentage', 'GDP']]
            X_poly_all = poly.fit_transform(X_all)
            y_pred_all = model.predict(X_poly_all)
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                mse = mean_squared_error(data['CO2_Emissions'], y_pred_all)
                st.metric("Mean Squared Error", f"{mse:.2f}")
            
            with col2:
                r2 = r2_score(data['CO2_Emissions'], y_pred_all)
                st.metric("R² Score", f"{r2:.4f}")
            
            # Plot predictions vs actual
            st.pyplot(create_prediction_plot(data['CO2_Emissions'], y_pred_all))
        
        with tab3:
            st.subheader("Data Analysis")
            
            # Show correlation matrix
            st.write("Feature Correlations")
            corr_matrix = data[['Energy_Consumption', 'Renewable_Percentage', 'GDP', 'CO2_Emissions']].corr()
            st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))
            
            # Show dataset
            st.write("Dataset Preview")
            st.dataframe(data.head())
            
            # Download feature
            if st.button("Download Sample Predictions"):
                predictions_df = pd.DataFrame({
                    'Actual_CO2': data['CO2_Emissions'],
                    'Predicted_CO2': y_pred_all,
                    'Energy_Consumption': data['Energy_Consumption'],
                    'Renewable_Percentage': data['Renewable_Percentage'],
                    'GDP': data['GDP']
                })
                
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="sustainability_predictions.csv",
                    mime="text/csv"
                )
                
    except FileNotFoundError:
        st.error("Please ensure 'polynomialRegModel.pkl' and 'sustainability_data.csv' are in the same directory as the app.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()