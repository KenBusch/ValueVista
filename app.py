import streamlit as st
import pandas as pd
import pickle  # If your models are pickled

# CSS to inject contained in a string
background_image_style = """
<style>
.stApp {
    background-image: url("DALL·E 2024-02-26 13.02.13 - Visualize a crystal-clear scene where a man is standing beside a car, his face turned towards it, clearly visible and marked by an expression of admir.webp");
    background-size: cover;
}
</style>
"""

st.markdown(background_image_style, unsafe_allow_html=True)


# preprocessor = pickle.load(open('path_to_preprocessor.pkl', 'rb'))
# rf_model = pickle.load(open('path_to_rf_model.pkl', 'rb'))

# Load your fitted preprocessor and model
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
with open('my_rf_model.pkl', 'rb') as f:
    pickled_model = pickle.load(f)


# Center-aligned title and subtitle
st.markdown("<h1 style='text-align: center;'>ValueVista</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Your Roadmap to the Best Car Deals</h2>", unsafe_allow_html=True)

# Introduction markdown
st.markdown('Welcome to ValueVista, a predictive tool for estimating the market value of used cars!')
st.markdown('Please enter the details of your car to get an estimated market price:')
            
# Input fields
st.header("Car Details")

# Using columns to organize inputs
col1, col2 = st.columns(2)

with col1:
    manufacturer = st.selectbox('Manufacturer', ['BMW', 'Audi', 'Toyota', 'Honda', 'Ford', 'Others'])
    model = st.text_input('Model')
    condition = st.selectbox('Condition', ['New', 'Like New', 'Excellent', 'Good', 'Fair', 'Salvage'])
    cylinders = st.selectbox('Cylinders', ['3 cylinders', '4 cylinders', '5 cylinders', '6 cylinders', '8 cylinders', '10 cylinders', '12 cylinders'])
    fuel = st.selectbox('Fuel Type', ['Gas', 'Diesel', 'Electric', 'Hybrid', 'Other'])
    paint_color = st.selectbox('Paint Color', ['Black', 'White', 'Grey', 'Red', 'Blue', 'Green', 'Yellow', 'Orange', 'Purple', 'Brown', 'Other'])
    car_age = st.number_input('Car Age', min_value=0)
    
with col2:
    odometer = st.number_input('Odometer', min_value=0)
    title_status = st.selectbox('Title Status', ['Clean', 'Salvage', 'Rebuilt', 'Parts Only', 'Lien', 'Missing'])
    transmission = st.selectbox('Transmission', ['Automatic', 'Manual', 'Other'])
    drive = st.selectbox('Drive', ['FWD', 'RWD', '4WD'])
    type_ = st.selectbox('Type', ['Sedan', 'SUV', 'Truck', 'Minivan', 'Coupe', 'Wagon', 'Convertible', 'Sports Car', 'Other'])
    lat = st.number_input('Latitude', format='%f')
    long = st.number_input('Longitude', format='%f')


# Button to predict price
if st.button("Predict Car Value"):
    # Aggregating user input into a dictionary
    my_car = {
        'manufacturer': manufacturer,
        'model': model,
        'condition': condition,
        'cylinders': cylinders,
        'fuel': fuel,
        'odometer': odometer,
        'title_status': title_status,
        'transmission': transmission,
        'drive': drive,
        'type': type_,
        'paint_color': paint_color,
        'lat': lat,
        'long': long,
        'car_age': car_age
    }
    
    # Converting dictionary to DataFrame
    my_car_df = pd.DataFrame([my_car])
    
    pickled_model = pickle.load(open('my_rf_model.pkl', 'rb'))

    my_car_preprocessed = preprocessor.transform(my_car_df)

    pred = pickled_model.predict(my_car_preprocessed)

    #final
    #end = ' USD '
    #final = (str(res) + end)
    
    #st.title(final)

    # Formatting the result to two decimal places
    res = f"{pred[0]:,.2f}"

    # Displaying the estimated value with a preceding sentence
    st.title(f"Your estimated car value is ${res} USD")
    
    #final = (str(res) + end)
    
    #st.title(final)

    # Displaying the estimated value
    #st.write(f"Estimated Market Value: ${predicted_price_rf[0]:,.2f}")
