import streamlit as st
import pandas as pd
import pickle  # If your models are pickled

# CSS to inject contained in a string
background_image_style = """
<style>
.stApp {
    background-image: url("BackgroundImage");
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

# Introduction markdown with center alignment
st.markdown("<p style='text-align: center;'>Welcome to ValueVista, a predictive tool for estimating the market value of used cars!</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Please enter the details of your car to get an estimated market price:</p>", unsafe_allow_html=True)
           
# Input fields
#st.header("Car Details")

# Using columns to organize inputs
col1, col2 = st.columns(2)

with col1:
    manufacturer = st.selectbox('Manufacturer', ['bmw', 'Audi', 'Toyota', 'Honda', 'Ford', 'Others'])
    model = st.selectbox('Model',['others'])
    condition = st.selectbox('Condition', ['excellent','New', 'Like New','Good', 'Fair', 'Salvage'])
    cylinders = st.selectbox('Cylinders', ['4 cylinders','3 cylinders','5 cylinders', '6 cylinders', '8 cylinders', '10 cylinders', '12 cylinders'])
    fuel = st.selectbox('Fuel Type', ['gas', 'Diesel', 'Electric', 'Hybrid', 'Other'])
    paint_color = st.selectbox('Paint Color', ['black', 'White', 'Grey', 'Red', 'Blue', 'Green', 'Yellow', 'Orange', 'Purple', 'Brown', 'Other'])
    car_age = st.number_input('Car Age', min_value=0, value=7)
    
with col2:
    odometer = st.number_input('Odometer', min_value=0, value=57000)
    title_status = st.selectbox('Title Status', ['clean', 'Salvage', 'Rebuilt', 'Parts Only', 'Lien', 'Missing'])
    transmission = st.selectbox('Transmission', ['automatic', 'Manual', 'Other'])
    drive = st.selectbox('Drive', ['4wd','FWD', 'RWD'])
    type_ = st.selectbox('Type', ['SUV', 'Sedan', 'Truck', 'Minivan', 'Coupe', 'Wagon', 'Convertible', 'Sports Car', 'Other'])
    lat = st.number_input('Latitude', format='%f', value=47.6061)
    long = st.number_input('Longitude', format='%f', value=122.3328)


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
