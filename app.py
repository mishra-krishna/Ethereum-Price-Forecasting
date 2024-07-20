from flask import Flask, jsonify
from model import load_model, predict, cross_validate_model

app = Flask(__name__)

@app.route('/')
def home():
    return 'Welcome to Ethereum Price Prediction Model, Try /predict for predictions!'

@app.route('/predict', methods=['GET'])
def get_prediction():
    model = load_model()  
    forecast = predict(model)
    forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records')
    
    return jsonify(forecast_data)

@app.route('/performance', methods=['GET'])
def get_performance():
    model = load_model()  
    performance = cross_validate_model(model)
    mean_mape = performance['mape'].mean()
    
    return jsonify({'mean_mape': mean_mape})

if __name__ == '__main__':
    app.run(debug=True)