import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 주식 데이터 가져오기
def fetch_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# 주가 예측 모델 학습
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

# 주가 예측 그래프 그리기 및 예측값 반환
def plot_predictions(model, X_test, y_test):
    predictions = model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, label='Predicted')
    plt.plot(y_test, label='Actual')
    plt.legend()
    plt.title('Stock Price Prediction')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.show()
    return predictions

# 메인 함수
def main():
    symbol = input("Enter stock symbol (e.g., AAPL): ")
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")

    # 주식 데이터 가져오기
    stock_data = fetch_stock_data(symbol, start_date, end_date)

    # 데이터 전처리
    stock_data['NextClose'] = stock_data['Close'].shift(-1)  # 다음날 종가
    stock_data.dropna(inplace=True)
    X = np.array(stock_data['Close']).reshape(-1, 1)
    y = np.array(stock_data['NextClose'])

    # 주가 예측 모델 학습
    model, X_test, y_test = train_model(X, y)

    # 주가 예측 그래프 그리기 및 예측값 반환
    predictions = plot_predictions(model, X_test, y_test)

    # 다음날 주가 예측값 출력
    next_day_prediction = predictions[-1]
    print("Predicted price for the next day:", next_day_prediction)

if __name__ == "__main__":
    main()
