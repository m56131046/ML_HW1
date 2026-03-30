import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. 獲取資料 (使用 yfinance 抓取標普500指數)
print("正在獲取 (^GSPC) 資料...")
ticker = yf.Ticker('^GSPC')

# 使用參數 Period start='2021-01-01', end='2025-12-31'
data = ticker.history(start='2021-01-01', end='2025-12-31')

# 我們預測的目標是收盤價 'Close'
df = pd.DataFrame(data['Close'])
df.columns = ['Close']

# 去除時區資訊，避免索引錯誤
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)

print("正在進行特徵工程...")
# 計算每天的收盤價變化量 (Difference)，幫助樹狀模型避免外推問題
df['Diff'] = df['Close'].diff()

# 加入滯後特徵 (前1~5天的收盤價變化量) 作為預測依據
for i in range(1, 6):
    df[f'Lag_{i}'] = df['Diff'].shift(i)

# 保留前一天的實際收盤價，後續用於還原預測價
df['Prev_Close'] = df['Close'].shift(1)

# 移除因為滯後特徵與差異計算產生缺失值的資料列
df.dropna(inplace=True)

# 目標值是我們預測的價格變化量 (今天的變化量)
df['Target'] = df['Diff']

# 2. 劃分訓練集(Training Set)與測試集(Testing Set)
# Training Set: 2021-01-01 到 2024-12-31
# Testing Set: 2025-01-01 到 2025-12-31
print("正在劃分資料集...")
train_data = df.loc['2021-01-01':'2024-12-31']
test_data = df.loc['2025-01-01':'2025-12-31']

features = [f'Lag_{i}' for i in range(1, 6)]
X_train = train_data[features]
y_train = train_data['Target']
X_test = test_data[features]
y_test = test_data['Target']

# 3. 訓練 Random Forest 模型
print("正在訓練 Random Forest 模型...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. 進行預測 (預測結果為 "價格的變化量")
train_diff_predictions = model.predict(X_train)
test_diff_predictions = model.predict(X_test)

# 還原預測結果: 預測收盤價 = 前一天實際收盤價 + 預測變化量
train_predictions = train_data['Prev_Close'] + train_diff_predictions
test_predictions = test_data['Prev_Close'] + test_diff_predictions

# 5. 計算 Mean Squared Error (MSE) (還原後的收盤價 vs 實際收盤價)
train_mse = mean_squared_error(train_data['Close'], train_predictions)
test_mse = mean_squared_error(test_data['Close'], test_predictions)

print("-" * 30)
print(f"訓練集 MSE (Training Set MSE): {train_mse:.4f}")
print(f"測試集 MSE (Testing Set MSE): {test_mse:.4f}")
print("-" * 30)

# 6. 畫出預測結果跟實際值的圖
print("正在產生預測結果圖表...")
plt.figure(figsize=(14, 7))

# 畫出訓練集的資料
plt.plot(train_data.index, train_data['Close'], label='Training Data (Actual)', color='blue', alpha=0.6)
plt.plot(train_data.index, train_predictions, label='Training Data (Predicted)', color='cyan', linestyle='dashed', alpha=0.8)

# 畫出測試集的資料
plt.plot(test_data.index, test_data['Close'], label='Testing Data (Actual)', color='orange')
plt.plot(test_data.index, test_predictions, label='Testing Data (Predicted)', color='red', linestyle='dashed')

plt.title('S&P 500 (^GSPC) Random Forest Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.tight_layout()

# 存檔圖片
output_image = 'sp500_rf_prediction.png'
plt.savefig(output_image)
print(f"圖表已成功存檔為: {output_image}")
