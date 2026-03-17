

import os, time, requests, numpy as np, pandas as pd, plotly.graph_objects as go,  torch, torch.nn as nn




def safe_get(url, params, max_retries=10):
    for attempt in range(max_retries):
        try: return requests.get(url, params=params, timeout=10)
        except requests.exceptions.RequestException:
            wait = min(2 ** attempt, 30)
            print(f"Binance is napping... retrying in {wait}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
    raise Exception("Binance did not wake up after all retries.")

def fetch_binance_daily_all(symbol="BTCUSDT"):
    url = "https://api.binance.com/api/v3/klines"
    interval = "1d"
    limit = 1000
    all_rows = []
    start_time = 0
    while True:
        params = {"symbol": symbol, "interval": interval, "limit": limit, "startTime": start_time}
        response = safe_get(url, params=params)
        data = response.json()
        if not data: break
        all_rows.extend(data)
        last_close_time = data[-1][6]
        start_time = last_close_time + 1
        time.sleep(0.2)
    df = pd.DataFrame(all_rows, columns=["open_time","open","high","low","close","volume","close_time","quote_asset_volume","num_trades","taker_buy_base","taker_buy_quote","ignore"])
    df = df.rename(columns={"open_time": "date", "volume": "vol"})
    df["date"] = pd.to_datetime(df["date"], unit="ms").dt.date
    numeric_cols = ["open","high","low","close","vol"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df[["date","open","high","low","close","vol"]]

DATA_FILE = "BTC_daily_data3.csv"
if os.path.exists(DATA_FILE):
    print("NB: You are using saved BTCUSDT, to save time fetching!")
    print("If you must fetch from Binance, just delete BTC_daily_data.csv.")
    BTC_data = pd.read_csv(DATA_FILE)
else:
    print("Fetching starting! This might take some minutes!")
    BTC_data = fetch_binance_daily_all()
    print("Fetching Finished! Your break is over lol!")
    BTC_data.to_csv(DATA_FILE, index=False)

if "hloc_avg" not in BTC_data.columns or "log_return" not in BTC_data.columns:
    BTC_data["hloc_avg"] = BTC_data[["high","low","open","close"]].mean(axis=1)
    BTC_data["log_return"] = np.log(BTC_data["hloc_avg"])

def rolling_ols_slope(series, window):
    slopes = []
    for i in range(len(series)):
        if i < window: slopes.append(np.nan)
        else:
            y = series.iloc[i - window + 1:i + 1].values
            x = np.arange(window)
            beta, _ = np.polyfit(x, y, 1)
            slopes.append(beta)
    return np.array(slopes)

BTC_data["ols_slope391"] = rolling_ols_slope(BTC_data["log_return"], 391)
BTC_data["ols_slope195"] = rolling_ols_slope(BTC_data["log_return"], 195)
BTC_data["ols_slope130"] = rolling_ols_slope(BTC_data["log_return"], 130)
BTC_data["ols_slope98"] = rolling_ols_slope(BTC_data["log_return"], 98)
BTC_data["ols_slope65"] = rolling_ols_slope(BTC_data["log_return"], 65)
BTC_data["ols_slope78"] = rolling_ols_slope(BTC_data["log_return"], 78)
BTC_data["ols_slope56"] = rolling_ols_slope(BTC_data["log_return"], 56)
BTC_data["ols_slope44"] = rolling_ols_slope(BTC_data["log_return"], 44)
BTC_data["ols_slope36"] = rolling_ols_slope(BTC_data["log_return"], 36)
BTC_data["ols_slope24"] = rolling_ols_slope(BTC_data["log_return"], 24)
BTC_data["ols_slope28"] = rolling_ols_slope(BTC_data["log_return"], 28)

s391 = BTC_data["ols_slope391"].values
s195 = BTC_data["ols_slope195"].values
s130 = BTC_data["ols_slope130"].values
s98 = BTC_data["ols_slope98"].values
s65 = BTC_data["ols_slope65"].values
s78 = BTC_data["ols_slope78"].values
s56 = BTC_data["ols_slope56"].values
s44 = BTC_data["ols_slope44"].values
s36 = BTC_data["ols_slope36"].values
s24 = BTC_data["ols_slope24"].values
s28 = BTC_data["ols_slope28"].values
X = np.stack([s391,s195,s130,s98,s65,s78,s56,s44,s36,s24,s28], 1).astype(np.float32)
#torch.save(model.state_dict(), "multi_transformer_weights.pth")#save
class MultiTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = 64 * 3
        self.inp = nn.Linear(11, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4 * 3,
            dim_feedforward=128 * 3,
            batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=1)
        self.buy_head  = nn.Linear(d_model, 1)
        self.sell_head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.unsqueeze(1)              # (B,1,11)
        h = self.enc(self.inp(x))       # (B,1,192)
        h = h[:, 0]                     # (B,192)
        return self.buy_head(h).squeeze(-1), self.sell_head(h).squeeze(-1)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTransformer().to(device)

state = torch.load("multi_transformer_weights.pth", map_location=device)
model.load_state_dict(state)

model.eval()


with torch.no_grad():
    xb = torch.from_numpy(X).to(device)
    out_buy, out_sell = model(xb)
    buy_prob  = torch.sigmoid(out_buy).cpu().numpy()
    sell_prob = torch.sigmoid(out_sell).cpu().numpy()
buy_hat  = (buy_prob  > 0.5).astype(int)
sell_hat = (sell_prob > 0.5).astype(int)

print("BUY predictions:",  buy_hat.sum())
print("SELL predictions:", sell_hat.sum())
# ---------- PLOT ----------
fig = go.Figure()
fig.add_trace(go.Scatter(x=BTC_data["date"], y=BTC_data["close"], mode="lines", name="Price", line=dict(color="white")))
fig.add_trace(go.Scatter(x=BTC_data["date"][buy_hat == 1], y=BTC_data["close"][buy_hat == 1], mode="markers", name="BUY", marker=dict(color="cyan", size=8, symbol="triangle-up")))
fig.add_trace(go.Scatter(x=BTC_data["date"][sell_hat == 1], y=BTC_data["close"][sell_hat == 1], mode="markers", name="SELL", marker=dict(color="orange", size=8, symbol="triangle-down")))
fig.update_layout(template="plotly_dark", title="BUY + SELL Predictions (Transformer 11→2)")
fig.show()