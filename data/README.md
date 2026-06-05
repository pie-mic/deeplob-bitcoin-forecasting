# Data

The raw BTCUSDT.P dataset used in this project cannot be redistributed here due to file size (~300–500 MB).

## Download

The dataset is publicly available on Kaggle:

**https://www.kaggle.com/datasets/siavashraz/bitcoin-perpetualbtcusdtp-limit-order-book-data**

Download the CSV, rename it to `BTCUSDT_LOB_250ms_Jan2023.csv`, and update `DATA_PATH` in notebook cell 5 to point to your local copy.

## Dataset Description

**Asset:** Bitcoin perpetual futures (BTCUSDT.P)  
**Exchange:** Binance  
**Sampling:** Uniform 250 ms clock-time snapshots  
**Window:** 9–15 January 2023 (5 full trading days retained after boundary trimming: 10–14 Jan)  
**Size:** ~1,723K snapshots  
**Features:** Top 10 bid and ask levels, each with price and volume → 40 features per snapshot

## Expected CSV Format

| Column | Description |
|---|---|
| `row_idx` | Integer row index |
| `datetime` | Timestamp (parseable by pandas) |
| `bid_p{1..10}` | Bid prices, level 1 = best bid |
| `bid_v{1..10}` | Bid volumes |
| `ask_p{1..10}` | Ask prices, level 1 = best ask |
| `ask_v{1..10}` | Ask volumes |

The notebook reorders columns into DeepLOB format `{ask_p_i, ask_v_i, bid_p_i, bid_v_i}` for i = 1..10, so that the first convolutional layer's (1×2) stride correctly pairs each price with its own volume at the same level.
