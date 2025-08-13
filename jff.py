import pandas as pd
from fifo_pnl_function import fifo_pnl_with_live_price


res = fifo_pnl_with_live_price("Stock_History/aug13.csv", "CRCL")