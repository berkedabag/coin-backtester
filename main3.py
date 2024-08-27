import hashlib
import hmac
import json
import logging
import os
import time
import pandas as pd
import pymongo
import requests
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator, StochRSIIndicator, AwesomeOscillatorIndicator, TSIIndicator
import numpy as np
import requests
from zipfile import ZipFile, ZIP_DEFLATED
import io
import pandas as pd
from datetime import datetime
import threading
from dotenv import load_dotenv
from decimal import Decimal, getcontext
from concurrent.futures import ThreadPoolExecutor, as_completed

db_lock = threading.Lock()
load_dotenv()
pd.set_option('display.precision', 25)
pd.set_option('display.float_format', '{:.25}'.format)
getcontext().prec = 28

os.environ["OMP_NUM_THREADS"] = "32"
os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["VECLIB_MAXIMUM_THREADS"] = "32"
os.environ["NUMEXPR_NUM_THREADS"] = "32"


class DatabaseManager:
    def __init__(self, db_name, connection_string, coin_pairs, intervals):
        self.client = pymongo.MongoClient(connection_string, maxPoolSize=100)
        self.coin_pairs = coin_pairs
        self.db_name = db_name
        self.db = self.client[self.db_name]
        self.intervals = intervals
        self.clear_database()
        self.initialize_database()

    def clear_database(self):
        try:
            self.client.drop_database(self.db_name)
            logging.info(f"Database {self.db_name} cleared.")
        except Exception as e:
            logging.error(f"Error in clear_database: {e}")

    def initialize_database(self):
        try:
            for coin_pair in self.coin_pairs:
                for interval in self.intervals:
                    collection_name = f"coin_{coin_pair}_{interval}_kline_data"
                    self.db[collection_name].create_index(
                        [("close_time", pymongo.DESCENDING),
                         ("interval", pymongo.ASCENDING)]
                    )

            self.db["trading_opportunities"].create_index(
                [("coin_pair", pymongo.ASCENDING), ("interval",
                                                    pymongo.ASCENDING), ("status", pymongo.ASCENDING)]
            )
            self.db["pnl_history"].create_index(
                [("timestamp", pymongo.DESCENDING),
                 ("interval", pymongo.ASCENDING)]
            )
            self.db["tracking_values"].create_index(
                [("timestamp", pymongo.DESCENDING),
                 ("interval", pymongo.ASCENDING)]
            )
            self.db["slots"].create_index(
                [("interval", pymongo.ASCENDING), ("algo_type",
                                                   pymongo.ASCENDING), ("slot_no", pymongo.ASCENDING)],
                unique=True
            )
        except Exception as e:
            logging.error(f"Error in initialize_database: {e}")

    def insert_to_database(self, df, coin_name, interval):
        try:
            collection_name = f"coin_{coin_name}_{interval}_kline_data"
            records = df.to_dict(orient='records')
            self.db[collection_name].insert_many(records)
            logging.info(
                f"Inserted {len(records)} records into {collection_name}")
        except Exception as e:
            logging.error(f"Error in insert_to_database: {e}")

    def record_opportunity(self, data):
        try:
            data['entry_fee'] = data['pos_size'] * 0.0002
            data['adjusted_pos_size'] = data['pos_size'] - data['entry_fee']
            self.db["trading_opportunities"].insert_one(data)
        except Exception as e:
            logging.error(f"Error in record_opportunity: {e}")

    def close_opportunity_and_record_pnl(self, id, exit_price, pnl, human_readable_date, actual_pnl_usd, reason, close_timestamp, closed_pos_size, status):
        try:
            update_data = {
                "status": status,
                "exit_price": exit_price,
                "pnl": pnl,
                "human_readable_close_date": human_readable_date,
                "actual_pnl_usd": actual_pnl_usd,
                "reason": reason,
                "close_timestamp": close_timestamp,
                "closed_pos_size": closed_pos_size
            }
            self.db["trading_opportunities"].update_one(
                {"_id": id},
                {"$set": update_data}
            )
        except Exception as e:
            logging.error(f"Error in close_opportunity_and_record_pnl: {e}")

    def initialize_slots(self, interval_list, algo_types, max_positions, total_init_capital):
        try:
            slot_size = total_init_capital / max_positions
            operations = [
                pymongo.InsertOne({
                    "interval": interval,
                    "algo_type": algo_type,
                    "slot_no": slot_no,
                    "slot_size": slot_size,
                    "occupied": False,
                    "latest_operated_coin": None,
                    "previous_value": None,
                    "latest_trading_opportunity_id": None
                })
                for interval in interval_list
                for algo_type in algo_types
                for slot_no in range(1, max_positions + 1)
            ]
            if operations:
                with db_lock:
                    self.db["slots"].bulk_write(operations)
        except Exception as e:
            logging.error(f"Error in initialize_slots: {e}")

    def fetch_and_occupy_slot(self, interval, algo_type, latest_operated_coin, latest_trading_opportunity_id):
        try:
            slot = self.db["slots"].find_one(
                {
                    "interval": interval,
                    "algo_type": algo_type,
                    "occupied": False,
                    "slot_size": {"$gt": 1}
                },
                sort=[("slot_size", pymongo.DESCENDING)]
            )

            if slot:
                previous_value = slot['slot_size']

                # Fetch the latest open and close prices for the coin
                coin_data = self.db[f"coin_{latest_operated_coin}_{interval}_kline_data"].find_one(
                    sort=[("close_time", pymongo.DESCENDING)]
                )

                open_price = coin_data.get('open') if coin_data else 'N/A'
                close_price = coin_data.get('close') if coin_data else 'N/A'

                slot = self.db["slots"].find_one_and_update(
                    {
                        "interval": interval,
                        "algo_type": algo_type,
                        "occupied": False,
                        "slot_size": {"$gt": 1},
                        "_id": slot["_id"]  # Ensure we're updating the correct slot
                    },
                    {
                        "$set": {
                            "occupied": True,
                            "latest_usage_human_readable": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "update_reason": "occupying_slot",
                            "latest_operated_coin": latest_operated_coin,
                            "previous_value": previous_value,
                            "latest_trading_opportunity_id": latest_trading_opportunity_id,
                            "open_price": open_price,  # Add open price
                            "close_price": close_price  # Add close price
                        }
                    },
                    return_document=pymongo.ReturnDocument.AFTER
                )
                return slot['slot_no'], slot['slot_size'], previous_value
            else:
                logging.info(f"No available slot for {interval} and {algo_type}")
                return None
        except Exception as e:
            logging.error(f"Error in fetch_and_occupy_slot: {e}")
            return None


    def release_slot(self, interval, algo_type, slot_no, adjustment_value, latest_operated_coin, latest_trading_opportunity_id):
        try:
            with db_lock:
                current_time = time.time()
                human_readable_time = datetime.fromtimestamp(
                    current_time).strftime('%Y-%m-%d %H:%M:%S')

                # Fetch the current slot size
                slot = self.db["slots"].find_one({
                    "interval": interval,
                    "algo_type": algo_type,
                    "slot_no": slot_no
                })

                if slot:
                    new_slot_size = slot['slot_size'] + adjustment_value
                    previous_value = slot["slot_size"]
                    coin_collection_name = f"coin_{latest_operated_coin}_{interval}_kline_data"
                    open_price = slot["open_price"]
                    close_price = slot["close_price"]
                
                    # Check if the new slot size is negative
                    if new_slot_size < 0:
                        logging.error(
                            f"Cannot release slot {slot_no} for {algo_type} at {interval} {new_slot_size}: adjustment would result in negative slot size. "
                            f"Coin: {latest_operated_coin}, Open Price: {open_price}, Close Price: {close_price}"
                        )
                        self.db["slots"].find_one_and_update(
                            {
                                "interval": interval,
                                "algo_type": algo_type,
                                "slot_no": slot_no,
                            },
                            {
                                "$set": {
                                    "occupied": True,
                                    "update_reason": "unavailable money",
                                    "latest_usage_human_readable": human_readable_time,
                                    "slot_size": new_slot_size
                                }
                            }
                        )
                        return

                    # If the new slot size exceeds $5000, split the slot
                    if new_slot_size > 5000:
                        half_slot_size = new_slot_size / 2
                        result = self.db["slots"].find_one_and_update(
                            {
                                "interval": interval,
                                "algo_type": algo_type,
                                "slot_no": slot_no
                            },
                            {
                                "$set": {
                                    "slot_size": half_slot_size,
                                    "occupied": False,
                                    "latest_usage_human_readable": human_readable_time,
                                    "update_reason": "releasing_slot",
                                    "latest_operated_coin": latest_operated_coin,
                                    "previous_value": previous_value,
                                    "latest_trading_opportunity_id": latest_trading_opportunity_id
                                }
                            },
                            return_document=pymongo.ReturnDocument.AFTER
                        )
                        if result:
                            # Create a new slot with the other half
                            new_slot_no = self.create_new_slot(
                                interval, algo_type, half_slot_size, latest_operated_coin, latest_trading_opportunity_id)
                            logging.info(
                                f"Slot {slot_no} split into {slot_no} and {new_slot_no}")
                        else:
                            logging.warning(
                                f"No slot found to split: Interval: {interval}, Algo type: {algo_type}, Slot no: {slot_no}")
                            return
                    else:
                        result = self.db["slots"].find_one_and_update(
                            {
                                "interval": interval,
                                "algo_type": algo_type,
                                "slot_no": slot_no
                            },
                            {
                                "$set": {
                                    "slot_size": new_slot_size,
                                    "occupied": False,
                                    "latest_usage_human_readable": human_readable_time,
                                    "update_reason": "releasing_slot",
                                    "latest_operated_coin": latest_operated_coin,
                                    "previous_value": previous_value,
                                    "latest_trading_opportunity_id": latest_trading_opportunity_id
                                }
                            },
                            return_document=pymongo.ReturnDocument.AFTER
                        )
                        if not result:
                            logging.warning(
                                f"No slot found to release: Interval: {interval}, Algo type: {algo_type}, Slot no: {slot_no}")
                else:
                    logging.warning(
                        f"Slot not found: Interval: {interval}, Algo type: {algo_type}, Slot no: {slot_no}")
        except Exception as e:
            logging.error(f"Error in release_slot: {e}")

    def create_new_slot(self, interval, algo_type, slot_size, latest_operated_coin, latest_trading_opportunity_id):
        try:
            new_slot_no = self.db["slots"].count_documents(
                {"interval": interval, "algo_type": algo_type}) + 1
            new_slot = {
                "interval": interval,
                "algo_type": algo_type,
                "slot_no": new_slot_no,
                "slot_size": slot_size,
                "occupied": False,
                "latest_operated_coin": latest_operated_coin,
                "previous_value": None,
                "latest_trading_opportunity_id": latest_trading_opportunity_id,
                "latest_usage_human_readable": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "update_reason": "created_new_slot"
            }
            self.db["slots"].insert_one(new_slot)
            return new_slot_no
        except Exception as e:
            logging.error(f"Error in create_new_slot: {e}")
            return None


class IndicatorCalculator:
    @staticmethod
    def calculate_indicators(data, coin_name):
        try:
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume', 'ignore'
            ])
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

            current_time_human_readable = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            df["data_entry_time"] = current_time_human_readable
            df['human_readable_date'] = pd.to_datetime(
                df['close_time'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')

            df = IndicatorCalculator.add_technical_indicators(df)
            df['alma'] = IndicatorCalculator.calculate_alma(df['close'])
            df["alma50"] = IndicatorCalculator.calculate_alma(
                df['close'], window=50)
            df = pd.concat(
                [df, IndicatorCalculator.calculate_heikin_ashi(df)], axis=1)
            df['chaikin_volatility'] = IndicatorCalculator.calculate_chaikin_volatility(
                df)
            df['chaikin_oscillator'] = IndicatorCalculator.calculate_chaikin_oscillator(
                df)
            df['donchian_channel_hband'] = IndicatorCalculator.calculate_donchian_channel(
                df['high'], window=20, channel='hband')
            df['donchian_channel_lband'] = IndicatorCalculator.calculate_donchian_channel(
                df['low'], window=20, channel='lband')
            df['donchian_channel_mband'] = (
                df['donchian_channel_hband'] + df['donchian_channel_lband']) / 2
            df['donchian_channel_hband96'] = IndicatorCalculator.calculate_donchian_channel(
                df['high'], window=96, channel='hband')
            df['donchian_channel_lband96'] = IndicatorCalculator.calculate_donchian_channel(
                df['low'], window=96, channel='lband')
            df['donchian_channel_mband96'] = (
                df['donchian_channel_hband96'] + df['donchian_channel_lband96']) / 2

            atr_indicator = AverageTrueRange(
                df['high'], df['low'], df['close'])
            df['atr'] = atr_indicator.average_true_range()

            return df
        except Exception as e:
            logging.error(f"Error in calculate_indicators: {e}")
            return None

    @staticmethod
    def add_technical_indicators(df):
        try:
            df = df.astype({'close': float, 'high': float,
                           'low': float, 'open': float, 'volume': float})

            macd = MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()

            macd_100 = MACD(df['close'], window_fast=50,
                            window_slow=100, window_sign=9)
            df['macd_100'] = macd_100.macd()
            df['macd_100_signal'] = macd_100.macd_signal()
            df['macd_100_diff'] = macd_100.macd_diff()

            bollinger = BollingerBands(df['close'])
            df['bollinger_mavg'] = bollinger.bollinger_mavg()
            df['bollinger_hband'] = bollinger.bollinger_hband()
            df['bollinger_lband'] = bollinger.bollinger_lband()

            df['ma9'] = df['close'].rolling(window=9).mean()
            df['ma200'] = df['close'].rolling(window=200).mean()

            df['rsi'] = RSIIndicator(df['close']).rsi()

            for window in [14, 100, 200]:
                stoch_rsi = StochRSIIndicator(
                    df['close'], window=window, smooth1=3, smooth2=3)
                df[f'stochrsi_{window}_k'] = stoch_rsi.stochrsi_k() * 100
                df[f'stochrsi_{window}_d'] = stoch_rsi.stochrsi_d() * 100

            awesome_oscillator = AwesomeOscillatorIndicator(
                df['high'], df['low'])
            df['awesome_oscillator'] = awesome_oscillator.awesome_oscillator()
            df['previous_ao'] = df['awesome_oscillator'].shift(1)
            df['ao_color'] = np.where(
                df['awesome_oscillator'] > df['previous_ao'], 'green', 'red')

            tsi_indicator = TSIIndicator(df['close'])
            df['tsi'] = tsi_indicator.tsi() / 100
            df['tsi_signal'] = df['tsi'].ewm(span=13, adjust=False).mean()
            df["tsi_diff"] = df["tsi"] - df["tsi_signal"]

            df['ema50'] = EMAIndicator(df['close'], window=50).ema_indicator()
            df['ema100'] = EMAIndicator(
                df['close'], window=100).ema_indicator()

            ema9 = EMAIndicator(df['close'], window=9).ema_indicator()
            ema_ema9 = EMAIndicator(ema9, window=9).ema_indicator()
            ema_ema_ema9 = EMAIndicator(ema_ema9, window=9).ema_indicator()
            df['tema'] = (3 * ema9) - (3 * ema_ema9) + ema_ema_ema9

            ema50 = EMAIndicator(df['close'], window=50).ema_indicator()
            ema_ema50 = EMAIndicator(ema50, window=50).ema_indicator()
            ema_ema_ema50 = EMAIndicator(ema_ema50, window=50).ema_indicator()
            df['tema50'] = (3 * ema50) - (3 * ema_ema50) + ema_ema_ema50

            ema100 = EMAIndicator(df['close'], window=100).ema_indicator()
            ema_ema100 = EMAIndicator(ema100, window=100).ema_indicator()
            ema_ema_ema100 = EMAIndicator(
                ema_ema100, window=100).ema_indicator()
            df['tema100'] = (3 * ema100) - (3 * ema_ema100) + ema_ema_ema100

            return df
        except Exception as e:
            logging.error(f"Error in add_technical_indicators: {e}")

    @staticmethod
    def calculate_heikin_ashi(df):
        try:
            ha_df = pd.DataFrame(index=df.index)
            ha_df['ha_close'] = (
                df[['open', 'high', 'low', 'close']].sum(axis=1)) / 4
            ha_df['ha_open'] = (
                (df['open'].shift(1) + df['close'].shift(1)) / 2).fillna(df['open'])
            ha_df['ha_high'] = pd.concat(
                [df['high'], ha_df[['ha_open', 'ha_close']]], axis=1).max(axis=1)
            ha_df['ha_low'] = pd.concat(
                [df['low'], ha_df[['ha_open', 'ha_close']]], axis=1).min(axis=1)
            ha_df['ha_color'] = np.where(
                ha_df['ha_open'] < ha_df['ha_close'], 'green', 'red')
            return ha_df
        except Exception as e:
            logging.error(f"Error in calculate_heikin_ashi: {e}")

    @staticmethod
    def calculate_chaikin_volatility(df, high_low_period=10, roc_period=10):
        try:
            high_low_range = df['high'] - df['low']
            ema_high_low = high_low_range.ewm(
                span=high_low_period, adjust=False).mean()
            chaikin_volatility = (ema_high_low.diff(
                periods=roc_period) / ema_high_low.shift(periods=roc_period)) * 100
            return chaikin_volatility
        except Exception as e:
            logging.error(f"Error in calculate_chaikin_volatility: {e}")

    @staticmethod
    def calculate_alma(series, window=9, sigma=6, offset=0.85):
        try:
            m = np.floor(offset * (window - 1))
            s = window / sigma
            weights = np.exp(-((np.arange(window) - m) ** 2) / (2 * s * s))
            weights /= weights.sum()
            alma = np.convolve(series, weights, mode='valid')
            alma_full = np.concatenate([np.zeros(window - 1), alma])
            return pd.Series(alma_full, index=series.index)
        except Exception as e:
            logging.error(f"Error in calculate_alma: {e}")

    @staticmethod
    def calculate_chaikin_oscillator(df):
        try:
            numeric_cols = ['close', 'high', 'low', 'open', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(
                pd.to_numeric, errors='coerce')

            N = ((df['close'] - df['low']) - (df['high'] -
                 df['close'])) / (df['high'] - df['low'])
            N.fillna(0, inplace=True)

            M = N * df['volume']
            ADL = M.cumsum()
            df['ad'] = ADL

            df['chaikin_oscillator'] = df['ad'].ewm(
                span=3, adjust=False).mean() - df['ad'].ewm(span=10, adjust=False).mean()
            return df['chaikin_oscillator']
        except Exception as e:
            logging.error(f"Error in calculate_chaikin_oscillator: {e}")

    @staticmethod
    def calculate_donchian_channel(series, window, channel):
        try:
            if channel == 'hband':
                return series.rolling(window=window).max()
            elif channel == 'lband':
                return series.rolling(window=window).min()
            else:
                raise ValueError("Channel must be either 'hband' or 'lband'")
        except Exception as e:
            logging.error(f"Error in calculate_donchian_channel: {e}")


class TrendAnalyzer:
    def __init__(self, db_manager, interval_list):
        self.db_manager = db_manager
        self.leverage = 10
        self.total_init_capital = 5000  # Total budget available
        self.max_positions = 50  # Maximum number of concurrent positions
        self.api_key = os.getenv("API_KEY")
        self.api_secret = os.getenv("API_SECRET")
        self.margins = {}
        self.algo_types = ["conditions1", "conditions2", "conditions3", "conditions4",
                           "conditions5", "conditions6", "conditions7", "conditions8",
                           "conditions9", "conditions10", "conditions11", "conditions12",
                           "conditions13", "conditions14", "conditions15", "conditions16"]
        self.db_manager.initialize_slots(
            interval_list, self.algo_types, self.max_positions, self.total_init_capital)

    def update_margins(self):
        """Fetch and update the maintenance margin ratio for all filtered coin pairs."""
        try:
            for coin_pair in self.db_manager.coin_pairs:  # Assuming access to filtered_coin_pairs list
                margin_ratio = self.fetch_maintenance_margin_ratio(coin_pair)
                if margin_ratio is not None:
                    self.margins[coin_pair] = margin_ratio
                else:
                    logging.warning(f"Could not update margin for {coin_pair}")
        except Exception as e:
            logging.error(f"Error in update_margins: {e}")

    def can_open_new_position(self, algo_type, interval, coin_pair):
        try:
            if self.db_manager.db["trading_opportunities"].find_one({"coin_pair": coin_pair, "interval": interval, "status": "OPEN", "algo_type": algo_type}):
                logging.info(
                    f"already open position in this algo type and slot {coin_pair} {interval} {algo_type}")
            else:
                latest_operated_coin = coin_pair
                latest_trading_opportunity_id = None  
                slot = self.db_manager.fetch_and_occupy_slot(
                    interval, algo_type, latest_operated_coin, latest_trading_opportunity_id)
                if slot:
                    slot_no, slot_size, previous_value = slot
                    return slot_no, slot_size * self.leverage
                else:
                    logging.info(
                        f"No available slot for {interval} and {algo_type}")
                    return None

        except Exception as e:
            logging.error(f"Error in can_open_new_position: {e}")
            return None

    def check_for_short(self, coin_pair, interval, current_row, prev_row, two_prev_row):
        try:

            numeric_keys = ['open', 'close',
                            'macd', 'tsi', 'stochrsi_14_k']
            for key in numeric_keys:
                if key in prev_row and key in current_row:
                    prev_row[key] = float(prev_row[key])
                    current_row[key] = float(current_row[key])
            trend_negative = current_row["bollinger_mavg"] < prev_row["bollinger_mavg"] < two_prev_row["bollinger_mavg"]

            conditions1 = {
                'price is lower than donchian middle band': current_row["close"] < current_row["donchian_channel_mband"],
                'rsi is lower higher than 55': current_row["rsi"] > 45,
                'price is lower than prev': current_row["close"] < prev_row["close"],
                'rsi is lower than prev': current_row["rsi"] < prev_row["rsi"],
                'macd is lower than prev': current_row["macd"] < prev_row["macd"]

            }
            conditions2 = {
                'price is lower than donchian middle band': current_row["close"] < current_row["donchian_channel_mband"],
                'rsi is lower higher than 55': current_row["rsi"] > 45,
                'price is lower than prev': current_row["close"] < prev_row["close"],
                'rsi is lower than prev': current_row["rsi"] < prev_row["rsi"],
                'macd is lower than prev': current_row["macd"] < prev_row["macd"],
                'macd negative': current_row["macd_diff"] < 0

            }
            conditions3 = {
                'price is lower than donchian middle band': current_row["close"] < current_row["donchian_channel_mband"],
                'rsi is lower higher than 55': current_row["rsi"] > 45,
                'price is lower than prev': current_row["close"] < prev_row["close"],
                'rsi is lower than prev': current_row["rsi"] < prev_row["rsi"],
                'macd is lower than prev': current_row["macd"] < prev_row["macd"],
                'trend negative': trend_negative

            }
            conditions4 = {
                'price is lower than donchian middle band': current_row["close"] < current_row["donchian_channel_mband"],
                'rsi is lower higher than 55': current_row["rsi"] > 45,
                'price is lower than prev': current_row["close"] < prev_row["close"],
                'rsi is lower than prev': current_row["rsi"] < prev_row["rsi"],
                'macd is lower than prev': current_row["macd"] < prev_row["macd"]

            }

            conditions5 = {
                'price is lower than donchian middle band': current_row["close"] < current_row["bollinger_mavg"],
                'price is lower than prev': current_row["close"] < prev_row["close"],
                'rsi is lower than prev': current_row["rsi"] < prev_row["rsi"],
                'macd is lower than prev': current_row["macd"] < prev_row["macd"],
                'macd negative': current_row["macd_diff"] < 0,
                'stoch_rsi200 highert than 60 lower than 80': (current_row["stochrsi_200_k"] > 60 and current_row["stochrsi_200_k"] < 80 and prev_row["stochrsi_200_k"] > 65)
                or (current_row["stochrsi_100_k"] > 60 and current_row["stochrsi_100_k"] < 80 and prev_row["stochrsi_100_k"] > 65),
            }

            conditions6 = {
                'price is lower than donchian middle band': current_row["close"] < current_row["donchian_channel_mband"],
                'rsi is lower higher than 55': current_row["rsi"] > 45,
                'price is lower than prev': current_row["close"] < prev_row["close"],
                'rsi is lower than prev': current_row["rsi"] < prev_row["rsi"],
                'macd is lower than prev': current_row["macd"] < prev_row["macd"],
                'macd negative': current_row["macd_diff"] < 0,
                'stoch_rsi200 highert than 60 lower than 80': (current_row["stochrsi_200_k"] > 60 and current_row["stochrsi_200_k"] < 80 and prev_row["stochrsi_200_k"] > 65)
                or (current_row["stochrsi_100_k"] > 60 and current_row["stochrsi_100_k"] < 80 and prev_row["stochrsi_100_k"] > 65),

            }

            conditions7 = {
                'price is lower than donchian middle band': current_row["close"] < current_row["bollinger_mavg"],
                'rsi is lower higher than 55': current_row["rsi"] > 45,
                'price is lower than prev': current_row["close"] < prev_row["close"],
                'rsi is lower than prev': current_row["rsi"] < prev_row["rsi"],
                'macd is lower than prev': current_row["macd"] < prev_row["macd"],
                'macd negative': current_row["macd_diff"] < 0

            }
            conditions8 = {
                'price is lower than donchian middle band': current_row["close"] < current_row["bollinger_mavg"] and prev_row["close"] > prev_row["bollinger_mavg"],
                'rsi is lower higher than 55': current_row["rsi"] > 45,
                'price is lower than prev': current_row["close"] < prev_row["close"],
                'rsi is lower than prev': current_row["rsi"] < prev_row["rsi"],
                'macd is lower than prev': current_row["macd"] < prev_row["macd"],
                'macd negative': current_row["macd_diff"] < 0

            }
            conditions9 = {
                'price is lower than donchian middle band': current_row["close"] < current_row["bollinger_mavg"] and prev_row["close"] > prev_row["bollinger_mavg"],
                'rsi is lower higher than 55': current_row["rsi"] > 45,
                'price is lower than prev': current_row["close"] < prev_row["close"],
                'rsi is lower than prev': current_row["rsi"] < prev_row["rsi"],
                'macd is lower than prev': current_row["macd"] < prev_row["macd"],
                'macd negative': current_row["macd_diff"] < 0

            }
            onditions10 = {
                'price is lower than donchian middle band': current_row["close"] < current_row["tema100"],
                'rsi is lower higher than 55': current_row["rsi"] > 45,
                'price is lower than prev': current_row["close"] < prev_row["close"],
                'rsi is lower than prev': current_row["rsi"] < prev_row["rsi"],
                'macd is lower than prev': current_row["macd"] < prev_row["macd"],
                'macd negative': current_row["macd_diff"] < 0

            }
            conditions11 = {
                'price is lower than donchian middle band': current_row["close"] < current_row["bollinger_mavg"],
                'rsi is lower higher than 55': current_row["rsi"] > 45,
                'price is lower than prev': current_row["close"] < prev_row["close"],
                'rsi is lower than prev': current_row["rsi"] < prev_row["rsi"],
                'macd is lower than prev': current_row["macd"] < prev_row["macd"],
                'macd negative': current_row["macd_diff"] < 0

            }
            conditions12 = {
                'price is lower than bollingermavg': current_row["close"] < current_row["bollinger_mavg"],
                'price is lower than prev': current_row["close"] < prev_row["close"],
                'rsi is lower than prev': current_row["rsi"] < prev_row["rsi"],
                'macd is lower than prev': current_row["macd"] < prev_row["macd"],
                'two heikin aski is red': current_row["ha_color"] == 'red' and prev_row["ha_color"] == 'red',
                'ao color is red': current_row["ao_color"] == 'red' and prev_row["ao_color"] == 'red',
                'stoch rsi200 is lower than prev ': current_row["stochrsi_200_k"] < prev_row["stochrsi_200_k"],
                'stoch rsi100 is lower than prev ': current_row["stochrsi_100_k"] < prev_row["stochrsi_100_k"],
                'stoch rsi14 is lower than prev ': current_row["stochrsi_14_k"] < prev_row["stochrsi_14_k"],
                'tsi is lower than prev': current_row["tsi"] < prev_row["tsi"],
            }
            conditions13 = {
                'price is lower than bollingermavg': current_row["close"] < current_row["bollinger_mavg"],
                'price is lower than prev': current_row["close"] < prev_row["close"],
                'rsi is lower than prev': current_row["rsi"] < prev_row["rsi"],
                'macd is lower than prev': current_row["macd"] < prev_row["macd"],
                'two heikin aski is red': current_row["ha_color"] == 'red' and prev_row["ha_color"] == 'red',
                'ao color is red': current_row["ao_color"] == 'red' and prev_row["ao_color"] == 'red',
                'stoch rsi200 is lower than prev ': current_row["stochrsi_200_k"] < prev_row["stochrsi_200_k"],
                'stoch rsi100 is lower than prev ': current_row["stochrsi_100_k"] < prev_row["stochrsi_100_k"],
                'stoch rsi14 is lower than prev ': current_row["stochrsi_14_k"] < prev_row["stochrsi_14_k"],
                'tsi is lower than prev': current_row["tsi"] < prev_row["tsi"],
            }
            conditions14 = {
                'price is lower than bollingermavg': current_row["close"] < current_row["bollinger_mavg"],
                'price is lower than prev': current_row["close"] < prev_row["close"],
                'rsi is lower than prev': current_row["rsi"] < prev_row["rsi"],
                'macd is lower than prev': current_row["macd"] < prev_row["macd"],
                'two heikin aski is red': current_row["ha_color"] == 'red' and prev_row["ha_color"] == 'red',
                'ao color is red': current_row["ao_color"] == 'red' and prev_row["ao_color"] == 'red',
                'stoch rsi200 is lower than prev ': current_row["stochrsi_200_k"] < prev_row["stochrsi_200_k"],
                'stoch rsi100 is lower than prev ': current_row["stochrsi_100_k"] < prev_row["stochrsi_100_k"],
                'stoch rsi14 is lower than prev ': current_row["stochrsi_14_k"] < prev_row["stochrsi_14_k"],
                'tsi is lower than prev': current_row["tsi"] < prev_row["tsi"],
            }
            conditions15 = {
                'price s lower than bollingermavg': current_row["close"] < current_row["bollinger_mavg"],
                'price is lower than prev': current_row["close"] < prev_row["close"],
                'rsi is lower than prev': current_row["rsi"] < prev_row["rsi"],
                'macd is lower than prev': current_row["macd"] < prev_row["macd"],
                'two heikin aski is red': current_row["ha_color"] == 'red' and prev_row["ha_color"] == 'red',
                'ao color is red': current_row["ao_color"] == 'red' and prev_row["ao_color"] == 'red',
                'stoch rsi200 is lower than prev ': current_row["stochrsi_200_k"] < prev_row["stochrsi_200_k"],
                'stoch rsi100 is lower than prev ': current_row["stochrsi_100_k"] < prev_row["stochrsi_100_k"],
                'stoch rsi14 is lower than prev ': current_row["stochrsi_14_k"] < prev_row["stochrsi_14_k"],
                'tsi is lower than prev': current_row["tsi"] < prev_row["tsi"],
            }
            conditions16 = {
                'price s lower than bollingermavg': current_row["close"] < current_row["bollinger_mavg"],
                'price is lower than prev': current_row["close"] < prev_row["close"],
                'rsi is lower than prev': current_row["rsi"] < prev_row["rsi"],
                'macd is lower than prev': current_row["macd"] < prev_row["macd"],
                'two heikin aski is red': current_row["ha_color"] == 'red' and prev_row["ha_color"] == 'red',
                'ao color is red': current_row["ao_color"] == 'red' and prev_row["ao_color"] == 'red',
                'stoch rsi200 is lower than prev ': current_row["stochrsi_200_k"] < prev_row["stochrsi_200_k"],
                'stoch rsi100 is lower than prev ': current_row["stochrsi_100_k"] < prev_row["stochrsi_100_k"],
                'stoch rsi14 is lower than prev ': current_row["stochrsi_14_k"] < prev_row["stochrsi_14_k"],
                'tsi is lower than prev': current_row["tsi"] < prev_row["tsi"],
            }

            conditions_list = [conditions1, conditions2, conditions3, conditions4,
                               conditions5, conditions6, conditions7, conditions8,
                               conditions9, onditions10, conditions11, conditions12,
                               conditions13, conditions14, conditions15, conditions16]

            for i, conditions in enumerate(conditions_list, start=1):

                if all(conditions.values()):
                    entry_price = float(current_row['close'])
                    atr = current_row["atr"]
                    atr_adjustment = atr * 3

                    stop_loss = entry_price + atr_adjustment
                    take_profit = entry_price - atr_adjustment

                    maint_margin_ratio = self.margins.get(coin_pair)
                    liquidation_price = self.calculate_liquidation_price(
                        entry_price, self.leverage, maint_margin_ratio, 'SHORT')
                    if liquidation_price is None:
                        logging.warning(f"Liquidity price is None for {coin_pair}, skipping opportunity.")
                        continue
                    timestamp = int(current_row['close_time'])
                    coin_link = f"https://www.binance.com/en/futures/{coin_pair}"

                    slot = self.can_open_new_position(
                        f"conditions{i}", interval, coin_pair)

                    if slot:
                        slot_no, slot_size = slot
                        logging.info("SHORT | Coin: %s, %s | human readable date: %s | Conditions %d ",
                                     coin_pair, interval, current_row['human_readable_date'], i)
                        opportunity_data = {
                            'coin_pair': coin_pair,
                            'opportunity_type': "SHORT",
                            'entry_price': entry_price,
                            'open_timestamp': timestamp,
                            'open_human_readable_date': current_row['human_readable_date'],
                            'interval': interval,
                            'algo_type': f"conditions{i}",
                            'status': "OPEN",
                            'liq_price': liquidation_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'pos_size': slot_size,
                            'slot_no': slot_no
                        }

                        self.db_manager.record_opportunity(opportunity_data)

                    elif not slot:
                        logging.info(
                            "available slot not found short: %s %s", f"conditions{i}", interval)
                        return False

        except Exception as e:
            logging.error(f"Error in check_for_short: {e}")

    def check_for_long(self, coin_pair, interval, current_row, prev_row, two_prev_row):
        try:
            numeric_keys = ['open', 'close',
                            'macd', 'tsi', 'stochrsi_14_k']
            for key in numeric_keys:
                if key in prev_row and key in current_row:
                    prev_row[key] = float(prev_row[key])
                    current_row[key] = float(current_row[key])
            trend_positive = current_row["bollinger_mavg"] > prev_row["bollinger_mavg"] > two_prev_row["bollinger_mavg"]
            conditions1 = {
                'price is higher than donchian middle band': current_row["close"] > current_row["donchian_channel_mband"],
                'price is higher than prev': current_row["close"] > prev_row["close"] > two_prev_row["close"],
                'rsi is lower than 55': current_row["rsi"] < 55,
                'rsi is higher than prev': current_row["rsi"] > prev_row["rsi"],
                'macd is higher than prev': current_row["macd"] > prev_row["macd"],
            }
            conditions2 = {
                'price is higher than donchian middle band': current_row["close"] > current_row["donchian_channel_mband"],
                'price is higher than prev': current_row["close"] > prev_row["close"] > two_prev_row["close"],
                'rsi is lower than 55': current_row["rsi"] < 55,
                'rsi is higher than prev': current_row["rsi"] > prev_row["rsi"],
                'macd is higher than prev': current_row["macd"] > prev_row["macd"],
                'macd positive': current_row["macd_diff"] > 0
            }
            conditions3 = {
                'price is higher than donchian middle band': current_row["close"] > current_row["donchian_channel_mband"],
                'price is higher than prev': current_row["close"] > prev_row["close"] > two_prev_row["close"],
                'rsi is lower than 55': current_row["rsi"] < 55,
                'rsi is higher than prev': current_row["rsi"] > prev_row["rsi"],
                'macd is higher than prev': current_row["macd"] > prev_row["macd"],
                'trend positive': trend_positive
            }
            conditions4 = {
                'price is higher than donchian middle band': current_row["close"] > current_row["donchian_channel_mband"],
                'price is higher than prev': current_row["close"] > prev_row["close"] > two_prev_row["close"],
                'rsi is lower than 55': current_row["rsi"] < 55,
                'rsi is higher than prev': current_row["rsi"] > prev_row["rsi"],
                'macd is higher than prev': current_row["macd"] > prev_row["macd"],

            }

            conditions5 = {
                'price is higher than donchian middle band': current_row["close"] > current_row["bollinger_mavg"],
                'price is higher than prev': current_row["close"] > prev_row["close"] > two_prev_row["close"],
                'rsi is higher than prev': current_row["rsi"] > prev_row["rsi"],
                'macd is higher than prev': current_row["macd"] > prev_row["macd"],
                'macd positive': current_row["macd_diff"] > 0,
                'stoch_rsi200 highert than 60 lower than 80': (current_row["stochrsi_200_k"] > 30 and current_row["stochrsi_200_k"] < 50 and prev_row["stochrsi_200_k"] < 30)
                or (current_row["stochrsi_100_k"] > 30 and current_row["stochrsi_100_k"] < 50 and prev_row["stochrsi_100_k"] < 30),
            }

            conditions6 = {
                'price is higher than donchian middle band': current_row["close"] > current_row["donchian_channel_mband"],
                'price is higher than prev': current_row["close"] > prev_row["close"] > two_prev_row["close"],
                'rsi is lower than 55': current_row["rsi"] < 55,
                'rsi is higher than prev': current_row["rsi"] > prev_row["rsi"],
                'macd is higher than prev': current_row["macd"] > prev_row["macd"],
                'macd positive': current_row["macd_diff"] > 0,
                'stoch_rsi200 highert than 60 lower than 80': (current_row["stochrsi_200_k"] > 30 and current_row["stochrsi_200_k"] < 50 and prev_row["stochrsi_200_k"] < 30)
                or (current_row["stochrsi_100_k"] > 30 and current_row["stochrsi_100_k"] < 50 and prev_row["stochrsi_100_k"] < 30),

            }

            conditions7 = {
                'price is higher than donchian middle band': current_row["close"] > current_row["bollinger_mavg"],
                'price is higher than prev': current_row["close"] > prev_row["close"] > two_prev_row["close"],
                'rsi is lower than 55': current_row["rsi"] < 55,
                'rsi is higher than prev': current_row["rsi"] > prev_row["rsi"],
                'macd is higher than prev': current_row["macd"] > prev_row["macd"],
                'macd positive': current_row["macd_diff"] > 0
            }
            conditions8 = {
                'price is higher than donchian middle band': current_row["close"] > current_row["bollinger_mavg"] and prev_row["close"] < prev_row["bollinger_mavg"],
                'price is higher than prev': current_row["close"] > prev_row["close"] > two_prev_row["close"],
                'rsi is lower than 55': current_row["rsi"] < 55,
                'rsi is higher than prev': current_row["rsi"] > prev_row["rsi"],
                'macd is higher than prev': current_row["macd"] > prev_row["macd"],
                'macd positive': current_row["macd_diff"] > 0
            }
            conditions9 = {
                'price is higher than donchian middle band': current_row["close"] > current_row["bollinger_mavg"] and prev_row["close"] < prev_row["bollinger_mavg"],
                'price is higher than prev': current_row["close"] > prev_row["close"] > two_prev_row["close"],
                'rsi is lower than 55': current_row["rsi"] < 55,
                'rsi is higher than prev': current_row["rsi"] > prev_row["rsi"],
                'macd is higher than prev': current_row["macd"] > prev_row["macd"],
                'macd positive': current_row["macd_diff"] > 0
            }
            conditions10 = {
                'price is higher than donchian middle band': current_row["close"] > current_row["bollinger_mavg"],
                'price is higher than prev': current_row["close"] > prev_row["close"] > two_prev_row["close"],
                'rsi is lower than 55': current_row["rsi"] < 55,
                'rsi is higher than prev': current_row["rsi"] > prev_row["rsi"],
                'macd is higher than prev': current_row["macd"] > prev_row["macd"],
                'macd positive': current_row["macd_diff"] > 0
            }
            conditions11 = {
                'price is higher than donchian middle band': current_row["close"] > current_row["tema100"],
                'price is higher than prev': current_row["close"] > prev_row["close"] > two_prev_row["close"],
                'rsi is lower than 55': current_row["rsi"] < 55,
                'rsi is higher than prev': current_row["rsi"] > prev_row["rsi"],
                'macd is higher than prev': current_row["macd"] > prev_row["macd"],
                'macd positive': current_row["macd_diff"] > 0
            }
            conditions12 = {
                'price is lower than bollingermavg': current_row["close"] > current_row["bollinger_mavg"],
                'price is lower than prev': current_row["close"] > prev_row["close"],
                'rsi is lower than prev': current_row["rsi"] > prev_row["rsi"],
                'macd is lower than prev': current_row["macd"] > prev_row["macd"],
                'two heikin aski is red': current_row["ha_color"] == 'green' and prev_row["ha_color"] == 'green',
                'ao color is red': current_row["ao_color"] == 'green' and prev_row["ao_color"] == 'green',
                'stoch rsi200 is lower than prev ': current_row["stochrsi_200_k"] > prev_row["stochrsi_200_k"],
                'stoch rsi100 is lower than prev ': current_row["stochrsi_100_k"] > prev_row["stochrsi_100_k"],
                'stoch rsi14 is lower than prev ': current_row["stochrsi_14_k"] > prev_row["stochrsi_14_k"],
                'tsi is lower than prev': current_row["tsi"] > prev_row["tsi"],
            }
            conditions13 = {
                'price is lower than bollingermavg': current_row["close"] > current_row["bollinger_mavg"],
                'price is lower than prev': current_row["close"] > prev_row["close"],
                'rsi is lower than prev': current_row["rsi"] > prev_row["rsi"],
                'macd is lower than prev': current_row["macd"] > prev_row["macd"],
                'two heikin aski is red': current_row["ha_color"] == 'green' and prev_row["ha_color"] == 'green',
                'ao color is red': current_row["ao_color"] == 'green' and prev_row["ao_color"] == 'green',
                'stoch rsi200 is lower than prev ': current_row["stochrsi_200_k"] > prev_row["stochrsi_200_k"],
                'stoch rsi100 is lower than prev ': current_row["stochrsi_100_k"] > prev_row["stochrsi_100_k"],
                'stoch rsi14 is lower than prev ': current_row["stochrsi_14_k"] > prev_row["stochrsi_14_k"],
                'tsi is lower than prev': current_row["tsi"] > prev_row["tsi"],
            }
            conditions14 = {
                'price is lower than bollingermavg': current_row["close"] > current_row["bollinger_mavg"],
                'price is lower than prev': current_row["close"] > prev_row["close"],
                'rsi is lower than prev': current_row["rsi"] > prev_row["rsi"],
                'macd is lower than prev': current_row["macd"] > prev_row["macd"],
                'two heikin aski is red': current_row["ha_color"] == 'green' and prev_row["ha_color"] == 'green',
                'ao color is red': current_row["ao_color"] == 'green' and prev_row["ao_color"] == 'green',
                'stoch rsi200 is lower than prev ': current_row["stochrsi_200_k"] > prev_row["stochrsi_200_k"],
                'stoch rsi100 is lower than prev ': current_row["stochrsi_100_k"] > prev_row["stochrsi_100_k"],
                'stoch rsi14 is lower than prev ': current_row["stochrsi_14_k"] > prev_row["stochrsi_14_k"],
                'tsi is lower than prev': current_row["tsi"] > prev_row["tsi"],
            }

            conditions15 = {
                'price is lower than bollingermavg': current_row["close"] > current_row["bollinger_mavg"],
                'price is lower than prev': current_row["close"] > prev_row["close"],
                'rsi is lower than prev': current_row["rsi"] > prev_row["rsi"],
                'macd is lower than prev': current_row["macd"] > prev_row["macd"],
                'two heikin aski is red': current_row["ha_color"] == 'green' and prev_row["ha_color"] == 'green',
                'ao color is red': current_row["ao_color"] == 'green' and prev_row["ao_color"] == 'green',
                'stoch rsi200 is lower than prev ': current_row["stochrsi_200_k"] > prev_row["stochrsi_200_k"],
                'stoch rsi100 is lower than prev ': current_row["stochrsi_100_k"] > prev_row["stochrsi_100_k"],
                'stoch rsi14 is lower than prev ': current_row["stochrsi_14_k"] > prev_row["stochrsi_14_k"],
                'tsi is lower than prev': current_row["tsi"] > prev_row["tsi"],
            }
            conditions16 = {
                'price is lower than bollingermavg': current_row["close"] > current_row["bollinger_mavg"],
                'price is lower than prev': current_row["close"] > prev_row["close"],
                'rsi is lower than prev': current_row["rsi"] > prev_row["rsi"],
                'macd is lower than prev': current_row["macd"] > prev_row["macd"],
                'two heikin aski is red': current_row["ha_color"] == 'green' and prev_row["ha_color"] == 'green',
                'ao color is red': current_row["ao_color"] == 'green' and prev_row["ao_color"] == 'green',
                'stoch rsi200 is lower than prev ': current_row["stochrsi_200_k"] > prev_row["stochrsi_200_k"],
                'stoch rsi100 is lower than prev ': current_row["stochrsi_100_k"] > prev_row["stochrsi_100_k"],
                'stoch rsi14 is lower than prev ': current_row["stochrsi_14_k"] > prev_row["stochrsi_14_k"],
                'tsi is lower than prev': current_row["tsi"] > prev_row["tsi"],
            }

            conditions_list = [conditions1, conditions2, conditions3, conditions4,
                               conditions5, conditions6, conditions7, conditions8,
                               conditions9, conditions10, conditions11, conditions12,
                               conditions13, conditions14, conditions15, conditions16]

            for i, conditions in enumerate(conditions_list, start=1):

                if all(conditions.values()):
                    entry_price = float(current_row['close'])
                    atr = current_row["atr"]
                    atr_adjustment = atr * 3

                    stop_loss = entry_price - atr_adjustment
                    take_profit = entry_price + atr_adjustment

                    maint_margin_ratio = self.margins.get(coin_pair)
                    liquidation_price = self.calculate_liquidation_price(
                        entry_price, self.leverage, maint_margin_ratio, 'LONG')

                    if liquidation_price is None:
                        logging.warning(f"Liquidity price is None for {coin_pair}, skipping opportunity.")
                        continue
                    timestamp = int(current_row['close_time'])
                    coin_link = f"https://www.binance.com/en/futures/{coin_pair}"

                    slot = self.can_open_new_position(
                        f"conditions{i}", interval, coin_pair)
                    if slot:
                        slot_no, slot_size = slot

                        logging.info("LONG | Coin: %s, %s | human readable date: %s | Conditions %d met",
                                     coin_pair, interval, current_row['human_readable_date'], i)
                        opportunity_data = {
                            'coin_pair': coin_pair,
                            'opportunity_type': "LONG",
                            'entry_price': entry_price,
                            'open_timestamp': timestamp,
                            'open_human_readable_date': current_row['human_readable_date'],
                            'interval': interval,
                            'algo_type': f"conditions{i}",
                            'status': "OPEN",
                            'liq_price': liquidation_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'pos_size': slot_size,
                            'slot_no': slot_no
                        }

                        self.db_manager.record_opportunity(opportunity_data)

                    elif not slot:
                        logging.info(
                            "available slot not found long: conditions%d %s", i, interval)

                        return False

        except Exception as e:
            logging.error("error in check_for_long: %s, %s, %s",
                          e, coin_pair, interval)

    def monitor_and_close_positions(self, interval, current_rows, prev_rows):
        try:
            open_positions = self.db_manager.db["trading_opportunities"].find({
                "status": "OPEN",
                "interval": interval
            })

            for position in open_positions:
                id = position.get('_id')
                coin_pair = position.get('coin_pair')
                entry_price = float(position.get('entry_price', 0))
                opportunity_type = position.get('opportunity_type')
                interval = position.get('interval')
                open_human_readable_date = position.get(
                    'open_human_readable_date')
                timestamp_from_db = position.get('timestamp_from_db')
                algo_type = position.get('algo_type')
                liq_price = position.get('liq_price')
                stop_loss = position.get('stop_loss')
                take_profit = position.get('take_profit')
                init_position_size = position.get('adjusted_pos_size')
                slot_no = position.get('slot_no')

                if init_position_size is None:
                    logging.info(
                        f"Initial position size is None for ID {id}, skipping operation.")
                    continue

                init_position_size = float(init_position_size)

                # Find the relevant current and previous rows for the specific coin pair
                current_row = next(
                    (row for row in current_rows if row['coin'] == coin_pair), None)
                prev_row = next(
                    (row for row in prev_rows if row['coin'] == coin_pair), None)

                if not current_row or not prev_row:
                    logging.warning(
                        f"Insufficient data for {coin_pair} at given timestamp in DB")
                    continue  # Skip if not enough data points are available

                current_close_price = float(current_row["close"])
                current_human_readable_date = current_row["human_readable_date"]
                close_timestamp = current_row["close_time"]
                if liq_price is None:
                    logging.warning(f"Liquidity price is None for ID {id}, skipping operation.")
                    continue
                pnl_percentage = self.calculate_pnl(
                    entry_price, current_close_price, opportunity_type)
                actual_pnl_usd_position = init_position_size * pnl_percentage
                realized_profit = init_position_size + actual_pnl_usd_position
                binance_fee = realized_profit * 0.0002
                actual_pnl_usd_position = actual_pnl_usd_position - binance_fee
                realized_profit = realized_profit - binance_fee

                current_macd_diff = current_row["macd_diff"]
                prev_macd_diff = prev_row["macd_diff"]
                # Calculate the percentage difference
                liq_percentage_difference = abs(
                    (current_close_price - liq_price) / liq_price) * 100

                short_term_trend_positive = current_row["tema"] > prev_row["tema"]
                short_term_trend_negative = current_row["tema"] < prev_row["tema"]

                status = None
                if algo_type == "conditions1" or algo_type == "conditions2" or algo_type == "conditions3" or algo_type == "conditions7" or algo_type == "conditions6" or algo_type == "conditions5" or algo_type == "conditions8" or algo_type == "conditions11" or algo_type == "conditions13":
                    if opportunity_type == "LONG" and (current_close_price < liq_price or liq_percentage_difference < 2.5):
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "LIQ", close_timestamp, realized_profit,  status="CLOSED")

                        status = "CLOSED"
                    elif opportunity_type == "SHORT" and (current_close_price > liq_price or liq_percentage_difference < 2.5):
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "LIQ", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"
                    if opportunity_type == "LONG" and current_row["rsi"] > 65:
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "kar al", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"
                    elif opportunity_type == "SHORT" and current_row["rsi"] < 40:
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "kar al", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"
                    if opportunity_type == "SHORT" and pnl_percentage > 0 and short_term_trend_positive:
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "PnL > positive and Trend Positive", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"
                    if opportunity_type == "LONG" and pnl_percentage > 0 and short_term_trend_negative:
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "PnL > positive and Trend Negative", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"
                elif algo_type == "conditions4":
                    if opportunity_type == "LONG" and (current_close_price < liq_price or liq_percentage_difference < 2.5):
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "LIQ", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"
                    elif opportunity_type == "SHORT" and (current_close_price > liq_price or liq_percentage_difference < 2.5):
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "LIQ", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"
                    if opportunity_type == "LONG" and current_row["rsi"] > 65:
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "kar al", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"
                    elif opportunity_type == "SHORT" and current_row["rsi"] < 40:
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "kar al", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"
                    if opportunity_type == "SHORT" and pnl_percentage > 0 and short_term_trend_positive:
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "PnL > positive and Trend Positive", close_timestamp,  realized_profit, status="CLOSED")

                        status = "CLOSED"
                    if opportunity_type == "LONG" and pnl_percentage > 0 and short_term_trend_negative:
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "PnL > positive and Trend Negative", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"
                    if opportunity_type == "LONG" and stop_loss > current_row["close"]:
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "stop loss", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"
                    elif opportunity_type == "SHORT" and stop_loss < current_row["close"]:
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "stop loss", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"

                elif algo_type == "conditions9":
                    if opportunity_type == "LONG" and (current_close_price < liq_price or liq_percentage_difference < 2.5):
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "LIQ", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"
                    elif opportunity_type == "SHORT" and (current_close_price > liq_price or liq_percentage_difference < 2.5):
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "LIQ", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"
                    if opportunity_type == "LONG" and current_row["high"] == current_row["donchian_channel_hband"]:
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "kar al", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"
                    elif opportunity_type == "SHORT" and current_row["low"] == current_row["donchian_channel_lband"]:
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "kar al", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"

                elif algo_type == "conditions10":
                    if opportunity_type == "LONG" and (current_close_price < liq_price or liq_percentage_difference < 2.5):
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "LIQ", close_timestamp, realized_profit,  status="CLOSED")

                        status = "CLOSED"
                    elif opportunity_type == "SHORT" and (current_close_price > liq_price or liq_percentage_difference < 2.5):
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "LIQ", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"
                    if opportunity_type == "LONG" and current_row["rsi"] > 65:
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "kar al", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"
                    elif opportunity_type == "SHORT" and current_row["rsi"] < 40:
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "kar al", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"
                    if opportunity_type == "SHORT" and pnl_percentage > 0 and short_term_trend_positive:
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "PnL > positive and Trend Positive", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"
                    if opportunity_type == "LONG" and pnl_percentage > 0 and short_term_trend_negative:
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "PnL > positive and Trend Negative", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"

                    if opportunity_type == "LONG" and pnl_percentage < 0 and (current_row["tema"] < prev_row["tema"] and current_row["rsi"] < prev_row["rsi"]):
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "PnL < 0 and Trend Negative", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"
                    if opportunity_type == "SHORT" and pnl_percentage < 0 and (current_row["tema"] > prev_row["tema"] and current_row["rsi"] > prev_row["rsi"]):
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "PnL < 0 and Trend Positive", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"

                elif algo_type == "conditions12":
                    if opportunity_type == "LONG" and (current_close_price < liq_price or liq_percentage_difference < 2.5):
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "LIQ", close_timestamp, realized_profit,  status="CLOSED")

                        status = "CLOSED"
                    elif opportunity_type == "SHORT" and (current_close_price > liq_price or liq_percentage_difference < 2.5):
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "LIQ", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"

                    if opportunity_type == "LONG" and current_row["ha_color"] == "red" and prev_row["ha_color"] == "red":
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "kar al", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"
                    elif opportunity_type == "SHORT" and current_row["ha_color"] == "green" and prev_row["ha_color"] == "green":
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "kar al", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"
                elif algo_type == "conditions14":
                    if opportunity_type == "LONG" and (current_close_price < liq_price or liq_percentage_difference < 2.5):
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "LIQ", close_timestamp, realized_profit,  status="CLOSED")

                        status = "CLOSED"
                    elif opportunity_type == "SHORT" and (current_close_price > liq_price or liq_percentage_difference < 2.5):
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "LIQ", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"

                    if opportunity_type == "LONG" and current_row["donchian_channel_hband"] == current_row["high"]:
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "kar al", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"
                    elif opportunity_type == "SHORT" and current_row["donchian_channel_lband"] == current_row["low"]:
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "kar al", close_timestamp, realized_profit, status="CLOSED")
                        status = "CLOSED"
                elif algo_type == "conditions15":
                    if opportunity_type == "LONG" and (current_close_price < liq_price or liq_percentage_difference < 2.5):
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "LIQ", close_timestamp, realized_profit,  status="CLOSED")

                        status = "CLOSED"
                    elif opportunity_type == "SHORT" and (current_close_price > liq_price or liq_percentage_difference < 2.5):
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "LIQ", close_timestamp, realized_profit, status="CLOSED")
                        status = "CLOSED"

                    if opportunity_type == "LONG" and current_row["ha_color"] == "red" and prev_row["ha_color"] == "red" and pnl_percentage > 0:
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "kar al", close_timestamp, realized_profit, status="CLOSED")
                        status = "CLOSED"
                    elif opportunity_type == "SHORT" and current_row["ha_color"] == "green" and prev_row["ha_color"] == "green" and pnl_percentage > 0:
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "kar al", close_timestamp, realized_profit, status="CLOSED")
                        status = "CLOSED"

                elif algo_type == "conditions16":
                    if opportunity_type == "LONG" and (current_close_price < liq_price or liq_percentage_difference < 2.5):
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "LIQ", close_timestamp, realized_profit,  status="CLOSED")

                        status = "CLOSED"
                    elif opportunity_type == "SHORT" and (current_close_price > liq_price or liq_percentage_difference < 2.5):
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "LIQ", close_timestamp, realized_profit, status="CLOSED")
                        status = "CLOSED"

                    if opportunity_type == "LONG" and current_row["ha_color"] == "red" and prev_row["ha_color"] == "red" and pnl_percentage > 0:
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "kar al", close_timestamp, realized_profit, status="CLOSED")
                        status = "CLOSED"
                    elif opportunity_type == "SHORT" and current_row["ha_color"] == "green" and prev_row["ha_color"] == "green" and pnl_percentage > 0:
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "kar al", close_timestamp, realized_profit, status="CLOSED")
                        status = "CLOSED"

                    if opportunity_type == "SHORT" and pnl_percentage > 0 and short_term_trend_positive:
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "PnL > positive and Trend Positive", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"
                    if opportunity_type == "LONG" and pnl_percentage > 0 and short_term_trend_negative:
                        self.db_manager.close_opportunity_and_record_pnl(
                            id, current_close_price, pnl_percentage, current_human_readable_date, actual_pnl_usd_position, "PnL > positive and Trend Negative", close_timestamp, realized_profit, status="CLOSED")

                        status = "CLOSED"

                if status == "CLOSED":
                    self.db_manager.release_slot(interval, algo_type, slot_no,
                                                 actual_pnl_usd_position, coin_pair, id)

        except Exception as e:
            logging.error("Error in monitor_and_close_positions:",
                          e, interval, coin_pair, id)

    def calculate_pnl(self, entry_price, current_price, position_type):
        try:
            if position_type == "LONG":
                return (current_price - entry_price) / entry_price
            elif position_type == "SHORT":
                return (entry_price - current_price) / entry_price
            else:
                logging.error("Unknown position type")
                return 0
        except Exception as e:
            logging.error("error in calculate_pnl", e)

    def fetch_maintenance_margin_ratio(self, symbol):
        try:
            api_key = self.api_key
            api_secret = self.api_secret
            base_url = "https://fapi.binance.com/fapi/v1/leverageBracket"
            timestamp = int(time.time() * 1000)
            query_string = f"symbol={symbol}&timestamp={timestamp}"

            # Generate the HMAC SHA256 signature
            signature = hmac.new(api_secret.encode(),
                                 query_string.encode(), hashlib.sha256).hexdigest()

            # Complete URL with the signature
            url = f"{base_url}?{query_string}&signature={signature}"

            headers = {
                'X-MBX-APIKEY': api_key
            }

            response = requests.get(url, headers=headers)
            data = response.json()
            # Check if the response contains an error
            if 'code' in data or not data:  # Added check if data is empty or not a list
                logging.error(
                    f"Error fetching maintenance margin ratio: {data}")
                return None

            if isinstance(data, list) and len(data) > 0:
                # Access the first item and then get 'brackets'
                brackets_data = data[0].get('brackets')
                if brackets_data:
                    for bracket in brackets_data:
                        # Assuming you want the first bracket or specific logic to select the bracket
                        return bracket['maintMarginRatio']
            return None  # Handle case where the maintenance margin ratio is not found
        except Exception as e:
            logging.error("error in fetch_maintenance_margin_ratio", e)

    def calculate_liquidation_price(self, entry_price, leverage, maint_margin_ratio, position_side='LONG'):
        if entry_price is None or leverage is None or maint_margin_ratio is None:
            logging.error("Invalid input(s). Cannot calculate liquidation price.",
                          entry_price, leverage, maint_margin_ratio)
            return None
        try:
            if position_side == 'LONG':
                # Simplified calculation for illustrative purposes
                liquidation_price = entry_price * \
                    (1 - (1 / leverage) + maint_margin_ratio)
            else:  # SHORT
                liquidation_price = entry_price * \
                    (1 + (1 / leverage) - maint_margin_ratio)
            return liquidation_price
        except Exception as e:
            logging.error("error in calculate_liquidation_price", e)

    def print_total_pnl(self):
        try:
            # Print total PnL for each condition per coin
            pipeline_conditions_per_coin = [
                {
                    "$group": {
                        "_id": {
                            "algo_type": "$algo_type",
                            "coin_pair": "$coin_pair"
                        },
                        "total_pnl": {"$sum": "$actual_pnl_usd"},
                        "total_gains": {"$sum": {"$cond": [{"$gt": ["$actual_pnl_usd", 0]}, "$actual_pnl_usd", 0]}},
                        "total_losses": {"$sum": {"$cond": [{"$lt": ["$actual_pnl_usd", 0]}, "$actual_pnl_usd", 0]}}
                    }
                }
            ]
            results_conditions_per_coin = self.db_manager.db["trading_opportunities"].aggregate(
                pipeline_conditions_per_coin)
            logging.info("Total PnL for each condition per coin:")
            for result in results_conditions_per_coin:
                algo_type = result["_id"]["algo_type"]
                coin_pair = result["_id"]["coin_pair"]
                total_pnl = result["total_pnl"]
                total_gains = result.get("total_gains", 0)
                total_losses = result.get("total_losses", 0)
                logging.info(
                    f"Algorithm: {algo_type}, Coin: {coin_pair}, Total PnL: {total_pnl}, Total Gains: {total_gains}, Total Losses: {total_losses}")

            # Print total PnL for each condition across all coins
            pipeline_conditions = [
                {
                    "$group": {
                        "_id": "$algo_type",
                        "total_pnl": {"$sum": "$actual_pnl_usd"},
                        "total_gains": {"$sum": {"$cond": [{"$gt": ["$actual_pnl_usd", 0]}, "$actual_pnl_usd", 0]}},
                        "total_losses": {"$sum": {"$cond": [{"$lt": ["$actual_pnl_usd", 0]}, "$actual_pnl_usd", 0]}}
                    }
                }
            ]
            results_conditions = self.db_manager.db["trading_opportunities"].aggregate(
                pipeline_conditions)
            logging.info("Total PnL for each condition across all coins:")
            for result in results_conditions:
                algo_type = result["_id"]
                total_pnl = result["total_pnl"]
                total_gains = result.get("total_gains", 0)
                total_losses = result.get("total_losses", 0)
                logging.info(
                    f"Algorithm: {algo_type}, Total PnL: {total_pnl}, Total Gains: {total_gains}, Total Losses: {total_losses}")

            # Print total PnL for each coin across all conditions
            pipeline_coins = [
                {
                    "$group": {
                        "_id": "$coin_pair",
                        "total_pnl": {"$sum": "$actual_pnl_usd"},
                        "total_gains": {"$sum": {"$cond": [{"$gt": ["$actual_pnl_usd", 0]}, "$actual_pnl_usd", 0]}},
                        "total_losses": {"$sum": {"$cond": [{"$lt": ["$actual_pnl_usd", 0]}, "$actual_pnl_usd", 0]}}
                    }
                }
            ]
            results_coins = self.db_manager.db["trading_opportunities"].aggregate(
                pipeline_coins)
            logging.info("Total PnL for each coin across all conditions:")
            for result in results_coins:
                coin_pair = result["_id"]
                total_pnl = result["total_pnl"]
                total_gains = result.get("total_gains", 0)
                total_losses = result.get("total_losses", 0)
                logging.info(
                    f"Coin: {coin_pair}, Total PnL: {total_pnl}, Total Gains: {total_gains}, Total Losses: {total_losses}")

            # Print total PnL across all conditions and coins
            pipeline_total = [
                {
                    "$group": {
                        "_id": None,
                        "total_pnl": {"$sum": "$actual_pnl_usd"},
                        "total_gains": {"$sum": {"$cond": [{"$gt": ["$actual_pnl_usd", 0]}, "$actual_pnl_usd", 0]}},
                        "total_losses": {"$sum": {"$cond": [{"$lt": ["$actual_pnl_usd", 0]}, "$actual_pnl_usd", 0]}}
                    }
                }
            ]
            result_total = list(
                self.db_manager.db["trading_opportunities"].aggregate(pipeline_total))
            if result_total:
                total_pnl = result_total[0]["total_pnl"]
                total_gains = result_total[0].get("total_gains", 0)
                total_losses = result_total[0].get("total_losses", 0)
                logging.info(
                    f"Total PnL across all conditions and coins: {total_pnl}, Total Gains: {total_gains}, Total Losses: {total_losses}")

        except Exception as e:
            logging.error(f"Error in print_total_pnl: {e}")


def fetch_total_volume():
    try:
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        response = requests.get(url)
        data = json.loads(response.text)

        if isinstance(data, list):
            volume_dict = {item['symbol']: float(
                item['quoteVolume']) for item in data if 'quoteVolume' in item}
            return volume_dict
        else:
            logging.error("Unexpected response format")
            return {}
    except Exception as e:
        logging.error(f"Error in fetch_total_volume: {e}")
        return {}


def fetch_data_for_month(coin_name, interval, year_month, base_url):
    url = f"{base_url}/{coin_name}/{interval}/{coin_name}-{interval}-{year_month}.zip"
    response = requests.get(url)

    if response.status_code == 200:
        with ZipFile(io.BytesIO(response.content)) as z:
            csv_data = [(year_month, z.open(file_name).read())
                        for file_name in z.namelist()]
        return csv_data
    elif response.status_code == 404:
        logging.warning(f"No data found for {coin_name} in {year_month}")
    return []



def fetch_and_bundle_coin_data(coin_name, interval='1h'):
    base_url = "https://data.binance.vision/data/futures/um/monthly/klines"
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 5, 1)

    csv_data = []
    current_date = end_date
    date_list = []

    while current_date >= start_date:
        year_month = current_date.strftime("%Y-%m")
        date_list.append(year_month)
        current_date = current_date.replace(day=1) - pd.DateOffset(days=1)

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(fetch_data_for_month, coin_name,
                                   interval, year_month, base_url) for year_month in date_list]
        for future in as_completed(futures):
            csv_data.extend(future.result())

    if not csv_data:
        print(
            f"No data found for {coin_name} within the specified date range.")
        return None

    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, 'w', ZIP_DEFLATED) as z:
        for year_month, data in csv_data:
            file_name = f"{coin_name}-{interval}-{year_month}.csv"
            z.writestr(file_name, data)

    zip_buffer.seek(0)

    csv_dfs = []
    with ZipFile(zip_buffer, 'r') as z:
        for file_name in z.namelist():
            with z.open(file_name) as f:
                df = pd.read_csv(f)
                csv_dfs.append(df)

    if not csv_dfs:
        print(f"No valid CSV files found for {coin_name}.")
        return None

    combined_df = pd.concat(csv_dfs, ignore_index=True)
    processed_df = IndicatorCalculator.calculate_indicators(
        combined_df, coin_name)

    return processed_df


def fetch_all_coins_data(coin_pairs, interval):
    combined_data = {}
    for coin in coin_pairs:
        df = fetch_and_bundle_coin_data(coin, interval)
        if df is not None:
            combined_data[coin] = df

    return combined_data

def combine_coin_data(combined_data):
    all_data = []
    
    for coin, df in combined_data.items():
        # Check if df is not None and not empty before appending
        if df is not None and not df.empty:
            df['coin'] = coin
            all_data.append(df)
        else:
            logging.warning(f"No valid data for {coin}. Skipping.")
    
    # If no valid data was collected, raise an appropriate warning or handle it
    if not all_data:
        raise ValueError("No valid data available to concatenate.")
    
    # Proceed with concatenating the non-empty DataFrames
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values(by='close_time')
    return combined_df




def save_to_database(db_manager, combined_df):
    records = combined_df.to_dict(orient='records')
    db_manager.db['combined_kline_data'].insert_many(records)
    logging.info(
        f"Inserted {len(records)} records into combined_kline_data collection")


from concurrent.futures import ThreadPoolExecutor

def process_data(db_manager, trend_analyzer, coin_pairs, interval):
    timestamps = db_manager.db['combined_kline_data'].distinct('close_time')

    with ThreadPoolExecutor(max_workers=16) as executor:  # Set a reasonable max_workers
        for timestamp in timestamps:
            current_rows = list(db_manager.db['combined_kline_data'].find({'close_time': timestamp}))
            prev_rows = list(db_manager.db['combined_kline_data'].find({'close_time': {'$lt': timestamp}})
                             .sort('close_time', pymongo.DESCENDING).limit(len(current_rows)))
            two_prev_rows = list(db_manager.db['combined_kline_data'].find({'close_time': {'$lt': timestamp}})
                                 .sort('close_time', pymongo.DESCENDING).limit(len(current_rows) * 2))

            futures = []
            for coin in set(row['coin'] for row in current_rows):
                current_row = next((row for row in current_rows if row['coin'] == coin), None)
                prev_row = next((row for row in prev_rows if row['coin'] == coin), None)
                two_prev_row = next((row for row in two_prev_rows if row['coin'] == coin), None)

                if current_row and prev_row and two_prev_row:
                    futures.append(executor.submit(analyze_trend, trend_analyzer, coin, interval, current_row, prev_row, two_prev_row, current_rows, prev_rows))
                    
            for future in futures:
                future.result()  # Ensures we wait for all threads to finish.

def analyze_trend(trend_analyzer, coin, interval, current_row, prev_row, two_prev_row, current_rows, prev_rows):
    trend_analyzer.check_for_short(coin, interval, current_row, prev_row, two_prev_row)
    trend_analyzer.check_for_long(coin, interval, current_row, prev_row, two_prev_row)
    trend_analyzer.monitor_and_close_positions(interval, current_rows, prev_rows)


def main():
    b_time = "4h"
    backtest_name = "backtesting_" + b_time + "main1"
    if os.path.isfile(backtest_name):
        os.remove(backtest_name)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(backtest_name),
            logging.StreamHandler()
        ]
    )

    start_time = time.time()
    volume_dict = fetch_total_volume()
    logging.info("Fetched total volume. Time taken: %.2f seconds",
                 time.time() - start_time)
    start_time = time.time()

    filtered_coin_pairs = [coin for coin,
                           volume in volume_dict.items() if volume > 120000000]
    filtered_coin_pairs = filtered_coin_pairs
    logging.info("Filtered coin pairs. Time taken: %.2f seconds",
                 time.time() - start_time)
    logging.info("Number of filtered coin pairs: %d", len(filtered_coin_pairs))
    logging.info("Filtered coin pairs: %s", filtered_coin_pairs)
    start_time = time.time()

    interval_list = [b_time]
    db_manager = DatabaseManager(
        backtest_name,
        'mongodb://berkeberke:M3j2veuyQWsjFLc5YpEy5LjK@localhost:27017', 
        filtered_coin_pairs, 
        interval_list
    )

    logging.info("Initialized DatabaseManager. Time taken: %.2f seconds",
                 time.time() - start_time)
    start_time = time.time()

    combined_data = fetch_all_coins_data(filtered_coin_pairs, b_time)
    combined_df = combine_coin_data(combined_data)
    save_to_database(db_manager, combined_df)

    trend_analyzer = TrendAnalyzer(db_manager, interval_list)
    trend_analyzer.update_margins()
    logging.info("Updated margins. Time taken: %.2f seconds",
                 time.time() - start_time)
    logging.info("Margins: %s", trend_analyzer.margins)
    start_time = time.time()

    process_data(db_manager, trend_analyzer, filtered_coin_pairs, b_time)
    logging.info("Processed data. Time taken: %.2f seconds",
                 time.time() - start_time)


if __name__ == "__main__":
    main()
