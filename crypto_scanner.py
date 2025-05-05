import time
import datetime
import pandas as pd
import numpy as np
from binance.client import Client
from typing import Dict, List, Tuple, Any, Optional
import logging
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# 币安API密钥 - 直接内置在代码中
API_KEY = "R1rNhHUjRNZ2Qkrbl05Odc7GseGaVSPqr7l7NHsI0AUHtY6sM4C24wJW14c01m5B"
API_SECRET = "AQPSTJN2CjfnvesLCdjKJffo5obacHqpMJIhtZPpoXwR40Ja90F03jSS9so5wJjW"

# 导入必要的模块
try:
    from integration_module import calculate_enhanced_indicators, comprehensive_market_analysis
    from quality_module import calculate_quality_score
    from indicators_module import get_smc_trend_and_duration
    from multi_timeframe_module import MultiTimeframeCoordinator
    from config import CONFIG

    print("成功导入所有必要模块")
except ImportError as e:
    print(f"警告: 无法导入一些必要模块: {e}")
    print("尝试使用内置的简化函数代替...")
    CONFIG = {
        "TRADE_PAIRS": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "SOLUSDT"],
        "MIN_MARGIN_BALANCE": 10,
        "MAX_PURCHASES_PER_ROUND": 10,
        "TREND_DURATION_THRESHOLD": 1440,
        "ATR_HEDGE_THRESHOLD": 1.5
    }


# 必要的模块无法导入时的简化实现
def simplified_calculate_enhanced_indicators(df):
    """简化的指标计算函数"""
    try:
        # 计算基本指标
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            # 计算ATR (平均真实范围)
            df['tr1'] = abs(df['high'] - df['low'])
            df['tr2'] = abs(df['high'] - df['close'].shift())
            df['tr3'] = abs(df['low'] - df['close'].shift())
            df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['ATR'] = df['TR'].rolling(window=14).mean()

            # 计算EMA (指数移动平均)
            df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
            df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()

            # 计算RSI (相对强弱指数)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).fillna(0)
            loss = -delta.where(delta < 0, 0).fillna(0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
            df['RSI'] = 100 - (100 / (1 + rs))

            # 计算布林带
            df['BB_Middle'] = df['close'].rolling(window=20).mean()
            df['BB_Std'] = df['close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
            df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']

            # 计算MACD
            df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA12'] - df['EMA26']
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

            # 计算ADX (平均方向指数)
            df['DM_plus'] = np.where((df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']),
                                     np.maximum(df['high'] - df['high'].shift(), 0), 0)
            df['DM_minus'] = np.where((df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()),
                                      np.maximum(df['low'].shift() - df['low'], 0), 0)
            df['DM_plus_smooth'] = df['DM_plus'].rolling(window=14).mean()
            df['DM_minus_smooth'] = df['DM_minus'].rolling(window=14).mean()
            df['ATR14'] = df['ATR'].rolling(window=14).mean()
            df['DI_plus'] = 100 * df['DM_plus_smooth'] / df['ATR14'].replace(0, np.finfo(float).eps)
            df['DI_minus'] = 100 * df['DM_minus_smooth'] / df['ATR14'].replace(0, np.finfo(float).eps)
            df['DX'] = 100 * abs(df['DI_plus'] - df['DI_minus']) / (df['DI_plus'] + df['DI_minus']).replace(0, np.finfo(
                float).eps)
            df['ADX'] = df['DX'].rolling(window=14).mean()

            print(f"简化指标计算完成, 计算了 {len(df)} 行数据")
            return df
        else:
            print("数据帧缺少必要的列")
            return None
    except Exception as e:
        print(f"简化指标计算出错: {e}")
        return None


def simplified_quality_score(df):
    """简化的质量评分计算函数"""
    try:
        # 首先检查是否包含必要指标
        if not all(col in df.columns for col in ['EMA5', 'EMA20', 'RSI', 'ATR', 'MACD']):
            df = simplified_calculate_enhanced_indicators(df)
            if df is None:
                return 5.0, {}  # 默认中等评分

        # 提取最新指标值
        latest = df.iloc[-1]

        # 1. 趋势评估 (0-3分)
        trend_score = 0
        if 'EMA5' in df.columns and 'EMA20' in df.columns:
            if latest['EMA5'] > latest['EMA20']:
                # 上升趋势
                trend_score = 3.0
                trend = "UP"
            elif latest['EMA5'] < latest['EMA20']:
                # 下降趋势
                trend_score = 1.0
                trend = "DOWN"
            else:
                # 中性趋势
                trend_score = 2.0
                trend = "NEUTRAL"
        else:
            trend_score = 1.5
            trend = "UNKNOWN"

        # 2. RSI评估 (0-2分)
        rsi_score = 0
        if 'RSI' in df.columns:
            rsi = latest['RSI']
            if rsi < 30:  # 超卖
                rsi_score = 2.0
            elif rsi > 70:  # 超买
                rsi_score = 0.5
            else:
                rsi_score = 1.0
        else:
            rsi_score = 1.0

        # 3. 波动性评估 (0-2分)
        volatility_score = 0
        if 'ATR' in df.columns:
            atr = latest['ATR']
            price = latest['close']
            atr_ratio = atr / price * 100  # ATR占价格的百分比

            if atr_ratio < 1.0:  # 低波动
                volatility_score = 0.5
            elif atr_ratio < 3.0:  # 中等波动
                volatility_score = 2.0
            else:  # 高波动
                volatility_score = 1.0
        else:
            volatility_score = 1.0

        # 4. MACD评估 (0-2分)
        macd_score = 0
        if 'MACD' in df.columns and 'MACD_signal' in df.columns:
            if latest['MACD'] > latest['MACD_signal']:
                macd_score = 2.0
            elif latest['MACD'] < latest['MACD_signal']:
                macd_score = 0.5
            else:
                macd_score = 1.0
        else:
            macd_score = 1.0

        # 5. 价格位置评估 (0-1分)
        price_position_score = 0
        if all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            price = latest['close']
            upper = latest['BB_Upper']
            lower = latest['BB_Lower']
            middle = latest['BB_Middle']

            if price < lower:  # 价格低于下轨
                price_position_score = 1.0
            elif price > upper:  # 价格高于上轨
                price_position_score = 0.2
            elif price < middle:  # 价格低于中轨
                price_position_score = 0.8
            else:  # 价格高于中轨
                price_position_score = 0.5
        else:
            price_position_score = 0.5

        # 计算总分
        total_score = trend_score + rsi_score + volatility_score + macd_score + price_position_score

        # 返回结果
        metrics = {
            "trend": trend,
            "trend_score": trend_score,
            "rsi": latest.get('RSI', 50),
            "rsi_score": rsi_score,
            "volatility_score": volatility_score,
            "macd_score": macd_score,
            "price_position_score": price_position_score
        }

        return total_score, metrics

    except Exception as e:
        print(f"简化质量评分计算出错: {e}")
        return 5.0, {"error": str(e)}  # 默认中等评分


def simplified_get_trend(df):
    """简化的趋势判断函数"""
    try:
        if not all(col in df.columns for col in ['EMA5', 'EMA20']):
            df = simplified_calculate_enhanced_indicators(df)
            if df is None:
                return "NEUTRAL", 0, {"confidence": "无"}

        # 检查EMA指标判断趋势
        if df['EMA5'].iloc[-1] > df['EMA20'].iloc[-1]:
            # 上升趋势
            trend = "UP"

            # 计算持续了多少根K线
            duration = 0
            for i in range(len(df) - 1, 0, -1):
                if df['EMA5'].iloc[i] > df['EMA20'].iloc[i]:
                    duration += 1
                else:
                    break

            # 转换为分钟
            duration_minutes = duration * 15  # 假设15分钟K线

            # 判断置信度
            if 'ADX' in df.columns and df['ADX'].iloc[-1] > 25:
                confidence = "高"
            elif duration > 10:
                confidence = "中高"
            else:
                confidence = "中"

        elif df['EMA5'].iloc[-1] < df['EMA20'].iloc[-1]:
            # 下降趋势
            trend = "DOWN"

            # 计算持续了多少根K线
            duration = 0
            for i in range(len(df) - 1, 0, -1):
                if df['EMA5'].iloc[i] < df['EMA20'].iloc[i]:
                    duration += 1
                else:
                    break

            # 转换为分钟
            duration_minutes = duration * 15  # 假设15分钟K线

            # 判断置信度
            if 'ADX' in df.columns and df['ADX'].iloc[-1] > 25:
                confidence = "高"
            elif duration > 10:
                confidence = "中高"
            else:
                confidence = "中"
        else:
            # 中性趋势
            trend = "NEUTRAL"
            duration_minutes = 0
            confidence = "无"

        trend_info = {
            "confidence": confidence,
            "reason": f"基于EMA5和EMA20的交叉判断"
        }

        return trend, duration_minutes, trend_info

    except Exception as e:
        print(f"简化趋势判断出错: {e}")
        return "NEUTRAL", 0, {"confidence": "无", "error": str(e)}


class MockMTFCoordinator:
    """模拟多时间框架协调器"""

    def __init__(self, client, logger):
        self.client = client
        self.logger = logger

    def generate_signal(self, symbol, quality_score):
        """生成简化的信号"""
        try:
            # 获取数据
            klines = self.client.get_klines(symbol=symbol, interval="15m", limit=100)
            df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'volume',
                                               'close_time', 'quote_volume', 'trades',
                                               'taker_buy_base', 'taker_buy_quote', 'ignore'])

            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])

            # 计算EMA
            df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
            df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()

            # 判断趋势
            if df['EMA5'].iloc[-1] > df['EMA20'].iloc[-1]:
                signal = "BUY"
                adjusted_score = quality_score * 1.1  # 上升趋势加分
            elif df['EMA5'].iloc[-1] < df['EMA20'].iloc[-1]:
                signal = "SELL"
                adjusted_score = quality_score * 0.9  # 下降趋势减分
            else:
                signal = "NEUTRAL"
                adjusted_score = quality_score

            # 模拟详细信息
            details = {
                "coherence": {
                    "agreement_level": "中等一致",
                    "dominant_trend": "UP" if signal == "BUY" else "DOWN" if signal == "SELL" else "NEUTRAL",
                    "recommendation": signal
                }
            }

            return signal, adjusted_score, details

        except Exception as e:
            print(f"模拟多时间框架信号生成失败: {e}")
            return "NEUTRAL", quality_score, {"coherence": {"agreement_level": "无", "dominant_trend": "NEUTRAL"}}


class CryptoCurrencyScanner:
    """加密货币扫描器，用于识别最适合现有交易算法的加密货币"""

    def __init__(self, api_key: str, api_secret: str, config: dict = None):
        """初始化扫描器"""
        self.api_key = api_key
        self.api_secret = api_secret
        self.config = config if config else CONFIG
        self.client = Client(api_key, api_secret)

        # 设置日志记录
        self.setup_logging()

        # 初始化多时间框架协调器
        try:
            self.mtf_coordinator = MultiTimeframeCoordinator(self.client, self.logger)
            print("成功初始化多时间框架协调器")
        except Exception as e:
            print(f"无法初始化多时间框架协调器: {e}")
            print("将使用模拟版本替代")
            self.mtf_coordinator = MockMTFCoordinator(self.client, self.logger)

        # 数据缓存
        self.historical_data_cache = {}
        self.quality_scores_cache = {}

        # 冷却追踪 - 存储 symbol: timestamp 条目
        self.cooldown_symbols = {}
        self.cooldown_period = 30 * 60  # 30分钟（秒）

        # 分析结果追踪
        self.scan_history = []
        self.prediction_accuracy = {}

        # 创建必要的目录
        os.makedirs("scan_results", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        self.logger.info(f"加密货币扫描器初始化完成")

        # 清理缓存
        self.clean_cache()

    def setup_logging(self):
        """设置日志配置"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        self.logger = logging.getLogger("crypto_scanner")
        self.logger.setLevel(logging.INFO)

        # 文件处理程序
        file_handler = logging.FileHandler(f"{log_dir}/crypto_scanner.log")
        file_handler.setLevel(logging.INFO)

        # 控制台处理程序
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 格式化程序
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加处理程序
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def clean_cache(self):
        """清理缓存文件和数据"""
        try:
            # 清理数据缓存
            self.historical_data_cache = {}

            # 清理扫描历史记录目录中的旧文件
            scan_results_dir = "scan_results"
            if os.path.exists(scan_results_dir):
                # 删除15天以上的文件
                current_time = time.time()
                for filename in os.listdir(scan_results_dir):
                    file_path = os.path.join(scan_results_dir, filename)
                    if os.path.isfile(file_path):
                        file_age = current_time - os.path.getmtime(file_path)
                        if file_age > 15 * 24 * 3600:  # 15天
                            os.remove(file_path)
                            print(f"已删除旧文件: {file_path}")

            # 清理日志目录中的旧文件
            log_dir = "logs"
            if os.path.exists(log_dir):
                # 保留最新的5个日志文件
                log_files = []
                for filename in os.listdir(log_dir):
                    file_path = os.path.join(log_dir, filename)
                    if os.path.isfile(file_path) and filename.endswith('.log'):
                        log_files.append((file_path, os.path.getmtime(file_path)))

                # 按修改时间排序
                log_files.sort(key=lambda x: x[1], reverse=True)

                # 删除旧文件
                for file_path, _ in log_files[5:]:
                    os.remove(file_path)
                    print(f"已删除旧日志文件: {file_path}")

            self.logger.info("缓存清理完成")
        except Exception as e:
            self.logger.error(f"清理缓存时出错: {e}")

    def get_all_usdt_pairs(self, min_volume: float = 8500000) -> List[str]:
        """
        获取所有有足够交易量的USDT期货交易对

        参数:
            min_volume: 最低24小时交易量（USDT），默认为850万USDT

        返回:
            交易对符号列表 (例如 'BTCUSDT')
        """
        try:
            # 获取期货交易所信息
            exchange_info = self.client.futures_exchange_info()

            # 筛选出当前可交易的USDT对
            usdt_pairs = [
                symbol['symbol'] for symbol in exchange_info['symbols']
                if symbol['symbol'].endswith('USDT') and
                   symbol['status'] == 'TRADING'
            ]

            self.logger.info(f"从期货交易所找到 {len(usdt_pairs)} 个USDT交易对")

            # 检查交易量
            valid_pairs = []

            # 获取24小时价格变动统计
            try:
                tickers = self.client.futures_ticker()
                volume_dict = {ticker['symbol']: float(ticker['quoteVolume']) for ticker in tickers}

                for symbol in usdt_pairs:
                    if symbol in volume_dict and volume_dict[symbol] >= min_volume:
                        valid_pairs.append(symbol)
                        self.logger.info(f"{symbol}交易量: {volume_dict[symbol]:.2f} USDT")
            except Exception as e:
                self.logger.error(f"获取期货交易量数据失败: {e}")
                # 如果无法获取交易量，则返回所有交易对
                valid_pairs = usdt_pairs

            self.logger.info(f"找到 {len(valid_pairs)} 个交易量 >= {min_volume} USDT 的USDT期货交易对")
            return valid_pairs

        except Exception as e:
            self.logger.error(f"获取USDT期货交易对时出错: {e}")
            # 如果发生错误，返回config中的交易对作为后备方案
            config_pairs = self.config.get("TRADE_PAIRS", [])
            self.logger.info(f"使用config中的交易对作为后备: {len(config_pairs)}个")
            return config_pairs

    def get_historical_data(self, symbol: str, interval: str = "15m",
                            limit: int = 200, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """获取期货交易对的历史OHLCV数据"""
        cache_key = f"{symbol}_{interval}_{limit}"
        current_time = time.time()

        # 缓存持续时间 - 3分钟
        cache_ttl = 180

        # 检查缓存，除非指定了强制刷新
        if not force_refresh and cache_key in self.historical_data_cache:
            cache_entry = self.historical_data_cache[cache_key]
            if current_time - cache_entry['timestamp'] < cache_ttl:
                return cache_entry['data']

        try:
            # 获取期货K线数据
            klines = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )

            if not klines:
                self.logger.warning(f"未返回{symbol}的期货K线数据")
                return None

            # 转换为DataFrame
            df = pd.DataFrame(klines, columns=[
                'time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'trades',
                'taker_base_vol', 'taker_quote_vol', 'ignore'
            ])

            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 转换时间列
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

            # 更新缓存
            self.historical_data_cache[cache_key] = {
                'data': df,
                'timestamp': current_time
            }

            return df

        except Exception as e:
            self.logger.error(f"获取{symbol}的期货历史数据时出错: {e}")
            return None

    def filter_overextended_coins(self, symbol: str) -> bool:
        """过滤已经大幅上涨可能即将回调的货币"""
        try:
            df = self.get_historical_data(symbol)
            if df is None or len(df) < 20:
                return False  # 数据不足，无法判断，默认不过滤

            # 计算短期涨幅
            current_price = df['close'].iloc[-1]
            price_5d_ago = df['close'].iloc[-20]  # 约5天前(20根15分钟K线)
            change_5d = (current_price - price_5d_ago) / price_5d_ago * 100

            # 检查RSI
            rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50

            # 检查MACD方向
            macd_trend = "unknown"
            if 'MACD' in df.columns and 'MACD_signal' in df.columns:
                macd = df['MACD'].iloc[-1]
                macd_prev = df['MACD'].iloc[-3]
                macd_signal = df['MACD_signal'].iloc[-1]

                if macd > macd_prev:
                    macd_trend = "up"
                else:
                    macd_trend = "down"

            # 过滤条件组合:
            # 1. 如果5天涨幅超过20%且RSI大于70，认为可能过度上涨
            if change_5d > 20 and rsi > 70:
                self.logger.info(f"{symbol} 5天内涨幅达{change_5d:.2f}%且RSI达{rsi:.2f}，可能过度上涨，跳过")
                return True

            # 2. 如果5天涨幅超过15%且RSI大于65且MACD开始向下
            if change_5d > 15 and rsi > 65 and macd_trend == "down":
                self.logger.info(f"{symbol} 5天内涨幅达{change_5d:.2f}%，RSI达{rsi:.2f}，MACD开始下降，可能即将回调，跳过")
                return True

            # 3. 如果5天内涨幅超过30%，无论其他指标如何，都认为风险过高
            if change_5d > 30:
                self.logger.info(f"{symbol} 5天内涨幅高达{change_5d:.2f}%，风险过高，跳过")
                return True

            return False  # 默认不过滤

        except Exception as e:
            self.logger.error(f"过滤{symbol}时出错: {e}")
            return False  # 出错时不过滤

    def calculate_expected_movement(self, df: pd.DataFrame, horizon_minutes: int = 60) -> Tuple[float, float]:
        """
        基于历史数据计算预期价格变动

        参数:
            df: 包含价格数据的DataFrame
            horizon_minutes: 价格预测的时间范围（分钟）

        返回:
            (预期变动百分比, 预测价格)的元组
        """
        if df is None or df.empty or len(df) < 20:
            return 0.0, 0.0

        try:
            # 获取当前价格
            current_price = df['close'].iloc[-1]

            # 计算波动性以提供上下文
            if 'ATR' in df.columns:
                volatility = df['ATR'].iloc[-1] / current_price * 100
            else:
                # 如果ATR不可用，简单计算波动性
                price_changes = df['close'].pct_change().dropna() * 100
                volatility = price_changes.std()

            # 使用简单线性回归进行预测
            window_length = min(60, len(df))
            window = df['close'].tail(window_length)
            smoothed = window.rolling(window=3, min_periods=1).mean().bfill()

            x = np.arange(len(smoothed))
            slope, intercept = np.polyfit(x, smoothed, 1)

            # 计算目标时间范围所需的周期数
            candle_minutes = 15  # 假设15分钟K线
            candles_needed = horizon_minutes / candle_minutes

            # 预测未来价格
            predicted_price = current_price + slope * candles_needed

            # 计算预期变动百分比
            expected_movement = abs(predicted_price - current_price) / current_price * 100

            # 根据市场条件应用乘数
            if 'ADX' in df.columns:
                adx = df['ADX'].iloc[-1]
                # 强趋势可以放大变动
                if adx > 30:
                    expected_movement *= 1.2
                # 弱趋势可能减少变动
                elif adx < 15:
                    expected_movement *= 0.8

            # 确保预测基于波动性是合理的
            max_expected = volatility * 2.5  # 上限为当前波动性的2.5倍
            expected_movement = min(expected_movement, max_expected)

            return expected_movement, predicted_price

        except Exception as e:
            self.logger.error(f"计算预期变动时出错: {e}")
            return 0.0, 0.0

    def is_in_cooldown(self, symbol: str) -> bool:
        """
        检查交易对是否在止损或止盈后的冷却期内

        参数:
            symbol: 交易对符号

        返回:
            如果在冷却期内则为True，否则为False
        """
        if symbol not in self.cooldown_symbols:
            return False

        cooldown_start = self.cooldown_symbols[symbol]
        current_time = time.time()

        # 检查冷却期是否已过
        if current_time - cooldown_start > self.cooldown_period:
            # 从冷却列表中移除
            del self.cooldown_symbols[symbol]
            return False

        # 仍在冷却期内
        return True

    def add_to_cooldown(self, symbol: str):
        """在达到止损或止盈后将交易对添加到冷却期"""
        self.cooldown_symbols[symbol] = time.time()
        self.logger.info(f"{symbol}已添加到冷却期，持续{self.cooldown_period / 60:.1f}分钟")

    def check_stop_loss_take_profit_hit(self, symbol: str) -> bool:
        """
        检查交易对是否最近触发了止损或止盈
        使用最近的价格变化和波动性来检测可能的止损/止盈事件

        参数:
            symbol: 交易对符号

        返回:
            如果可能触发了止损或止盈则为True，否则为False
        """
        df = self.get_historical_data(symbol)
        if df is None or len(df) < 30:
            return False

        try:
            # 获取最近的K线
            recent_df = df.tail(30)

            # 计算最近的波动性
            recent_volatility = recent_df['close'].pct_change().std() * 100

            # 寻找可能表明止损/止盈的大幅价格变化
            price_changes = recent_df['close'].pct_change() * 100
            max_up_move = price_changes.max()
            max_down_move = price_changes.min()

            # 基于波动性的阈值
            tp_threshold = max(2.5, recent_volatility * 3)
            sl_threshold = min(-1.75, recent_volatility * -2.5)

            # 检查是否有任何变动超过阈值
            if max_up_move > tp_threshold or max_down_move < sl_threshold:
                # 计算事件发生时间（以K线为单位）
                if max_up_move > tp_threshold:
                    event_index = price_changes.idxmax()
                    event_type = "止盈"
                else:
                    event_index = price_changes.idxmin()
                    event_type = "止损"

                # 计算事件发生在多少K线之前
                event_candles_ago = len(recent_df) - recent_df.index.get_loc(event_index)

                # 只考虑最近10根K线内的事件（15分钟K线为2.5小时）
                if event_candles_ago <= 10:
                    self.logger.info(f"{symbol}可能在{event_candles_ago}根K线前触发了{event_type}")
                    return True

            return False

        except Exception as e:
            self.logger.error(f"检查{symbol}的止损/止盈时出错: {e}")
            return False

    def analyze_single_coin(self, symbol: str) -> Dict[str, Any]:
        """
        全面分析单个加密货币

        参数:
            symbol: 交易对符号

        返回:
            包含分析结果的字典
        """
        self.logger.info(f"分析{symbol}...")

        # 首先检查冷却期
        if self.is_in_cooldown(symbol):
            self.logger.info(f"{symbol}处于冷却期，跳过分析")
            return {
                "symbol": symbol,
                "quality_score": 0,
                "expected_movement": 0.0,
                "status": "cooldown"
            }

        # 检查是否最近触发了止损/止盈
        if self.check_stop_loss_take_profit_hit(symbol):
            self.logger.info(f"{symbol}最近触发了止损或止盈，添加到冷却期")
            self.add_to_cooldown(symbol)
            return {
                "symbol": symbol,
                "quality_score": 0,
                "expected_movement": 0.0,
                "status": "recent_sl_tp"
            }

        try:
            # 获取历史数据
            df = self.get_historical_data(symbol)
            if df is None or len(df) < 20:
                return {
                    "symbol": symbol,
                    "quality_score": 0,
                    "expected_movement": 0.0,
                    "status": "insufficient_data"
                }

            # 计算指标
            try:
                df = calculate_enhanced_indicators(df)
            except Exception as e:
                print(f"使用原始enhanced_indicators函数失败: {e}")
                df = simplified_calculate_enhanced_indicators(df)

            if df is None or df.empty:
                return {
                    "symbol": symbol,
                    "quality_score": 0,
                    "expected_movement": 0.0,
                    "status": "indicators_failed"
                }

            # 计算质量评分
            try:
                quality_score, metrics = calculate_quality_score(df, self.client, symbol)
            except Exception as e:
                print(f"使用原始quality_score函数失败: {e}")
                quality_score, metrics = simplified_quality_score(df)

            # 获取趋势信息
            try:
                trend, duration, trend_info = get_smc_trend_and_duration(df)
            except Exception as e:
                print(f"使用原始trend函数失败: {e}")
                trend, duration, trend_info = simplified_get_trend(df)

            # 使用多时间框架分析获取额外的信号强度
            signal, adjusted_score, details = self.mtf_coordinator.generate_signal(symbol, quality_score)

            # 计算预期价格变动
            expected_movement, predicted_price = self.calculate_expected_movement(df)

            # 获取当前价格
            current_price = df['close'].iloc[-1]

            # 尝试进行综合市场分析
            try:
                market_analysis = comprehensive_market_analysis(df)
            except Exception as e:
                print(f"综合市场分析失败: {e}")
                market_analysis = {"overall": {"signal": "NEUTRAL", "quality_score": quality_score}}

            # 确定信号方向
            if signal.startswith("BUY"):
                signal_type = "BUY"
            elif signal.startswith("SELL"):
                signal_type = "SELL"
            else:
                signal_type = "NEUTRAL"

            # 编译结果
            analysis_result = {
                "symbol": symbol,
                "timestamp": time.time(),
                "current_price": current_price,
                "quality_score": quality_score,
                "adjusted_score": adjusted_score,
                "trend": trend,
                "trend_confidence": trend_info["confidence"],
                "trend_duration": duration,
                "signal": signal_type,
                "expected_movement": expected_movement,
                "predicted_price": predicted_price,
                "status": "analyzed",
                "mtf_analysis": {
                    "signal": signal,
                    "coherence": details.get("coherence", {})
                },
                "metrics": metrics
            }

            self.logger.info(
                f"{symbol}分析完成 - 评分: {adjusted_score:.2f}, 预期变动: {expected_movement:.2f}%, 信号: {signal_type}")
            return analysis_result

        except Exception as e:
            self.logger.error(f"分析{symbol}时出错: {e}")
            return {
                "symbol": symbol,
                "quality_score": 0,
                "expected_movement": 0.0,
                "status": "analysis_error",
                "error": str(e)
            }

    def run_scan_round(self, symbols_to_scan: List[str],
                       min_expected_movement: float = 1.7) -> List[Dict[str, Any]]:
        """
        对提供的交易对运行完整的扫描轮次

        参数:
            symbols_to_scan: 要分析的交易对列表
            min_expected_movement: 最低预期价格变动百分比

        返回:
            按评分排序的分析结果列表
        """
        self.logger.info(f"开始对{len(symbols_to_scan)}个交易对进行扫描轮次...")
        scan_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        results = []

        # 使用并行处理提高效率 - 增加工作线程数量以加快处理
        with ThreadPoolExecutor(max_workers=10) as executor:
            # 提交所有分析任务
            future_to_symbol = {
                executor.submit(self.analyze_single_coin, symbol): symbol
                for symbol in symbols_to_scan
            }

            # 处理结果，按完成顺序
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    analysis = future.result()

                    # 过滤掉低质量或低变动的交易对
                    if (analysis["status"] == "analyzed" and
                            analysis["expected_movement"] >= min_expected_movement):
                        results.append(analysis)

                except Exception as e:
                    self.logger.error(f"处理{symbol}时出错: {e}")

        # 按质量评分排序（降序）
        results.sort(key=lambda x: x["adjusted_score"], reverse=True)

        self.logger.info(f"扫描轮次完成。找到{len(results)}个预期变动 >= {min_expected_movement}%的交易对")

        # 保存轮次结果
        round_result = {
            "timestamp": scan_time,
            "scanned_symbols": len(symbols_to_scan),
            "qualified_symbols": len(results),
            "min_expected_movement": min_expected_movement,
            "results": results
        }

        self.scan_history.append(round_result)
        return results

    def run_multiple_scan_rounds(self, num_rounds: int = 3,
                                 round_interval: int = 20) -> Dict[str, Any]:
        """
        运行多轮扫描以提高可靠性并跟踪预测准确性

        参数:
            num_rounds: 要运行的扫描轮次数
            round_interval: 轮次间隔时间（秒）(调整为20秒，大大降低总分析时间)

        返回:
            包含聚合扫描结果的字典
        """
        self.logger.info(f"开始多轮扫描分析({num_rounds}轮)...")

        # 获取所有可用的USDT交易对进行扫描 - 最低交易量调整为850万USDT
        all_symbols = self.get_all_usdt_pairs(min_volume=8500000)

        # 如果配置中的交易对尚未包含，则添加
        config_pairs = self.config.get("TRADE_PAIRS", [])
        for symbol in config_pairs:
            if symbol not in all_symbols and symbol.endswith("USDT"):
                all_symbols.append(symbol)

        self.logger.info(f"将在{num_rounds}轮中扫描{len(all_symbols)}个交易对")

        all_round_results = []
        consistent_coins = {}  # 跟踪在多轮中出现的交易对

        # 运行每一轮
        for round_num in range(1, num_rounds + 1):
            self.logger.info(f"开始扫描轮次 {round_num}/{num_rounds}")

            # 运行扫描
            round_results = self.run_scan_round(all_symbols)
            all_round_results.append(round_results)

            # 跟踪具有一致信号的交易对
            for result in round_results:
                symbol = result["symbol"]
                if symbol not in consistent_coins:
                    consistent_coins[symbol] = []

                consistent_coins[symbol].append(result)

            # 保存价格以便后续准确性检查
            for result in round_results:
                symbol = result["symbol"]
                if symbol not in self.prediction_accuracy:
                    self.prediction_accuracy[symbol] = []

                self.prediction_accuracy[symbol].append({
                    "timestamp": time.time(),
                    "current_price": result["current_price"],
                    "predicted_price": result["predicted_price"],
                    "prediction_horizon": 60  # 分钟
                })

            # 在下一轮之前等待（最后一轮除外）
            if round_num < num_rounds:
                self.logger.info(f"等待{round_interval}秒进入下一轮...")
                time.sleep(round_interval)

        # 聚合所有轮次的结果
        aggregated_results = self.aggregate_multi_round_results(consistent_coins, num_rounds)

        # 保存最终结果
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scan_results/multi_round_scan_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(aggregated_results, f, indent=4)

        self.logger.info(f"多轮扫描完成。结果已保存到{filename}")

        # 自动更新config.py中的TRADE_PAIRS
        self.update_config_trade_pairs(aggregated_results["top_coins"])

        return aggregated_results

    def aggregate_multi_round_results(self, consistent_coins: Dict[str, List[Dict]],
                                      num_rounds: int) -> Dict[str, Any]:
        """
        聚合多轮扫描的结果

        参数:
            consistent_coins: 将交易对映射到其在各轮中的结果的字典
            num_rounds: 运行的轮次数

        返回:
            包含聚合结果的字典
        """
        aggregated = {
            "timestamp": time.time(),
            "timestamp_readable": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_rounds": num_rounds,
            "top_coins": [],
            "all_analyzed_coins": len(consistent_coins),
            "consistent_coins": 0
        }

        # 处理每个交易对的结果
        coin_summaries = []

        for symbol, results in consistent_coins.items():
            # 跳过没有在所有轮次中出现的交易对
            if len(results) < num_rounds:
                continue

            aggregated["consistent_coins"] += 1

            # 计算平均分数和变动
            avg_quality_score = sum(r["quality_score"] for r in results) / len(results)
            avg_adjusted_score = sum(r["adjusted_score"] for r in results) / len(results)
            avg_expected_movement = sum(r["expected_movement"] for r in results) / len(results)

            # 检查信号的一致性
            signals = [r["signal"] for r in results]
            buy_signals = signals.count("BUY")
            sell_signals = signals.count("SELL")
            neutral_signals = signals.count("NEUTRAL")

            # 确定主导信号
            if buy_signals > sell_signals and buy_signals > neutral_signals:
                dominant_signal = "BUY"
                signal_consistency = buy_signals / len(signals)
            elif sell_signals > buy_signals and sell_signals > neutral_signals:
                dominant_signal = "SELL"
                signal_consistency = sell_signals / len(signals)
            else:
                dominant_signal = "NEUTRAL"
                signal_consistency = neutral_signals / len(signals)

            # 计算预测价格一致性
            predicted_prices = [r["predicted_price"] for r in results]
            price_std_dev = np.std(predicted_prices)
            price_variance = price_std_dev / np.mean(predicted_prices) if np.mean(predicted_prices) > 0 else 0

            # 最新价格
            latest_price = results[-1]["current_price"]

            # 如果可能，计算预测准确性
            prediction_accuracy = self.calculate_prediction_accuracy(symbol)

            # 创建交易对摘要
            summary = {
                "symbol": symbol,
                "avg_quality_score": avg_quality_score,
                "avg_adjusted_score": avg_adjusted_score,
                "avg_expected_movement": avg_expected_movement,
                "dominant_signal": dominant_signal,
                "signal_consistency": signal_consistency * 100,  # 转为百分比
                "price_consistency": (1 - price_variance) * 100,  # 转为百分比
                "current_price": latest_price,
                "prediction_accuracy": prediction_accuracy,
                "ranking_score": 0  # 将在下面计算
            }

            # 计算最终排名得分（因素的加权组合）
            # 分数越高越好
            ranking_score = (
                    avg_adjusted_score * 0.4 +  # 40% - 质量评分
                    avg_expected_movement * 0.2 +  # 20% - 预期变动
                    signal_consistency * 100 * 0.2 +  # 20% - 信号一致性
                    (1 - price_variance) * 100 * 0.1 +  # 10% - 价格预测一致性
                    (prediction_accuracy if prediction_accuracy > 0 else 50) * 0.1  # 10% - 历史预测准确性
            )

            # 预期变动较大的奖励
            if avg_expected_movement > 3.0:
                ranking_score *= 1.1

            # 买入信号的奖励（按要求）
            if dominant_signal == "BUY":
                ranking_score *= 1.05

            summary["ranking_score"] = ranking_score
            coin_summaries.append(summary)

        # 按排名得分排序
        coin_summaries.sort(key=lambda x: x["ranking_score"], reverse=True)

        # 获取排名靠前的交易对
        aggregated["top_coins"] = coin_summaries[:10]
        aggregated["all_evaluated_coins"] = coin_summaries

        return aggregated

    def calculate_prediction_accuracy(self, symbol: str) -> float:
        """
        如果有数据，计算交易对的历史预测准确性

        参数:
            symbol: 交易对符号

        返回:
            准确性百分比(0-100)，如果没有数据则为0
        """
        if symbol not in self.prediction_accuracy:
            return 0

        predictions = self.prediction_accuracy[symbol]

        # 需要至少两个预测来检查准确性
        if len(predictions) < 2:
            return 0

        # 检查有足够时间成熟的最早预测
        oldest_prediction = predictions[0]
        prediction_time = oldest_prediction["timestamp"]
        current_time = time.time()

        # 将时间范围从分钟转换为秒
        horizon_seconds = oldest_prediction["prediction_horizon"] * 60

        # 检查是否已经过了足够的时间来评估预测
        if current_time - prediction_time < horizon_seconds:
            return 0

        try:
            # 获取当前价格进行比较
            df = self.get_historical_data(symbol, force_refresh=True)
            if df is None or df.empty:
                return 0

            actual_price = df['close'].iloc[-1]
            predicted_price = oldest_prediction["predicted_price"]
            initial_price = oldest_prediction["current_price"]

            # 计算价格变化
            actual_change = (actual_price - initial_price) / initial_price
            predicted_change = (predicted_price - initial_price) / initial_price

            # 如果两个变化方向相同
            if (actual_change >= 0 and predicted_change >= 0) or \
                    (actual_change < 0 and predicted_change < 0):
                # 方向正确，现在检查幅度
                if abs(predicted_change) > 0:
                    magnitude_accuracy = min(abs(actual_change / predicted_change), 1)
                else:
                    magnitude_accuracy = 0

                # 结合方向（占70%）和幅度（占30%）
                accuracy = 70 + (30 * magnitude_accuracy)
            else:
                # 方向错误
                accuracy = 0

            return accuracy

        except Exception as e:
            self.logger.error(f"计算{symbol}的预测准确性时出错: {e}")
            return 0

    def get_recommended_trade_pairs(self) -> List[str]:
        """
        基于最新扫描结果获取推荐的交易对

        返回:
            推荐的交易对符号列表
        """
        if not self.scan_history:
            return []

        # 获取最新的聚合扫描
        latest_scan = self.scan_history[-1]

        # 提取排名前10的交易对
        top_coins = latest_scan.get("results", [])[:10]
        return [coin["symbol"] for coin in top_coins]

    def update_config_trade_pairs(self, top_coins):
        """
        更新config.py中的TRADE_PAIRS列表

        参数:
            top_coins: 排名靠前的交易对列表
        """
        try:
            # 提取推荐的交易对
            recommended_pairs = [coin["symbol"] for coin in top_coins]

            # 备份原始config.py
            if os.path.exists("config.py"):
                shutil.copy2("config.py", f"config_backup_{int(time.time())}.py")

                # 读取当前config.py内容
                with open("config.py", "r") as f:
                    config_content = f.read()

                # 查找TRADE_PAIRS定义
                import re
                trade_pairs_pattern = r"TRADE_PAIRS\s*=\s*\[(.*?)\]"
                trade_pairs_match = re.search(trade_pairs_pattern, config_content, re.DOTALL)

                if trade_pairs_match:
                    # 生成新的TRADE_PAIRS字符串
                    new_trade_pairs = 'TRADE_PAIRS = [\n    "' + '",\n    "'.join(recommended_pairs) + '"\n]'

                    # 替换旧的TRADE_PAIRS
                    updated_content = config_content.replace(
                        config_content[trade_pairs_match.start():trade_pairs_match.end()],
                        new_trade_pairs
                    )

                    # 写入更新后的内容
                    with open("config.py", "w") as f:
                        f.write(updated_content)

                    self.logger.info(f"已成功更新config.py中的TRADE_PAIRS列表，包含{len(recommended_pairs)}个交易对")

                    # 在日志中记录所有推荐的交易对
                    pairs_str = ", ".join(recommended_pairs)
                    self.logger.info(f"推荐的交易对: {pairs_str}")

                    # 将信息同时打印到控制台
                    print(f"\n💰 已成功更新config.py中的TRADE_PAIRS列表，包含{len(recommended_pairs)}个交易对")
                    print(f"📊 推荐的交易对: {pairs_str}")
                else:
                    self.logger.warning("在config.py中未找到TRADE_PAIRS定义，无法更新")
            else:
                self.logger.warning("未找到config.py文件，无法更新TRADE_PAIRS")
        except Exception as e:
            self.logger.error(f"更新config.py时出错: {e}")

    def generate_html_report(self, aggregated_results: Dict[str, Any]) -> str:
        """
        使用扫描结果生成HTML报告

        参数:
            aggregated_results: 聚合的扫描结果

        返回:
            保存的HTML报告的文件名
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scan_results/crypto_scanner_report_{timestamp}.html"

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>加密货币扫描器报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #0066cc; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .buy {{ color: green; font-weight: bold; }}
                .sell {{ color: red; font-weight: bold; }}
                .neutral {{ color: gray; }}
                .high {{ color: green; }}
                .medium {{ color: orange; }}
                .low {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>加密货币扫描器报告</h1>
            <p>生成于 {aggregated_results["timestamp_readable"]}</p>

            <div class="summary">
                <h2>扫描摘要</h2>
                <p>在{aggregated_results["num_rounds"]}轮中分析了{aggregated_results["all_analyzed_coins"]}个交易对。</p>
                <p>找到{aggregated_results["consistent_coins"]}个在所有轮次中有一致信号的交易对。</p>
            </div>

            <h2>推荐的前10个交易对</h2>
            <table>
                <tr>
                    <th>排名</th>
                    <th>交易对</th>
                    <th>信号</th>
                    <th>质量评分</th>
                    <th>预期变动</th>
                    <th>信号一致性</th>
                    <th>预测一致性</th>
                    <th>预测准确性</th>
                    <th>排名得分</th>
                </tr>
        """

        # 将排名靠前的交易对添加到表格
        for i, coin in enumerate(aggregated_results["top_coins"], 1):
            # 确定CSS类用于样式设置
            signal_class = coin["dominant_signal"].lower()

            # 质量评分类
            if coin["avg_quality_score"] >= 7:
                quality_class = "high"
            elif coin["avg_quality_score"] >= 5:
                quality_class = "medium"
            else:
                quality_class = "low"

            # 信号一致性类
            if coin["signal_consistency"] >= 80:
                consistency_class = "high"
            elif coin["signal_consistency"] >= 60:
                consistency_class = "medium"
            else:
                consistency_class = "low"

            html += f"""
                <tr>
                    <td>{i}</td>
                    <td>{coin["symbol"]}</td>
                    <td class="{signal_class}">{coin["dominant_signal"]}</td>
                    <td class="{quality_class}">{coin["avg_quality_score"]:.2f}</td>
                    <td>{coin["avg_expected_movement"]:.2f}%</td>
                    <td class="{consistency_class}">{coin["signal_consistency"]:.1f}%</td>
                    <td>{coin["price_consistency"]:.1f}%</td>
                    <td>{coin["prediction_accuracy"]:.1f}%</td>
                    <td>{coin["ranking_score"]:.2f}</td>
                </tr>
            """

        html += """
            </table>

            <h2>config.py的推荐</h2>
            <p>基于分析，建议使用以下交易对:</p>
            <pre>
        """

        # 生成推荐的TRADE_PAIRS配置
        recommended_pairs = [coin["symbol"] for coin in aggregated_results["top_coins"]]
        trade_pairs_str = 'TRADE_PAIRS = [\n    "' + '",\n    "'.join(recommended_pairs) + '"\n]'
        html += trade_pairs_str

        html += """
            </pre>

            <h3>集成说明</h3>
            <p>要使用这些推荐的交易对:</p>
            <ol>
                <li>将config.py中的TRADE_PAIRS列表替换为上面的列表。</li>
                <li>重启交易机器人以应用更改。</li>
            </ol>
            <p>注意: 此扫描器已自动更新了config.py中的TRADE_PAIRS。</p>

            <h2>所有评估过的交易对</h2>
            <p>下表显示了在扫描轮次中始终出现的所有交易对。</p>
            <table>
                <tr>
                    <th>交易对</th>
                    <th>信号</th>
                    <th>质量评分</th>
                    <th>预期变动</th>
                    <th>信号一致性</th>
                    <th>排名得分</th>
                </tr>
        """

        # 添加所有评估过的交易对
        for coin in aggregated_results["all_evaluated_coins"]:
            signal_class = coin["dominant_signal"].lower()

            html += f"""
                <tr>
                    <td>{coin["symbol"]}</td>
                    <td class="{signal_class}">{coin["dominant_signal"]}</td>
                    <td>{coin["avg_quality_score"]:.2f}</td>
                    <td>{coin["avg_expected_movement"]:.2f}%</td>
                    <td>{coin["signal_consistency"]:.1f}%</td>
                    <td>{coin["ranking_score"]:.2f}</td>
                </tr>
            """

        html += """
            </table>
        </body>
        </html>
        """

        # 将HTML写入文件
        with open(filename, 'w') as f:
            f.write(html)

        self.logger.info(f"HTML报告已生成: {filename}")
        return filename


def main():
    """运行扫描器的主函数"""
    # 使用直接内置的API密钥，无需命令行参数
    api_key = API_KEY
    api_secret = API_SECRET


    # 初始化扫描器
    try:
        from config import CONFIG
        scanner = CryptoCurrencyScanner(api_key, api_secret, CONFIG)
        print("已使用当前config.py中的配置初始化扫描器")
    except Exception as e:
        print(f"使用config.py失败: {e}")
        scanner = CryptoCurrencyScanner(api_key, api_secret)
        print("已使用默认配置初始化扫描器")

    # 扫描参数
    min_volume = 8500000  # 850万USDT的最低24小时交易量
    rounds = 3  # 扫描轮次数
    round_interval = 20  # 轮次间隔（秒）(调整为20秒，大大降低总分析时间)
    min_movement = 1.7  # 最低预期价格变动百分比

    print(f"启动加密货币扫描器，运行{rounds}轮...")
    print(f"最低交易量: {min_volume} USDT")
    print(f"最低预期变动: {min_movement}%")
    print(f"轮次间隔: {round_interval}秒")

    # 扫描前清理缓存
    scanner.clean_cache()
    print("缓存已清理，准备开始扫描")

    # 运行多轮扫描
    aggregated_results = scanner.run_multiple_scan_rounds(
        num_rounds=rounds,
        round_interval=round_interval
    )

    # 生成HTML报告
    report_file = scanner.generate_html_report(aggregated_results)

    print("\n=== 扫描完成 ===")
    print(f"推荐的交易对:")
    for i, coin in enumerate(aggregated_results["top_coins"], 1):
        signal_icon = "📈" if coin['dominant_signal'] == "BUY" else "📉" if coin['dominant_signal'] == "SELL" else "⚖️"
        print(
            f"{i}. {signal_icon} {coin['symbol']} - {coin['dominant_signal']} - 预期变动: {coin['avg_expected_movement']:.2f}% - 评分: {coin['ranking_score']:.2f}")

    print(f"\n详细报告已保存到: {report_file}")
    print("\nconfig.py中的TRADE_PAIRS已自动更新，重启交易机器人即可应用新的交易对")
    print("\n如需在止盈止损后再次运行筛选器，只需执行以下命令:")
    print("python crypto_scanner.py")


if __name__ == "__main__":
    main()