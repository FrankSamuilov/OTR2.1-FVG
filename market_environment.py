import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union

class MarketEnvironmentClassifier:
    """
    增强版市场环境检测与动荡过滤器
    结合市场环境分类与动荡检测功能，提供更全面的市场分析
    """

    import pandas as pd
    import numpy as np
    from typing import Dict, Tuple, Any, Optional
    from logger_utils import Colors, print_colored

    class EnhancedMarketDetector:
        """结合市场环境分类与动荡检测的增强版市场分析器"""

        def __init__(self, parent_bot=None):
            """
            初始化增强版市场分析器

            参数:
                parent_bot: 父级交易机器人对象
            """
            self.parent_bot = parent_bot
            self.volatility_history = {}  # 保存历史波动性检测结果
            self.environment_history = {}  # 保存历史环境分类结果
            self.market_classifier = MarketEnvironmentClassifier()  # 市场环境分类器
            self.dynamic_tp_sl_enabled = True  # 启用动态止盈止损

            # 波动性检测参数
            self.volatility_threshold = 0.4  # 波动性评分阈值，超过视为动荡市场

            print_colored("✅ 增强版市场环境检测与动荡过滤器初始化完成", Colors.GREEN)

        def classify_environment(self, df: pd.DataFrame) -> Dict[str, Any]:
            """
            分析并分类市场环境

            参数:
                df: 价格数据DataFrame

            返回:
                Dict: 包含环境分类和详细信息的字典
            """
            try:
                # 默认结果
                result = {
                    "environment": "unknown",
                    "confidence": 0.0,
                    "details": {}
                }

                if df is None or len(df) < 20:
                    print_colored("⚠️ 数据不足，无法分析市场环境", Colors.WARNING)
                    return result

                # 计算基本指标
                if 'ATR' not in df.columns and len(df) >= 14:
                    # 计算ATR
                    tr1 = abs(df['high'] - df['low'])
                    tr2 = abs(df['high'] - df['close'].shift(1))
                    tr3 = abs(df['low'] - df['close'].shift(1))
                    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
                    df['ATR'] = tr.rolling(window=14).mean()

                # 1. 趋势市场检测
                is_trending = False
                trend_direction = "neutral"
                trend_confidence = 0.0

                # 检查价格是否形成更高高点/更低低点
                if len(df) >= 10:
                    highs = df['high'].rolling(window=5).max()
                    lows = df['low'].rolling(window=5).min()

                    higher_highs = df['high'].iloc[-1] > highs.iloc[-5]
                    higher_lows = df['low'].iloc[-1] > lows.iloc[-5]
                    lower_highs = df['high'].iloc[-1] < highs.iloc[-5]
                    lower_lows = df['low'].iloc[-1] < lows.iloc[-5]

                    if higher_highs and higher_lows:
                        is_trending = True
                        trend_direction = "uptrend"
                        trend_confidence = 0.7
                    elif lower_highs and lower_lows:
                        is_trending = True
                        trend_direction = "downtrend"
                        trend_confidence = 0.7

                # 检查ADX (如果可用)
                if 'ADX' in df.columns:
                    adx = df['ADX'].iloc[-1]
                    trend_strength = 0.0

                    if adx > 25:  # 强趋势
                        is_trending = True
                        trend_strength = min(1.0, adx / 50)  # 归一化到0-1
                        trend_confidence = max(trend_confidence, trend_strength)

                        # 检查趋势方向
                        if 'Plus_DI' in df.columns and 'Minus_DI' in df.columns:
                            if df['Plus_DI'].iloc[-1] > df['Minus_DI'].iloc[-1]:
                                trend_direction = "uptrend"
                            else:
                                trend_direction = "downtrend"

                    result["details"]["adx"] = float(adx)
                    result["details"]["trend_strength"] = float(trend_strength)

                # 2. 区间震荡市场检测
                is_ranging = False
                range_confidence = 0.0

                # 检查价格是否在一定范围内波动
                if len(df) >= 20:
                    price_range = (df['high'].max() - df['low'].min()) / df['close'].mean()
                    recent_range = (df['high'].iloc[-10:].max() - df['low'].iloc[-10:].min()) / df['close'].iloc[-1]

                    if price_range < 0.1:  # 10%范围内
                        is_ranging = True
                        range_confidence = 0.6

                    # 检查波动性是否低
                    if 'ATR' in df.columns:
                        atr_ratio = df['ATR'].iloc[-1] / df['ATR'].rolling(window=20).mean().iloc[-1]
                        if atr_ratio < 0.8:  # 低波动性
                            is_ranging = True
                            range_confidence = max(range_confidence, 0.7)

                    result["details"]["price_range"] = float(price_range)
                    result["details"]["recent_range"] = float(recent_range)

                # 3. 突破检测
                is_breakout = False
                breakout_direction = "unknown"
                breakout_confidence = 0.0

                # 检查布林带突破
                if all(col in df.columns for col in ['BB_Upper', 'BB_Lower']):
                    price = df['close'].iloc[-1]
                    prev_price = df['close'].iloc[-2]

                    # 检查是否突破布林带
                    if price > df['BB_Upper'].iloc[-1] and prev_price <= df['BB_Upper'].iloc[-2]:
                        is_breakout = True
                        breakout_direction = "upward"
                        breakout_confidence = 0.8
                    elif price < df['BB_Lower'].iloc[-1] and prev_price >= df['BB_Lower'].iloc[-2]:
                        is_breakout = True
                        breakout_direction = "downward"
                        breakout_confidence = 0.8

                    result["details"]["bb_position"] = (price - df['BB_Lower'].iloc[-1]) / (
                                df['BB_Upper'].iloc[-1] - df['BB_Lower'].iloc[-1])

                # 4. 极端波动市场检测
                is_extreme_volatility = False
                volatility_confidence = 0.0

                if 'ATR' in df.columns:
                    atr_ratio = df['ATR'].iloc[-1] / df['ATR'].rolling(window=20).mean().iloc[-1]
                    if atr_ratio > 2.0:  # 波动性是平均的2倍以上
                        is_extreme_volatility = True
                        volatility_confidence = min(1.0, atr_ratio / 3)

                    result["details"]["atr_ratio"] = float(atr_ratio)

                # 确定最终市场环境
                environments = [
                    ("trending", is_trending, trend_confidence),
                    ("ranging", is_ranging, range_confidence),
                    ("breakout", is_breakout, breakout_confidence),
                    ("extreme_volatility", is_extreme_volatility, volatility_confidence)
                ]

                # 选择置信度最高的环境类型
                best_env = max(environments, key=lambda x: x[2])
                result["environment"] = best_env[0]
                result["confidence"] = best_env[2] * 100  # 转换为百分比

                # 添加方向信息
                if result["environment"] == "trending":
                    result["details"]["trend_direction"] = trend_direction
                elif result["environment"] == "breakout":
                    result["details"]["breakout_direction"] = breakout_direction

                # 打印分析结果
                env_color = (
                    Colors.GREEN if result["environment"] == "trending" and trend_direction == "uptrend" else
                    Colors.RED if result["environment"] == "trending" and trend_direction == "downtrend" else
                    Colors.YELLOW if result["environment"] == "ranging" else
                    Colors.CYAN if result["environment"] == "breakout" else
                    Colors.RED + Colors.BOLD if result["environment"] == "extreme_volatility" else
                    Colors.GRAY
                )

                print_colored(
                    f"市场环境分析: {env_color}{result['environment']}{Colors.RESET}, "
                    f"置信度: {result['confidence']:.1f}%",
                    Colors.INFO
                )

                if result["environment"] == "trending":
                    dir_color = Colors.GREEN if trend_direction == "uptrend" else Colors.RED
                    print_colored(f"趋势方向: {dir_color}{trend_direction}{Colors.RESET}", Colors.INFO)
                elif result["environment"] == "breakout":
                    dir_color = Colors.GREEN if breakout_direction == "upward" else Colors.RED
                    print_colored(f"突破方向: {dir_color}{breakout_direction}{Colors.RESET}", Colors.INFO)

                return result

            except Exception as e:
                print_colored(f"❌ 市场环境分类失败: {e}", Colors.ERROR)
                return {
                    "environment": "unknown",
                    "confidence": 0.0,
                    "details": {"error": str(e)}
                }

        def get_optimal_strategy_params(self, env_result: Dict[str, Any]) -> Dict[str, Any]:
            """
            根据市场环境获取最优交易策略参数

            参数:
                env_result: 环境分类结果

            返回:
                Dict: 包含最优交易参数的字典
            """
            result = {
                "entry_type": "market",
                "position_size": 1.0,  # 默认标准仓位
                "take_profit_pct": 0.025,  # 默认2.5%止盈
                "stop_loss_pct": 0.020,  # 默认2.0%止损
                "trailing_stop": False,
                "trailing_callback": 0.01  # 默认1%回调
            }

            environment = env_result["environment"]
            confidence = env_result["confidence"] / 100  # 转换为0-1

            if environment == "trending":
                result["entry_type"] = "market"
                result["take_profit_pct"] = 0.04  # 4%止盈
                result["stop_loss_pct"] = 0.025  # 2.5%止损
                result["trailing_stop"] = True
                result["trailing_callback"] = 0.015  # 1.5%回调

                # 根据趋势方向调整
                trend_dir = env_result["details"].get("trend_direction", "neutral")
                if trend_dir == "uptrend":
                    result["bias"] = "long"
                elif trend_dir == "downtrend":
                    result["bias"] = "short"

            elif environment == "ranging":
                result["entry_type"] = "limit"  # 限价单
                result["take_profit_pct"] = 0.02  # 2%止盈
                result["stop_loss_pct"] = 0.015  # 1.5%止损
                result["trailing_stop"] = False
                result["position_size"] = 0.8  # 80%标准仓位

            elif environment == "breakout":
                result["entry_type"] = "market"
                result["take_profit_pct"] = 0.035  # 3.5%止盈
                result["stop_loss_pct"] = 0.02  # 2%止损
                result["trailing_stop"] = True
                result["trailing_callback"] = 0.015  # 1.5%回调

                # 根据突破方向调整
                breakout_dir = env_result["details"].get("breakout_direction", "unknown")
                if breakout_dir == "upward":
                    result["bias"] = "long"
                elif breakout_dir == "downward":
                    result["bias"] = "short"

            elif environment == "extreme_volatility":
                result["entry_type"] = "market"
                result["take_profit_pct"] = 0.05  # 5%止盈
                result["stop_loss_pct"] = 0.03  # 3%止损
                result["trailing_stop"] = True
                result["trailing_callback"] = 0.02  # 2%回调
                result["position_size"] = 0.6  # 60%标准仓位

            return result

        def detect_market_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
            """
            使用多指标组合检测市场是否处于动荡期

            参数:
                df: 包含技术指标的DataFrame

            返回:
                Dict: 包含市场状态和详细分析的字典
            """
            print_colored("🔍 开始检测市场动荡状态...", Colors.BLUE)

            # 初始化结果字典
            result = {
                "is_volatile": False,
                "volatility_score": 0.0,
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "indicators": {},
                "recommendation": "WAIT"
            }

            try:
                # 确保数据充分
                if df is None or df.empty or len(df) < 30:
                    print_colored("⚠️ 数据不足，无法分析市场状态", Colors.WARNING)
                    return result

                # 指标评分初始化
                volatility_scores = {}
                direction_votes = {"UP": 0, "DOWN": 0, "NEUTRAL": 0}

                # 1. 布林带宽度分析 - 检测市场波动性
                if all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
                    # 计算布林带宽度
                    bb_width = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
                    current_bb_width = bb_width.iloc[-1]
                    avg_bb_width = bb_width.rolling(20).mean().iloc[-1]
                    bb_width_ratio = current_bb_width / avg_bb_width if avg_bb_width > 0 else 1.0

                    # 布林带宽度异常扩大表示波动性增加
                    if current_bb_width > 0.05:  # 宽度大于5%
                        bb_vol_score = min(1.0, current_bb_width * 10)  # 最高1分
                        volatility_scores["bollinger_width"] = bb_vol_score

                        print_colored(
                            f"布林带宽度: {current_bb_width:.4f} (均值比: {bb_width_ratio:.2f}), 波动评分: {bb_vol_score:.2f}",
                            Colors.WARNING if bb_vol_score > 0.5 else Colors.INFO
                        )

                        # 判断价格位置，推断方向
                        if df['close'].iloc[-1] > df['BB_Upper'].iloc[-1]:
                            direction_votes["UP"] += 1
                        elif df['close'].iloc[-1] < df['BB_Lower'].iloc[-1]:
                            direction_votes["DOWN"] += 1
                    else:
                        volatility_scores["bollinger_width"] = 0.0
                        print_colored(f"布林带宽度: {current_bb_width:.4f}, 波动性正常", Colors.GREEN)

                    # 检查价格是否在布林带通道外
                    price_outside_bb = (
                            (df['close'].iloc[-1] > df['BB_Upper'].iloc[-1]) or
                            (df['close'].iloc[-1] < df['BB_Lower'].iloc[-1])
                    )

                    if price_outside_bb:
                        volatility_scores["price_outside_bb"] = 0.5
                        print_colored("⚠️ 价格位于布林带通道外，表明波动增加", Colors.WARNING)

                        # 判断出轨方向
                        if df['close'].iloc[-1] > df['BB_Upper'].iloc[-1]:
                            direction_votes["UP"] += 1
                            print_colored("价格位于布林带上轨之上，趋势向上", Colors.GREEN)
                        else:
                            direction_votes["DOWN"] += 1
                            print_colored("价格位于布林带下轨之下，趋势向下", Colors.RED)

                # 2. RSI波动分析
                if 'RSI' in df.columns:
                    rsi = df['RSI'].iloc[-1]
                    rsi_prev = df['RSI'].iloc[-5] if len(df) > 5 else 50
                    rsi_change = abs(rsi - rsi_prev)

                    # RSI剧烈变化表示波动增加
                    if rsi_change > 15:
                        rsi_vol_score = min(1.0, rsi_change / 30)
                        volatility_scores["rsi_change"] = rsi_vol_score
                        print_colored(
                            f"RSI变化: {rsi_change:.2f} (从 {rsi_prev:.2f} 到 {rsi:.2f}), 波动评分: {rsi_vol_score:.2f}",
                            Colors.WARNING
                        )
                    else:
                        volatility_scores["rsi_change"] = 0.0

                    # 极端RSI值表示市场可能过度
                    if rsi > 75 or rsi < 25:
                        ext_rsi_score = min(1.0, (abs(rsi - 50) - 25) / 25)
                        volatility_scores["extreme_rsi"] = ext_rsi_score

                        if rsi > 75:
                            direction_votes["UP"] += 1
                            print_colored(f"RSI处于超买区域: {rsi:.2f}, 可能反转向下", Colors.RED)
                        else:
                            direction_votes["DOWN"] += 1
                            print_colored(f"RSI处于超卖区域: {rsi:.2f}, 可能反转向上", Colors.GREEN)

                    # 记录RSI方向
                    if rsi > 60:
                        direction_votes["UP"] += 0.5
                    elif rsi < 40:
                        direction_votes["DOWN"] += 0.5

                # 3. ADX分析 - 趋势强度
                if 'ADX' in df.columns:
                    adx = df['ADX'].iloc[-1]

                    # ADX低表示无明确趋势，可能是震荡市场
                    if adx < 20:
                        adx_vol_score = max(0.0, (20 - adx) / 20)
                        volatility_scores["low_adx"] = adx_vol_score
                        direction_votes["NEUTRAL"] += 1
                        print_colored(f"ADX低: {adx:.2f} < 20, 表明无明确趋势，波动评分: {adx_vol_score:.2f}",
                                      Colors.YELLOW)
                    elif adx > 40:
                        # 强趋势可能不是动荡期
                        volatility_scores["high_adx"] = -0.5  # 负分，减少动荡概率

                        # 检查DI+和DI-来确定方向
                        if 'Plus_DI' in df.columns and 'Minus_DI' in df.columns:
                            if df['Plus_DI'].iloc[-1] > df['Minus_DI'].iloc[-1]:
                                direction_votes["UP"] += 2  # 强烈上升趋势
                                print_colored(f"ADX高: {adx:.2f} > 40, 强烈上升趋势", Colors.GREEN + Colors.BOLD)
                            else:
                                direction_votes["DOWN"] += 2  # 强烈下降趋势
                                print_colored(f"ADX高: {adx:.2f} > 40, 强烈下降趋势", Colors.RED + Colors.BOLD)

                # 4. CCI分析 - 商品通道指数
                if 'CCI' in df.columns:
                    cci = df['CCI'].iloc[-1]
                    cci_prev = df['CCI'].iloc[-5] if len(df) > 5 else 0
                    cci_change = abs(cci - cci_prev)

                    # CCI急剧变化表示波动
                    if cci_change > 100:
                        cci_vol_score = min(1.0, cci_change / 200)
                        volatility_scores["cci_change"] = cci_vol_score
                        print_colored(f"CCI急剧变化: {cci_change:.2f}, 波动评分: {cci_vol_score:.2f}", Colors.WARNING)

                    # 极端CCI值表示可能过度
                    if abs(cci) > 200:
                        ext_cci_score = min(1.0, (abs(cci) - 100) / 200)
                        volatility_scores["extreme_cci"] = ext_cci_score

                        if cci > 200:
                            direction_votes["UP"] += 0.5
                            print_colored(f"CCI极高: {cci:.2f}, 可能过度买入", Colors.RED)
                        elif cci < -200:
                            direction_votes["DOWN"] += 0.5
                            print_colored(f"CCI极低: {cci:.2f}, 可能过度卖出", Colors.GREEN)

                # 5. 随机指标(Stochastic)分析
                if all(col in df.columns for col in ['Stochastic_K', 'Stochastic_D']):
                    k = df['Stochastic_K'].iloc[-1]
                    d = df['Stochastic_D'].iloc[-1]

                    # 检查超买超卖
                    if (k > 80 and d > 80) or (k < 20 and d < 20):
                        stoch_vol_score = 0.7
                        volatility_scores["stochastic_extreme"] = stoch_vol_score

                        if k > 80 and d > 80:
                            direction_votes["DOWN"] += 0.5  # 可能即将反转向下
                            print_colored(f"随机指标超买: K:{k:.2f}, D:{d:.2f}, 可能反转向下", Colors.RED)
                        else:
                            direction_votes["UP"] += 0.5  # 可能即将反转向上
                            print_colored(f"随机指标超卖: K:{k:.2f}, D:{d:.2f}, 可能反转向上", Colors.GREEN)

                    # 检查随机指标交叉
                    if 'Stochastic_Cross_Up' in df.columns and df['Stochastic_Cross_Up'].iloc[-1] == 1:
                        direction_votes["UP"] += 1
                        print_colored("随机指标金叉，趋势向上", Colors.GREEN)
                    elif 'Stochastic_Cross_Down' in df.columns and df['Stochastic_Cross_Down'].iloc[-1] == 1:
                        direction_votes["DOWN"] += 1
                        print_colored("随机指标死叉，趋势向下", Colors.RED)

                # 6. ATR波动性分析
                if 'ATR' in df.columns:
                    atr = df['ATR'].iloc[-1]
                    avg_atr = df['ATR'].rolling(14).mean().iloc[-1]
                    atr_ratio = atr / avg_atr if avg_atr > 0 else 1.0

                    if atr_ratio > 1.5:
                        atr_vol_score = min(1.0, (atr_ratio - 1) * 0.7)
                        volatility_scores["high_atr"] = atr_vol_score
                        print_colored(f"ATR比率高: {atr_ratio:.2f}倍, 波动评分: {atr_vol_score:.2f}", Colors.WARNING)
                    else:
                        print_colored(f"ATR比率: {atr_ratio:.2f}倍, 波动性正常", Colors.GREEN)

                # 综合评分
                if volatility_scores:
                    volatility_score = sum(volatility_scores.values()) / max(1, len(volatility_scores))
                    # 如果有高ADX，降低波动评分
                    if "high_adx" in volatility_scores:
                        volatility_score = max(0, volatility_score * 0.7)

                    result["volatility_score"] = volatility_score
                    result["is_volatile"] = volatility_score > self.volatility_threshold  # 使用阈值判断
                    result["indicators"] = volatility_scores

                # 确定方向
                if direction_votes:
                    max_direction = max(direction_votes.items(), key=lambda x: x[1])
                    direction = max_direction[0]
                    # 计算置信度 - 最高票数除以总票数
                    total_votes = sum(direction_votes.values())
                    confidence = max_direction[1] / total_votes if total_votes > 0 else 0

                    result["direction"] = direction
                    result["confidence"] = confidence
                    result["direction_votes"] = direction_votes

                # 根据波动性和方向给出建议
                if result["is_volatile"]:
                    if result["direction"] == "UP" and result["confidence"] > 0.6:
                        result["recommendation"] = "BUY"
                        print_colored(f"⚠️ 检测到动荡市场，但趋势明确向上 (置信度: {confidence:.2f})", Colors.GREEN)
                    elif result["direction"] == "DOWN" and result["confidence"] > 0.6:
                        result["recommendation"] = "SELL"
                        print_colored(f"⚠️ 检测到动荡市场，但趋势明确向下 (置信度: {confidence:.2f})", Colors.RED)
                    else:
                        result["recommendation"] = "WAIT"
                        print_colored(f"⚠️ 检测到动荡市场，趋势不明确，建议观望", Colors.YELLOW)
                else:
                    result["recommendation"] = "NORMAL"
                    print_colored("✅ 市场波动性正常，使用标准策略", Colors.GREEN)

                print_colored(f"波动评分: {result['volatility_score']:.2f}/1.00",
                              Colors.RED if result["is_volatile"] else Colors.GREEN)

                return result

            except Exception as e:
                print_colored(f"❌ 检测市场动荡状态出错: {e}", Colors.ERROR)
                return result

        def analyze_market_environment(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
            """
            综合分析市场环境，结合动荡检测和环境分类

            参数:
                df: 价格数据DataFrame
                symbol: 交易对符号

            返回:
                Dict: 包含市场环境分析的详细信息
            """
            print_colored(f"\n===== {symbol} 市场环境综合分析 =====", Colors.BLUE + Colors.BOLD)

            try:
                # 初始化结果
                result = {
                    "market_state": "UNKNOWN",
                    "environment": "unknown",
                    "is_volatile": False,
                    "volatility_score": 0.0,
                    "direction": "NEUTRAL",
                    "confidence": 0.0,
                    "optimal_strategy": {}
                }

                # 1. 进行动荡性检测
                volatility_result = self.detect_market_volatility(df)
                result.update(volatility_result)

                # 2. 进行市场环境分类
                env_result = self.market_classifier.classify_environment(df)

                # 记录环境分类结果
                result["environment"] = env_result["environment"]
                result["env_confidence"] = env_result["confidence"]
                result["env_details"] = env_result["details"]

                # 3. 获取最优策略参数
                strategy_params = self.market_classifier.get_optimal_strategy_params(env_result)
                result["optimal_strategy"] = strategy_params

                # 4. 整合动荡分析和环境分类，得出最终的市场状态
                # 如果市场动荡，优先使用动荡检测结果
                if result["is_volatile"]:
                    result["market_state"] = "VOLATILE"

                    # 只有高可信度的方向建议才考虑
                    if result["confidence"] > 0.6:
                        if result["direction"] == "UP":
                            result["trading_bias"] = "LONG"
                            result["strategy"] = "CAUTIOUS_LONG"
                        elif result["direction"] == "DOWN":
                            result["trading_bias"] = "SHORT"
                            result["strategy"] = "CAUTIOUS_SHORT"
                        else:
                            result["trading_bias"] = "NEUTRAL"
                            result["strategy"] = "WAIT"
                    else:
                        result["trading_bias"] = "NEUTRAL"
                        result["strategy"] = "WAIT"
                else:
                    # 非动荡市场，使用环境分类结果
                    result["market_state"] = env_result["environment"].upper()

                    # 根据环境设置交易偏好
                    if env_result["environment"] == "trending":
                        trend_dir = env_result["details"].get("trend_direction", "neutral")
                        if trend_dir == "uptrend":
                            result["trading_bias"] = "LONG"
                            result["strategy"] = "TREND_FOLLOWING_LONG"
                        elif trend_dir == "downtrend":
                            result["trading_bias"] = "SHORT"
                            result["strategy"] = "TREND_FOLLOWING_SHORT"
                        else:
                            result["trading_bias"] = "NEUTRAL"
                            result["strategy"] = "STANDARD"

                    elif env_result["environment"] == "ranging":
                        result["trading_bias"] = "NEUTRAL"
                        result["strategy"] = "RANGE_TRADING"

                    elif env_result["environment"] == "breakout":
                        breakout_dir = env_result["details"].get("breakout_direction", "unknown")
                        if breakout_dir == "upward":
                            result["trading_bias"] = "LONG"
                            result["strategy"] = "BREAKOUT_LONG"
                        elif breakout_dir == "downward":
                            result["trading_bias"] = "SHORT"
                            result["strategy"] = "BREAKOUT_SHORT"
                        else:
                            result["trading_bias"] = "NEUTRAL"
                            result["strategy"] = "STANDARD"

                    elif env_result["environment"] == "extreme_volatility":
                        result["trading_bias"] = "NEUTRAL"
                        result["strategy"] = "MINIMAL_EXPOSURE"
                        result["is_volatile"] = True  # 更新波动状态

                # 打印市场环境分析结果
                print_colored(f"\n----- {symbol} 市场环境分析结果 -----", Colors.BLUE)

                state_color = (
                    Colors.RED if result["market_state"] == "VOLATILE" or result[
                        "market_state"] == "EXTREME_VOLATILITY" else
                    Colors.GREEN if result["market_state"] == "TRENDING" else
                    Colors.YELLOW if result["market_state"] == "RANGING" else
                    Colors.CYAN if result["market_state"] == "BREAKOUT" else
                    Colors.GRAY
                )

                print_colored(f"市场状态: {state_color}{result['market_state']}{Colors.RESET}", Colors.BOLD)
                print_colored(f"动荡评分: {result['volatility_score']:.2f}/1.00", Colors.INFO)
                print_colored(f"环境置信度: {result['env_confidence']:.2f}/100", Colors.INFO)

                bias_color = (
                    Colors.GREEN if result["trading_bias"] == "LONG" else
                    Colors.RED if result["trading_bias"] == "SHORT" else
                    Colors.GRAY
                )

                print_colored(f"交易偏好: {bias_color}{result['trading_bias']}{Colors.RESET}", Colors.BOLD)
                print_colored(f"建议策略: {result['strategy']}", Colors.BOLD)

                # 打印策略参数
                print_colored("\n推荐交易参数:", Colors.BLUE)
                for key, value in result["optimal_strategy"].items():
                    print_colored(f"  - {key}: {value}", Colors.INFO)

                # 更新环境历史
                self.environment_history[symbol] = {
                    "timestamp": pd.Timestamp.now(),
                    "market_state": result["market_state"],
                    "trading_bias": result["trading_bias"],
                    "volatility_score": result["volatility_score"]
                }

                return result

            except Exception as e:
                print_colored(f"❌ 市场环境分析失败: {e}", Colors.ERROR)
                return {
                    "market_state": "ERROR",
                    "error": str(e)
                }

        def ema_slope_trend_filter(self, df: pd.DataFrame, short_period: int = 5, long_period: int = 20) -> Dict[
            str, Any]:
            """
            使用EMA和斜率分析来寻找趋势方向，用于动荡市场的信号过滤

            参数:
                df: 价格数据DataFrame
                short_period: 短期EMA周期
                long_period: 长期EMA周期

            返回:
                Dict: 包含趋势方向和置信度的字典
            """
            print_colored(f"🔍 使用EMA+斜率进行趋势方向过滤...", Colors.BLUE)

            result = {
                "trend": "NEUTRAL",
                "slope_direction": "NEUTRAL",
                "confidence": 0.0,
                "ema_alignment": False,
                "signal": "WAIT"
            }

            try:
                # 确保数据充分
                if df is None or df.empty or len(df) < long_period + 5:
                    print_colored("⚠️ 数据不足，无法进行EMA斜率分析", Colors.WARNING)
                    return result

                # 计算EMA (如果不存在)
                ema_short_col = f'EMA{short_period}'
                ema_long_col = f'EMA{long_period}'

                if ema_short_col not in df.columns:
                    df[ema_short_col] = df['close'].ewm(span=short_period, adjust=False).mean()

                if ema_long_col not in df.columns:
                    df[ema_long_col] = df['close'].ewm(span=long_period, adjust=False).mean()

                # 获取最近的EMA值
                ema_short = df[ema_short_col].iloc[-10:].values
                ema_long = df[ema_long_col].iloc[-10:].values

                # 计算EMA斜率
                short_slope = np.polyfit(range(len(ema_short)), ema_short, 1)[0]
                long_slope = np.polyfit(range(len(ema_long)), ema_long, 1)[0]

                # 归一化斜率为百分比变化率
                short_slope_pct = short_slope / ema_short[-1] * 100
                long_slope_pct = long_slope / ema_long[-1] * 100

                print_colored(f"短期EMA斜率: {short_slope_pct:.4f}% / 周期",
                              Colors.GREEN if short_slope_pct > 0 else Colors.RED)
                print_colored(f"长期EMA斜率: {long_slope_pct:.4f}% / 周期",
                              Colors.GREEN if long_slope_pct > 0 else Colors.RED)

                # EMA交叉状态
                ema_cross_up = df[ema_short_col].iloc[-1] > df[ema_long_col].iloc[-1] and \
                               df[ema_short_col].iloc[-2] <= df[ema_long_col].iloc[-2]

                ema_cross_down = df[ema_short_col].iloc[-1] < df[ema_long_col].iloc[-1] and \
                                 df[ema_short_col].iloc[-2] >= df[ema_long_col].iloc[-2]

                # 检查EMA方向一致性
                ema_aligned = (short_slope_pct > 0 and long_slope_pct > 0) or \
                              (short_slope_pct < 0 and long_slope_pct < 0)

                # 记录结果
                result["ema_alignment"] = ema_aligned
                result["short_slope"] = short_slope_pct
                result["long_slope"] = long_slope_pct
                result["ema_cross_up"] = ema_cross_up
                result["ema_cross_down"] = ema_cross_down

                # 确定斜率方向
                if short_slope_pct > 0.02:  # 明显向上
                    result["slope_direction"] = "UP"
                    slope_confidence = min(1.0, short_slope_pct / 0.1)  # 最大置信度为1.0
                elif short_slope_pct < -0.02:  # 明显向下
                    result["slope_direction"] = "DOWN"
                    slope_confidence = min(1.0, abs(short_slope_pct) / 0.1)
                else:
                    result["slope_direction"] = "NEUTRAL"
                    slope_confidence = 0.3

                # 综合分析趋势方向
                if ema_cross_up or (df[ema_short_col].iloc[-1] > df[ema_long_col].iloc[-1] and short_slope_pct > 0):
                    result["trend"] = "UP"
                    confidence = 0.7

                    # 增强因素
                    if ema_aligned and short_slope_pct > 0:
                        confidence += 0.2
                    if ema_cross_up:
                        confidence += 0.1
                        print_colored("检测到EMA金叉，看涨信号增强", Colors.GREEN + Colors.BOLD)

                    result["confidence"] = min(1.0, confidence)
                    result["signal"] = "BUY"

                elif ema_cross_down or (df[ema_short_col].iloc[-1] < df[ema_long_col].iloc[-1] and short_slope_pct < 0):
                    result["trend"] = "DOWN"
                    confidence = 0.7

                    # 增强因素
                    if ema_aligned and short_slope_pct < 0:
                        confidence += 0.2
                    if ema_cross_down:
                        confidence += 0.1
                        print_colored("检测到EMA死叉，看跌信号增强", Colors.RED + Colors.BOLD)

                    result["confidence"] = min(1.0, confidence)
                    result["signal"] = "SELL"

                else:
                    # 当EMA关系不明确时，使用斜率方向
                    result["trend"] = result["slope_direction"]
                    result["confidence"] = slope_confidence

                    if result["trend"] == "UP" and result["confidence"] > 0.5:
                        result["signal"] = "BUY"
                    elif result["trend"] == "DOWN" and result["confidence"] > 0.5:
                        result["signal"] = "SELL"
                    else:
                        result["signal"] = "WAIT"

                # 输出分析结果
                trend_color = Colors.GREEN if result["trend"] == "UP" else Colors.RED if result[
                                                                                             "trend"] == "DOWN" else Colors.GRAY
                print_colored(
                    f"EMA+斜率分析结果: {trend_color}{result['trend']}{Colors.RESET}, "
                    f"置信度: {result['confidence']:.2f}, "
                    f"信号: {result['signal']}",
                    Colors.BOLD
                )

                if ema_aligned:
                    print_colored("✅ 短期和长期EMA方向一致，趋势可靠性提高", Colors.GREEN)
                else:
                    print_colored("⚠️ 短期和长期EMA方向不一致，趋势可能存在冲突", Colors.YELLOW)

                return result

            except Exception as e:
                print_colored(f"❌ EMA斜率分析出错: {e}", Colors.ERROR)
                return result

        def generate_filtered_signal(self, df: pd.DataFrame, symbol: str, original_signal: str,
                                     quality_score: float) -> Dict[str, Any]:
            """
            综合市场环境分析和EMA斜率过滤，生成最终交易信号

            参数:
                df: 价格数据DataFrame
                symbol: 交易对符号
                original_signal: 原始交易信号
                quality_score: 原始质量评分

            返回:
                Dict: 包含过滤后的信号和相关信息
            """
            print_colored("🔄 开始生成环境适应型交易信号...", Colors.BLUE)

            result = {
                "original_signal": original_signal,
                "filtered_signal": original_signal,  # 默认保持原始信号
                "original_quality": quality_score,
                "adjusted_quality": quality_score,
                "market_state": "NORMAL",
                "reason": "保持原始信号",
                "strategy_params": {}
            }

            try:
                # 1. 进行市场环境综合分析
                env_analysis = self.analyze_market_environment(df, symbol)

                # 2. 根据市场状态调整信号和策略
                result["market_state"] = env_analysis["market_state"]
                result["environment"] = env_analysis["environment"]
                result["strategy_params"] = env_analysis["optimal_strategy"]

                # 3. 应用不同的信号过滤策略
                if env_analysis["market_state"] == "VOLATILE" or env_analysis["market_state"] == "EXTREME_VOLATILITY":
                    # 在动荡市场使用EMA斜率分析
                    print_colored(f"检测到动荡市场，启用EMA斜率过滤", Colors.WARNING)

                    # 进行EMA斜率分析
                    ema_trend = self.ema_slope_trend_filter(df)
                    result["ema_trend"] = ema_trend

                    # 高置信度EMA信号优先于原始信号
                    if ema_trend["confidence"] >= 0.7:
                        # 使用EMA趋势信号
                        result["filtered_signal"] = ema_trend["signal"]
                        result["reason"] = f"动荡市场中使用高置信度({ema_trend['confidence']:.2f})EMA趋势信号"

                        # 调整质量评分
                        confidence_bonus = (ema_trend["confidence"] - 0.5) * 2  # 0.7->0.4, 0.8->0.6, 0.9->0.8
                        result["adjusted_quality"] = min(10, quality_score * (1 + confidence_bonus * 0.2))

                        print_colored(
                            f"使用EMA趋势信号: {ema_trend['signal']}, "
                            f"调整后质量评分: {result['adjusted_quality']:.2f} (原始: {quality_score:.2f})",
                            Colors.BOLD + (Colors.GREEN if ema_trend['signal'] == 'BUY' else
                                           Colors.RED if ema_trend['signal'] == 'SELL' else Colors.YELLOW)
                        )

                    # 否则根据交易偏好过滤信号
                    elif env_analysis["trading_bias"] != "NEUTRAL":
                        if env_analysis["trading_bias"] == "LONG" and original_signal == "BUY":
                            result["filtered_signal"] = "BUY"
                            result["reason"] = "动荡市场但交易偏好为多头且原始信号为买入"
                        elif env_analysis["trading_bias"] == "SHORT" and original_signal == "SELL":
                            result["filtered_signal"] = "SELL"
                            result["reason"] = "动荡市场但交易偏好为空头且原始信号为卖出"
                        else:
                            result["filtered_signal"] = "WAIT"
                            result["reason"] = "动荡市场中原始信号与交易偏好不一致，建议观望"
                            result["adjusted_quality"] = quality_score * 0.7  # 降低质量评分
                    else:
                        # 无明确偏好，建议观望
                        result["filtered_signal"] = "WAIT"
                        result["reason"] = "动荡市场中无明确方向，建议观望"
                        result["adjusted_quality"] = quality_score * 0.6  # 大幅降低质量评分

                elif env_analysis["market_state"] == "TRENDING":
                    # 趋势市场 - 强化与趋势方向一致的信号，减弱逆趋势信号
                    trend_dir = env_analysis["env_details"].get("trend_direction", "neutral")

                    if trend_dir == "uptrend" and original_signal == "BUY":
                        # 增强买入信号
                        result["filtered_signal"] = "BUY"
                        result["adjusted_quality"] = min(10, quality_score * 1.2)  # 提高质量评分
                        result["reason"] = "顺应上升趋势的买入信号"
                    elif trend_dir == "downtrend" and original_signal == "SELL":
                        # 增强卖出信号
                        result["filtered_signal"] = "SELL"
                        result["adjusted_quality"] = min(10, quality_score * 1.2)  # 提高质量评分
                        result["reason"] = "顺应下降趋势的卖出信号"
                    elif trend_dir == "uptrend" and original_signal == "SELL":
                        # 削弱逆势卖出信号
                        result["filtered_signal"] = "WAIT"
                        result["adjusted_quality"] = quality_score * 0.7
                        result["reason"] = "在上升趋势中出现卖出信号，建议观望"
                    elif trend_dir == "downtrend" and original_signal == "BUY":
                        # 削弱逆势买入信号
                        result["filtered_signal"] = "WAIT"
                        result["adjusted_quality"] = quality_score * 0.7
                        result["reason"] = "在下降趋势中出现买入信号，建议观望"
                    else:
                        # 保持原始信号
                        result["reason"] = "保持原始信号，趋势不明确"

                elif env_analysis["market_state"] == "RANGING":
                    # 区间震荡市场 - 在区间边缘时反转交易，区间中间时保持原始信号
                    bb_width = df['BB_Width'].iloc[-1] if 'BB_Width' in df.columns else 0
                    price = df['close'].iloc[-1]
                    bb_upper = df['BB_Upper'].iloc[-1] if 'BB_Upper' in df.columns else float('inf')
                    bb_lower = df['BB_Lower'].iloc[-1] if 'BB_Lower' in df.columns else 0

                    # 计算当前价格在布林带中的位置 (0-1)
                    if bb_upper > bb_lower:
                        band_position = (price - bb_lower) / (bb_upper - bb_lower)

                        if band_position > 0.8 and original_signal == "BUY":
                            # 价格接近上轨且原始信号为买入，可能是追高，降低质量评分
                            result["adjusted_quality"] = quality_score * 0.8
                            result["reason"] = "区间震荡市场中价格接近上轨，买入风险增加"
                        elif band_position < 0.2 and original_signal == "SELL":
                            # 价格接近下轨且原始信号为卖出，可能是追低，降低质量评分
                            result["adjusted_quality"] = quality_score * 0.8
                            result["reason"] = "区间震荡市场中价格接近下轨，卖出风险增加"
                        else:
                            # 价格处于中间位置，保持原始信号
                            result["reason"] = "区间震荡市场中价格处于适中位置"

                elif env_analysis["market_state"] == "BREAKOUT":
                    # 突破市场 - 强化与突破方向一致的信号
                    breakout_dir = env_analysis["env_details"].get("breakout_direction", "unknown")

                    if breakout_dir == "upward" and original_signal == "BUY":
                        # 增强向上突破的买入信号
                        result["filtered_signal"] = "BUY"
                        result["adjusted_quality"] = min(10, quality_score * 1.3)  # 大幅提高质量评分
                        result["reason"] = "向上突破市场中的买入信号"
                    elif breakout_dir == "downward" and original_signal == "SELL":
                        # 增强向下突破的卖出信号
                        result["filtered_signal"] = "SELL"
                        result["adjusted_quality"] = min(10, quality_score * 1.3)  # 大幅提高质量评分
                        result["reason"] = "向下突破市场中的卖出信号"
                    else:
                        # 与突破方向不一致的信号，保持原样但给出警告
                        result["reason"] = f"与{breakout_dir}突破方向不一致的信号，请谨慎"

                # 将"WAIT"信号转换为"HOLD"以适配原有逻辑
                if result["filtered_signal"] == "WAIT":
                    result["filtered_signal"] = "HOLD"

                # 最终输出
                if result["filtered_signal"] != original_signal or abs(
                        result["adjusted_quality"] - quality_score) > 0.1:
                    signal_color = (Colors.GREEN if result["filtered_signal"] == "BUY" else
                                    Colors.RED if result["filtered_signal"] == "SELL" else
                                    Colors.YELLOW if result["filtered_signal"] == "HOLD" else Colors.GRAY)

                    original_color = (Colors.GREEN if original_signal == "BUY" else
                                      Colors.RED if original_signal == "SELL" else Colors.GRAY)

                    print_colored("\n----- 信号适配结果 -----", Colors.BLUE)
                    print_colored(
                        f"原始信号: {original_color}{original_signal}{Colors.RESET} -> "
                        f"适配信号: {signal_color}{result['filtered_signal']}{Colors.RESET}",
                        Colors.BOLD
                    )

                    if abs(result["adjusted_quality"] - quality_score) > 0.1:
                        print_colored(
                            f"质量评分: {quality_score:.2f} -> {result['adjusted_quality']:.2f} "
                            f"({(result['adjusted_quality'] - quality_score) / quality_score * 100:+.1f}%)",
                            Colors.INFO
                        )

                    print_colored(f"原因: {result['reason']}", Colors.INFO)
                else:
                    print_colored(f"保持原始信号 {original_signal} 和质量评分 {quality_score:.2f}", Colors.GREEN)

                return result

            except Exception as e:
                print_colored(f"❌ 生成环境适应型信号出错: {e}", Colors.ERROR)
                # 发生错误时返回原始信号
                result["filtered_signal"] = original_signal
                result["reason"] = f"分析过程出错: {str(e)}"
                return result

        def dynamic_take_profit_with_supertrend(self, df: pd.DataFrame, entry_price: float, position_side: str,
                                                market_state: str) -> Dict[str, Any]:
            """
            基于超级趋势指标和市场状态的动态止盈止损计算

            参数:
                df: 价格数据DataFrame
                entry_price: 入场价格
                position_side: 仓位方向 ('LONG' 或 'SHORT')
                market_state: 市场状态

            返回:
                Dict: 包含止盈参数的字典
            """
            print_colored("🎯 计算环境适应型止盈止损...", Colors.BLUE)

            result = {
                "take_profit_price": None,
                "stop_loss_price": None,
                "take_profit_pct": 0.025,  # 默认2.5%
                "stop_loss_pct": 0.02,  # 默认2%
                "use_trailing_stop": False,
                "trailing_callback": 0.01,  # 默认1%
                "supertrend_based": False
            }

            try:
                # 确保数据充分且有超级趋势指标
                if df is None or df.empty or len(df) < 20 or 'Supertrend' not in df.columns:
                    print_colored("⚠️ 数据不足或无超级趋势指标，使用默认止盈止损", Colors.WARNING)

                    # 根据市场状态调整默认值
                    if market_state == "TRENDING":
                        result["take_profit_pct"] = 0.04  # 4%
                        result["stop_loss_pct"] = 0.025  # 2.5%
                    elif market_state == "VOLATILE" or market_state == "EXTREME_VOLATILITY":
                        result["take_profit_pct"] = 0.05  # 5%
                        result["stop_loss_pct"] = 0.03  # 3%
                    elif market_state == "RANGING":
                        result["take_profit_pct"] = 0.02  # 2%
                        result["stop_loss_pct"] = 0.015  # 1.5%
                    elif market_state == "BREAKOUT":
                        result["take_profit_pct"] = 0.035  # 3.5%
                        result["stop_loss_pct"] = 0.02  # 2%

                    # 计算价格
                    if position_side == "LONG":
                        result["take_profit_price"] = entry_price * (1 + result["take_profit_pct"])
                        result["stop_loss_price"] = entry_price * (1 - result["stop_loss_pct"])
                    else:  # SHORT
                        result["take_profit_price"] = entry_price * (1 - result["take_profit_pct"])
                        result["stop_loss_price"] = entry_price * (1 + result["stop_loss_pct"])

                    return result

                # 获取超级趋势信息
                supertrend = df['Supertrend'].iloc[-1]
                supertrend_dir = df['Supertrend_Direction'].iloc[-1] if 'Supertrend_Direction' in df.columns else 0
                current_price = df['close'].iloc[-1]

                # 获取超级趋势强度
                supertrend_strength = 1.0
                if 'Supertrend_Strength' in df.columns:
                    supertrend_strength = df['Supertrend_Strength'].iloc[-1]

                # 获取ATR用于止损计算
                atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else (current_price * 0.01)

                # 根据市场状态调整基础止盈止损比例
                if market_state == "TRENDING":
                    base_tp_pct = 0.04  # 4%
                    base_sl_pct = 0.025  # 2.5%
                    trailing_enabled = True
                    trailing_callback = 0.015  # 1.5%
                elif market_state == "VOLATILE" or market_state == "EXTREME_VOLATILITY":
                    base_tp_pct = 0.05  # 5%
                    base_sl_pct = 0.03  # 3%
                    trailing_enabled = True
                    trailing_callback = 0.02  # 2%
                elif market_state == "RANGING":
                    base_tp_pct = 0.02  # 2%
                    base_sl_pct = 0.015  # 1.5%
                    trailing_enabled = False
                    trailing_callback = 0.01  # 1%
                elif market_state == "BREAKOUT":
                    base_tp_pct = 0.035  # 3.5%
                    base_sl_pct = 0.02  # 2%
                    trailing_enabled = True
                    trailing_callback = 0.015  # 1.5%
                else:
                    base_tp_pct = 0.025  # 2.5%
                    base_sl_pct = 0.02  # 2%
                    trailing_enabled = False
                    trailing_callback = 0.01  # 1%

                # 根据超级趋势和位置关系计算止盈止损
                if position_side == "LONG":
                    if supertrend_dir > 0:  # 超级趋势向上
                        # 使用超级趋势作为跟踪止盈线
                        result["take_profit_price"] = max(
                            entry_price * (1 + base_tp_pct),  # 确保至少有默认止盈
                            current_price * (1 + 0.01)  # 保证至少比当前价高1%
                        )

                        # 止损设为超级趋势线下方
                        result["stop_loss_price"] = min(
                            supertrend - (0.5 * atr),  # 超级趋势线下0.5个ATR
                            entry_price * (1 - base_sl_pct)  # 默认止损
                        )

                        # 启用追踪止盈
                        result["use_trailing_stop"] = trailing_enabled
                        result["trailing_callback"] = trailing_callback
                        result["supertrend_based"] = True

                        print_colored(
                            f"多头超级趋势止盈: ↑ 追踪止盈并保护利润，当前超级趋势线 {supertrend:.6f}",
                            Colors.GREEN
                        )

                    else:  # 超级趋势向下或中性
                        # 标准止盈，但提高止损警惕性
                        result["take_profit_price"] = entry_price * (1 + base_tp_pct)
                        result["stop_loss_price"] = max(
                            supertrend,  # 直接使用超级趋势线
                            entry_price * (1 - base_sl_pct * 1.2)  # 略微收紧止损
                        )

                        print_colored(
                            f"多头逆超级趋势: ⚠️ 使用更紧的止损，超级趋势线 {supertrend:.6f}",
                            Colors.YELLOW
                        )

                else:  # SHORT
                    if supertrend_dir < 0:  # 超级趋势向下
                        # 使用超级趋势作为跟踪止盈线
                        result["take_profit_price"] = min(
                            entry_price * (1 - base_tp_pct),  # 确保至少有默认止盈
                            current_price * (1 - 0.01)  # 保证至少比当前价低1%
                        )

                        # 止损设为超级趋势线上方
                        result["stop_loss_price"] = max(
                            supertrend + (0.5 * atr),  # 超级趋势线上0.5个ATR
                            entry_price * (1 + base_sl_pct)  # 默认止损
                        )

                        # 启用追踪止盈
                        result["use_trailing_stop"] = trailing_enabled
                        result["trailing_callback"] = trailing_callback
                        result["supertrend_based"] = True

                        print_colored(
                            f"空头超级趋势止盈: ↓ 追踪止盈并保护利润，当前超级趋势线 {supertrend:.6f}",
                            Colors.RED
                        )

                    else:  # 超级趋势向上或中性
                        # 标准止盈，但提高止损警惕性
                        result["take_profit_price"] = entry_price * (1 - base_tp_pct)
                        result["stop_loss_price"] = min(
                            supertrend,  # 直接使用超级趋势线
                            entry_price * (1 + base_sl_pct * 1.2)  # 略微收紧止损
                        )

                        print_colored(
                            f"空头逆超级趋势: ⚠️ 使用更紧的止损，超级趋势线 {supertrend:.6f}",
                            Colors.YELLOW
                        )

                # 记录实际止盈止损百分比
                result["take_profit_pct"] = abs(result["take_profit_price"] - entry_price) / entry_price
                result["stop_loss_pct"] = abs(result["stop_loss_price"] - entry_price) / entry_price

                # 打印最终止盈止损设置
                price_change_pct = (result["take_profit_price"] - entry_price) / entry_price * 100
                sl_change_pct = (result["stop_loss_price"] - entry_price) / entry_price * 100

                print_colored(
                    f"入场价: {entry_price:.6f}, 止盈价: {result['take_profit_price']:.6f} "
                    f"({price_change_pct:+.2f}%)",
                    Colors.GREEN
                )
                print_colored(
                    f"止损价: {result['stop_loss_price']:.6f} ({sl_change_pct:+.2f}%)",
                    Colors.RED
                )

                if result["use_trailing_stop"]:
                    print_colored(
                        f"启用追踪止盈，回调: {result['trailing_callback'] * 100:.2f}%",
                        Colors.INFO
                    )

                return result

            except Exception as e:
                print_colored(f"❌ 计算动态止盈出错: {e}", Colors.ERROR)

                # 发生错误时使用默认值
                if position_side == "LONG":
                    result["take_profit_price"] = entry_price * 1.025  # 2.5%止盈
                    result["stop_loss_price"] = entry_price * 0.98  # 2%止损
                else:  # SHORT
                    result["take_profit_price"] = entry_price * 0.975  # 2.5%止盈
                    result["stop_loss_price"] = entry_price * 1.02  # 2%止损

                return result

        def apply_dynamic_tp_sl(self, symbol: str, df: pd.DataFrame, position_info: Dict[str, Any]) -> Dict[str, Any]:
            """
            应用动态止盈止损策略

            参数:
                symbol: 交易对符号
                df: 价格数据DataFrame
                position_info: 持仓信息

            返回:
                更新后的持仓信息
            """
            if not self.dynamic_tp_sl_enabled:
                return position_info

            try:
                entry_price = position_info.get("entry_price", 0)
                position_side = position_info.get("position_side", "LONG")

                if entry_price <= 0:
                    print_colored(f"⚠️ {symbol} 无有效入场价格，无法计算动态止盈止损", Colors.WARNING)
                    return position_info

                # 获取市场状态
                market_state = "NORMAL"
                if symbol in self.environment_history:
                    market_state = self.environment_history[symbol].get("market_state", "NORMAL")
                else:
                    # 进行市场环境分析
                    env_analysis = self.analyze_market_environment(df, symbol)
                    market_state = env_analysis["market_state"]

                # 计算动态止盈止损
                tp_sl_result = self.dynamic_take_profit_with_supertrend(df, entry_price, position_side, market_state)

                # 更新持仓信息
                if position_side == "LONG":
                    position_info["dynamic_take_profit"] = (tp_sl_result[
                                                                "take_profit_price"] - entry_price) / entry_price
                    position_info["stop_loss"] = (tp_sl_result["stop_loss_price"] - entry_price) / entry_price
                else:  # SHORT
                    position_info["dynamic_take_profit"] = (entry_price - tp_sl_result[
                        "take_profit_price"]) / entry_price
                    position_info["stop_loss"] = (entry_price - tp_sl_result["stop_loss_price"]) / entry_price

                position_info["tp_price"] = tp_sl_result["take_profit_price"]
                position_info["sl_price"] = tp_sl_result["stop_loss_price"]

                # 追踪止盈设置
                if tp_sl_result["use_trailing_stop"]:
                    position_info["use_trailing_stop"] = True
                    position_info["trailing_callback"] = tp_sl_result["trailing_callback"]
                    print_colored(f"{symbol} 启用追踪止盈，回调: {tp_sl_result['trailing_callback'] * 100:.2f}%",
                                  Colors.INFO)

                print_colored(
                    f"{symbol} {position_side} 环境适应型止盈止损已应用: "
                    f"止盈 {position_info['dynamic_take_profit'] * 100:.2f}%, "
                    f"止损 {abs(position_info['stop_loss']) * 100:.2f}%",
                    Colors.GREEN
                )

                return position_info

            except Exception as e:
                print_colored(f"❌ {symbol} 应用动态止盈止损失败: {e}", Colors.ERROR)
                return position_info

        def get_market_environment_stats(self) -> Dict[str, Any]:
            """
            获取所有交易对的市场环境统计

            返回:
                包含环境统计的字典
            """
            stats = {
                "trending_count": 0,
                "ranging_count": 0,
                "breakout_count": 0,
                "volatile_count": 0,
                "symbols": {},
                "global_environment": "NEUTRAL"
            }

            current_time = pd.Timestamp.now()
            valid_history = {}

            # 过滤近期的环境历史记录
            for symbol, history in self.environment_history.items():
                if (current_time - history["timestamp"]).total_seconds() < 7200:  # 2小时内的记录
                    valid_history[symbol] = history
                    market_state = history["market_state"]

                    if market_state == "TRENDING":
                        stats["trending_count"] += 1
                    elif market_state == "RANGING":
                        stats["ranging_count"] += 1
                    elif market_state == "BREAKOUT":
                        stats["breakout_count"] += 1
                    elif market_state in ["VOLATILE", "EXTREME_VOLATILITY"]:
                        stats["volatile_count"] += 1

                    # 记录到symbols字典
                    stats["symbols"][symbol] = {
                        "market_state": market_state,
                        "trading_bias": history["trading_bias"],
                        "volatility_score": history["volatility_score"]
                    }

            # 计算总数
            total = len(valid_history)

            if total > 0:
                # 计算百分比
                stats["trending_pct"] = (stats["trending_count"] / total * 100)
                stats["ranging_pct"] = (stats["ranging_count"] / total * 100)
                stats["breakout_pct"] = (stats["breakout_count"] / total * 100)
                stats["volatile_pct"] = (stats["volatile_count"] / total * 100)

                # 确定全局市场环境
                if stats["volatile_pct"] > 40:
                    stats["global_environment"] = "VOLATILE"
                elif stats["trending_pct"] > 50:
                    stats["global_environment"] = "TRENDING"
                elif stats["ranging_pct"] > 50:
                    stats["global_environment"] = "RANGING"
                elif stats["breakout_pct"] > 30:
                    stats["global_environment"] = "BREAKOUT"
                else:
                    stats["global_environment"] = "MIXED"

            return stats
