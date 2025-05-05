
"""
市场状态分类和反转检测模块
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from logger_utils import Colors, print_colored
from indicators_module import get_smc_trend_and_duration, find_swing_points


def classify_market_state(df: pd.DataFrame, trend_filter: str = 'default') -> Dict[str, Any]:
    """
    市场状态分类

    参数:
        df: 价格数据
        trend_filter: 趋势过滤器类型

    返回:
        市场状态信息字典
    """
    if df is None or len(df) < 20:
        return {"state": "UNKNOWN", "confidence": 0, "description": "数据不足"}

    # 获取SMC趋势
    trend, duration, trend_info = get_smc_trend_and_duration(df)

    # 计算价格波动性
    volatility = 0.0
    if 'ATR' in df.columns:
        current_atr = df['ATR'].iloc[-1]
        avg_atr = df['ATR'].iloc[-20:].mean()
        if avg_atr > 0:
            volatility = current_atr / avg_atr

    # 检查是否处于震荡市场
    is_ranging = False
    if 'ADX' in df.columns:
        adx = df['ADX'].iloc[-1]
        is_ranging = adx < 20

    # 使用布林带检查是否处于压缩状态
    is_compressed = False
    bb_width = 0.0
    if all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
        bb_width = (df['BB_Upper'].iloc[-1] - df['BB_Lower'].iloc[-1]) / df['BB_Middle'].iloc[-1]
        bb_width_avg = ((df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']).iloc[-20:].mean()
        is_compressed = bb_width < 0.6 * bb_width_avg

    # 综合评估市场状态
    state = "NEUTRAL"
    confidence = 0.5
    description = "市场状态不明确"

    if trend == "UP":
        if is_ranging:
            state = "WEAK_UPTREND"
            confidence = 0.6
            description = "弱上升趋势，出现震荡"
        elif is_compressed:
            state = "UPTREND_COMPRESSION"
            confidence = 0.7
            description = "上升趋势压缩，可能即将突破"
        elif volatility > 1.5:
            state = "VOLATILE_UPTREND"
            confidence = 0.75
            description = "波动性上升趋势"
        else:
            state = "STRONG_UPTREND"
            confidence = 0.85
            description = "强劲上升趋势"

    elif trend == "DOWN":
        if is_ranging:
            state = "WEAK_DOWNTREND"
            confidence = 0.6
            description = "弱下降趋势，出现震荡"
        elif is_compressed:
            state = "DOWNTREND_COMPRESSION"
            confidence = 0.7
            description = "下降趋势压缩，可能即将突破"
        elif volatility > 1.5:
            state = "VOLATILE_DOWNTREND"
            confidence = 0.75
            description = "波动性下降趋势"
        else:
            state = "STRONG_DOWNTREND"
            confidence = 0.85
            description = "强劲下降趋势"

    else:  # NEUTRAL trend
        if is_compressed:
            state = "CONSOLIDATION"
            confidence = 0.8
            description = "价格盘整，蓄势待发"
        elif volatility > 1.3:
            state = "CHOPPY"
            confidence = 0.65
            description = "震荡市场，高波动无趋势"
        elif is_ranging:
            state = "RANGING"
            confidence = 0.75
            description = "区间震荡，无明确趋势"
        else:
            state = "NEUTRAL"
            confidence = 0.5
            description = "中性市场，方向不明"

    # 日志输出
    state_color = (
        Colors.GREEN if "UPTREND" in state else
        Colors.RED if "DOWNTREND" in state else
        Colors.YELLOW if state in ["RANGING", "CONSOLIDATION"] else
        Colors.GRAY
    )

    print_colored(f"市场状态: {state_color}{state}{Colors.RESET} ({description})", Colors.INFO)
    print_colored(f"状态置信度: {confidence:.2f}, 趋势持续: {duration}分钟", Colors.INFO)

    result = {
        "state": state,
        "confidence": confidence,
        "description": description,
        "trend": trend,
        "duration": duration,
        "trend_info": trend_info,
        "volatility": volatility,
        "is_ranging": is_ranging,
        "is_compressed": is_compressed,
        "bb_width": bb_width
    }

    return result


def detect_price_structure_reversal(df: pd.DataFrame, lookback: int = 20, threshold: float = 0.6) -> Tuple[
    float, List[str]]:
    """
    检测价格结构反转信号

    参数:
        df: 价格数据
        lookback: 回溯检查的K线数量
        threshold: 反转确认阈值

    返回:
        (反转评分, 信号列表)
    """
    if df is None or len(df) < lookback:
        return 0.0, []

    reversal_score = 0.0
    signals = []

    try:
        # 获取最近的趋势方向
        trend, _, _ = get_smc_trend_and_duration(df)

        # 获取摆动高点和低点
        swing_highs, swing_lows = find_swing_points(df, window=3)

        # 上升趋势中检测顶部反转模式
        if trend == "UP":
            # 检测连续两个较低高点(lower highs)
            if len(swing_highs) >= 2 and swing_highs[-1] < swing_highs[-2]:
                reversal_score += 0.3
                signals.append("检测到下降高点模式")

            # 检测长上影线蜡烛（卖压信号）
            recent_candles = min(5, len(df) - 1)
            for i in range(1, recent_candles + 1):
                upper_wick = df['high'].iloc[-i] - max(df['open'].iloc[-i], df['close'].iloc[-i])
                candle_range = df['high'].iloc[-i] - df['low'].iloc[-i]

                if candle_range > 0 and upper_wick / candle_range > 0.6:
                    reversal_score += 0.2
                    signals.append(f"检测到{i}根K线前有长上影线蜡烛")
                    break

            # 检测是否有击穿支撑水平
            if 'EMA20' in df.columns:
                if df['close'].iloc[-2] > df['EMA20'].iloc[-2] and df['close'].iloc[-1] < df['EMA20'].iloc[-1]:
                    reversal_score += 0.25
                    signals.append("价格跌破EMA20支撑")

        # 下降趋势中检测底部反转模式
        elif trend == "DOWN":
            # 检测连续两个较高低点(higher lows)
            if len(swing_lows) >= 2 and swing_lows[-1] > swing_lows[-2]:
                reversal_score += 0.3
                signals.append("检测到上升低点模式")

            # 检测长下影线蜡烛（买压信号）
            recent_candles = min(5, len(df) - 1)
            for i in range(1, recent_candles + 1):
                lower_wick = min(df['open'].iloc[-i], df['close'].iloc[-i]) - df['low'].iloc[-i]
                candle_range = df['high'].iloc[-i] - df['low'].iloc[-i]

                if candle_range > 0 and lower_wick / candle_range > 0.6:
                    reversal_score += 0.2
                    signals.append(f"检测到{i}根K线前有长下影线蜡烛")
                    break

            # 检测是否有突破阻力水平
            if 'EMA20' in df.columns:
                if df['close'].iloc[-2] < df['EMA20'].iloc[-2] and df['close'].iloc[-1] > df['EMA20'].iloc[-1]:
                    reversal_score += 0.25
                    signals.append("价格突破EMA20阻力")

        # 检查布林带反转信号
        if all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            if trend == "UP" and df['close'].iloc[-1] < df['BB_Middle'].iloc[-1]:
                reversal_score += 0.2
                signals.append("价格从上方跌破布林带中轨")
            elif trend == "DOWN" and df['close'].iloc[-1] > df['BB_Middle'].iloc[-1]:
                reversal_score += 0.2
                signals.append("价格从下方突破布林带中轨")

        # 检查交易量变化
        if 'volume' in df.columns:
            avg_volume = df['volume'].iloc[-lookback:-1].mean()
            if avg_volume > 0:
                vol_change = df['volume'].iloc[-1] / avg_volume
                if vol_change > 2.0:
                    reversal_score += 0.15
                    signals.append(f"交易量突然增加{vol_change:.1f}倍")

    except Exception as e:
        print_colored(f"检测价格结构反转出错: {e}", Colors.ERROR)

    return reversal_score, signals


def detect_indicator_divergences(df: pd.DataFrame, sensitivity: float = 0.5) -> Tuple[float, List[str]]:
    """
    检测指标与价格的背离

    参数:
        df: 价格数据
        sensitivity: 灵敏度参数

    返回:
        (背离评分, 信号列表)
    """
    if df is None or len(df) < 10:
        return 0.0, []

    divergence_score = 0.0
    signals = []

    try:
        # RSI背离检测
        if 'RSI' in df.columns:
            # 获取最近的高点和低点
            recent = 10
            price_highs = []
            price_lows = []
            rsi_highs = []
            rsi_lows = []

            for i in range(2, recent + 2):
                # 确认i是局部高点
                if i < len(df) and i - 2 >= 0:
                    if df['close'].iloc[-i] > df['close'].iloc[-i - 1] and df['close'].iloc[-i] > df['close'].iloc[
                        -i + 1]:
                        price_highs.append((len(df) - i, df['close'].iloc[-i]))

                    # 确认i是局部低点
                    if df['close'].iloc[-i] < df['close'].iloc[-i - 1] and df['close'].iloc[-i] < df['close'].iloc[
                        -i + 1]:
                        price_lows.append((len(df) - i, df['close'].iloc[-i]))

                    # 确认RSI高点
                    if df['RSI'].iloc[-i] > df['RSI'].iloc[-i - 1] and df['RSI'].iloc[-i] > df['RSI'].iloc[-i + 1]:
                        rsi_highs.append((len(df) - i, df['RSI'].iloc[-i]))

                    # 确认RSI低点
                    if df['RSI'].iloc[-i] < df['RSI'].iloc[-i - 1] and df['RSI'].iloc[-i] < df['RSI'].iloc[-i + 1]:
                        rsi_lows.append((len(df) - i, df['RSI'].iloc[-i]))

            # 检测看跌背离（价格创新高但RSI未创新高）
            if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                # 价格创新高
                if price_highs[0][1] > price_highs[1][1]:
                    # RSI未创新高
                    if rsi_highs[0][1] < rsi_highs[1][1]:
                        divergence_score += 0.4
                        signals.append("RSI看跌背离")

            # 检测看涨背离（价格创新低但RSI未创新低）
            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                # 价格创新低
                if price_lows[0][1] < price_lows[1][1]:
                    # RSI未创新低
                    if rsi_lows[0][1] > rsi_lows[1][1]:
                        divergence_score += 0.4
                        signals.append("RSI看涨背离")

        # MACD背离检测
        if 'MACD' in df.columns and 'MACD_signal' in df.columns and 'MACD_histogram' in df.columns:
            recent = 10

            # 检测MACD柱状图与价格的背离
            highs_idx = []
            lows_idx = []

            for i in range(2, recent + 2):
                if i < len(df) and i - 2 >= 0:
                    # 寻找价格高点
                    if df['close'].iloc[-i] > df['close'].iloc[-i - 1] and df['close'].iloc[-i] > df['close'].iloc[
                        -i + 1]:
                        highs_idx.append(len(df) - i)

                    # 寻找价格低点
                    if df['close'].iloc[-i] < df['close'].iloc[-i - 1] and df['close'].iloc[-i] < df['close'].iloc[
                        -i + 1]:
                        lows_idx.append(len(df) - i)

            # 检测看跌背离
            if len(highs_idx) >= 2:
                h1, h2 = highs_idx[0], highs_idx[1]
                if df['close'].iloc[h1] > df['close'].iloc[h2]:  # 价格高点上升
                    if df['MACD_histogram'].iloc[h1] < df['MACD_histogram'].iloc[h2]:  # MACD柱状图下降
                        divergence_score += 0.35
                        signals.append("MACD柱状图看跌背离")

            # 检测看涨背离
            if len(lows_idx) >= 2:
                l1, l2 = lows_idx[0], lows_idx[1]
                if df['close'].iloc[l1] < df['close'].iloc[l2]:  # 价格低点下降
                    if df['MACD_histogram'].iloc[l1] > df['MACD_histogram'].iloc[l2]:  # MACD柱状图上升
                        divergence_score += 0.35
                        signals.append("MACD柱状图看涨背离")

        # 适应灵敏度设置
        divergence_score *= sensitivity

    except Exception as e:
        print_colored(f"检测指标背离出错: {e}", Colors.ERROR)

    return divergence_score, signals


def detect_volatility_volume_changes(df: pd.DataFrame, atr_threshold: float = 1.5, volume_threshold: float = 2.0) -> \
Tuple[float, List[str]]:
    """
    检测波动性和交易量的突然变化

    参数:
        df: 价格数据
        atr_threshold: ATR变化阈值
        volume_threshold: 交易量变化阈值

    返回:
        (变化评分, 信号列表)
    """
    if df is None or len(df) < 10:
        return 0.0, []

    change_score = 0.0
    signals = []

    try:
        # ATR突变检测
        if 'ATR' in df.columns:
            atr_ratio = df['ATR'].iloc[-1] / df['ATR'].iloc[-5:].mean()
            if atr_ratio > atr_threshold:
                change_score += 0.3
                signals.append(f"ATR突然增加: {atr_ratio:.2f}倍")

        # 成交量突变检测
        if 'volume' in df.columns:
            vol_ratio = df['volume'].iloc[-1] / df['volume'].iloc[-10:].mean()
            if vol_ratio > volume_threshold:
                change_score += 0.2
                signals.append(f"成交量突增: {vol_ratio:.2f}倍")

        # 布林带宽度变化检测
        if all(x in df.columns for x in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            current_width = (df['BB_Upper'].iloc[-1] - df['BB_Lower'].iloc[-1]) / df['BB_Middle'].iloc[-1]
            avg_width = ((df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']).iloc[-10:].mean()
            width_change = current_width / avg_width

            # 布林带突然扩张超过50%
            if width_change > 1.5:
                change_score += 0.2
                signals.append(f"布林带突然扩张: {width_change:.2f}倍")

    except Exception as e:
        print_colored(f"检测波动性和交易量变化出错: {e}", Colors.ERROR)

    return change_score, signals


def detect_fvg_structure_changes(df: pd.DataFrame, fvg_data: List[Dict[str, Any]], trend_data: Dict[str, Any]) -> Tuple[
    float, List[str]]:
    """
    通过FVG和市场结构变化检测趋势逆转

    参数:
        df: 价格数据
        fvg_data: FVG信息
        trend_data: 趋势信息

    返回:
        (结构变化评分, 信号列表)
    """
    if df is None or len(df) < 10 or not fvg_data:
        return 0.0, []

    structure_change_score = 0.0
    signals = []

    try:
        # 检测最近的FVG是否已被快速填补
        recent_filled_fvgs = [fvg for fvg in fvg_data if fvg['is_filled'] and
                              (fvg['fill_time'] - fvg['end_idx']) < 5]

        if recent_filled_fvgs:
            structure_change_score += 0.3
            signals.append(f"最近{len(recent_filled_fvgs)}个FVG被快速填补")

        # 检测相反方向的新FVG形成
        current_trend = trend_data.get('direction', 'NEUTRAL')
        opposing_fvgs = [fvg for fvg in fvg_data if fvg['direction'] != current_trend and
                         not fvg['is_filled'] and fvg['age'] <= 3]

        if opposing_fvgs:
            structure_change_score += 0.3
            signals.append(f"检测到{len(opposing_fvgs)}个逆趋势FVG")

        # 检测关键结构水平突破
        if current_trend == "UP" and 'EMA20' in df.columns:
            if df['close'].iloc[-2] > df['EMA20'].iloc[-2] and df['close'].iloc[-1] < df['EMA20'].iloc[-1]:
                structure_change_score += 0.25
                signals.append("价格跌破关键EMA20支撑")

        elif current_trend == "DOWN" and 'EMA20' in df.columns:
            if df['close'].iloc[-2] < df['EMA20'].iloc[-2] and df['close'].iloc[-1] > df['EMA20'].iloc[-1]:
                structure_change_score += 0.25
                signals.append("价格突破关键EMA20阻力")

    except Exception as e:
        print_colored(f"检测FVG和结构变化出错: {e}", Colors.ERROR)

    return structure_change_score, signals


def detect_market_reversal(df: pd.DataFrame, fvg_data: List[Dict[str, Any]], trend_data: Dict[str, Any],
                           market_state: Dict[str, Any], sensitivity: float = 0.5) -> Dict[str, Any]:
    """
    综合反转检测系统 - 整合多个信号源

    参数:
        df: 价格数据
        fvg_data: FVG信息
        trend_data: 趋势信息
        market_state: 市场状态信息
        sensitivity: 灵敏度参数

    返回:
        反转检测结果字典
    """
    # 根据市场状态调整灵敏度
    if market_state["state"].startswith("STRONG_"):
        # 强趋势需要更强的反转信号
        adjusted_sensitivity = sensitivity * 0.7
    elif market_state["state"].startswith("WEAK_"):
        # 弱趋势更容易反转
        adjusted_sensitivity = sensitivity * 1.2
    else:
        adjusted_sensitivity = sensitivity

    # 收集各子系统的反转分数
    price_score, price_signals = detect_price_structure_reversal(df)
    div_score, div_signals = detect_indicator_divergences(df, adjusted_sensitivity)
    vol_score, vol_signals = detect_volatility_volume_changes(df)
    struct_score, struct_signals = detect_fvg_structure_changes(df, fvg_data, trend_data)

    # 整合所有信号
    total_signals = price_signals + div_signals + vol_signals + struct_signals

    # 加权计算总反转分数
    weights = {
        'price': 0.3,
        'divergence': 0.3,
        'volatility': 0.2,
        'structure': 0.2
    }

    total_score = (
            weights['price'] * price_score +
            weights['divergence'] * div_score +
            weights['volatility'] * vol_score +
            weights['structure'] * struct_score
    )

    # 应用最终灵敏度调整
    reversal_probability = min(1.0, total_score * adjusted_sensitivity)

    # 确定反转信号的强度
    if reversal_probability >= 0.8:
        strength = "强"
    elif reversal_probability >= 0.6:
        strength = "中等"
    elif reversal_probability >= 0.4:
        strength = "弱"
    else:
        strength = "无"

    # 综合结果
    result = {
        'probability': reversal_probability,
        'strength': strength,
        'signals': total_signals,
        'price_score': price_score,
        'divergence_score': div_score,
        'volatility_score': vol_score,
        'structure_score': struct_score,
        'adjusted_sensitivity': adjusted_sensitivity
    }

    # 日志输出
    if total_signals:
        signal_color = (
            Colors.RED + Colors.BOLD if reversal_probability >= 0.8 else
            Colors.RED if reversal_probability >= 0.6 else
            Colors.YELLOW if reversal_probability >= 0.4 else
            Colors.GRAY
        )

        print_colored(
            f"市场反转检测: {signal_color}{strength}反转信号{Colors.RESET} (概率: {reversal_probability:.2f})",
            Colors.INFO)
        for i, signal in enumerate(total_signals[:3]):  # 只显示前3个信号
            print_colored(f"  {i + 1}. {signal}", Colors.INFO)

        if len(total_signals) > 3:
            print_colored(f"  ... 共{len(total_signals)}个信号", Colors.INFO)
    else:
        print_colored("市场反转检测: 无明显反转信号", Colors.INFO)

    return result