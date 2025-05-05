# 文件: fvg_module.py
"""
FVG (Fair Value Gap) 和市场不平衡检测模块
用于检测和分析市场不平衡和各类价格缺口
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from logger_utils import Colors, print_colored


def detect_fair_value_gap(df: pd.DataFrame, timeframe: str = "15m", lookback: int = 100,
                          sensitivity: float = 0.5, use_volume: bool = True) -> List[Dict[str, Any]]:
    """
    检测Fair Value Gaps (FVGs)

    参数:
        df: 包含OHLC数据的DataFrame
        timeframe: 时间框架
        lookback: 回溯检查的K线数量
        sensitivity: 灵敏度参数 (0-1)
        use_volume: 是否使用成交量分析

    返回:
        FVG列表，每个FVG包含详细信息
    """
    if df is None or len(df) < 4:
        print_colored("⚠️ 数据不足，无法检测FVG", Colors.WARNING)
        return []

    fvg_list = []
    min_gap_threshold = 0.001  # 最小缺口阈值，避免微小缺口

    # 计算ATR用于过滤小缺口
    if 'ATR' in df.columns:
        atr = df['ATR'].iloc[-1]
        # 根据灵敏度调整缺口阈值
        min_gap_threshold = atr * (0.1 + (1 - sensitivity) * 0.3)
        print_colored(f"FVG最小缺口阈值: {min_gap_threshold:.6f} ({sensitivity:.1f}倍灵敏度)", Colors.INFO)

    # 限制回溯范围
    check_range = min(lookback, len(df) - 3)

    # 从最新的K线开始回溯检查
    for i in range(len(df) - 3, len(df) - 3 - check_range, -1):
        if i < 0:
            break

        # 获取三根K线
        candle1 = {
            'open': df['open'].iloc[i],
            'high': df['high'].iloc[i],
            'low': df['low'].iloc[i],
            'close': df['close'].iloc[i],
            'volume': df['volume'].iloc[i] if 'volume' in df.columns else 0
        }

        candle2 = {
            'open': df['open'].iloc[i + 1],
            'high': df['high'].iloc[i + 1],
            'low': df['low'].iloc[i + 1],
            'close': df['close'].iloc[i + 1],
            'volume': df['volume'].iloc[i + 1] if 'volume' in df.columns else 0
        }

        candle3 = {
            'open': df['open'].iloc[i + 2],
            'high': df['high'].iloc[i + 2],
            'low': df['low'].iloc[i + 2],
            'close': df['close'].iloc[i + 2],
            'volume': df['volume'].iloc[i + 2] if 'volume' in df.columns else 0
        }

        # 检测看涨FVG (Bullish FVG)
        if candle1['low'] > candle3['high']:
            gap_size = candle1['low'] - candle3['high']

            # 确保缺口足够大
            if gap_size >= min_gap_threshold:
                # 计算中间K线的爆发强度
                candle2_range = candle2['high'] - candle2['low']
                candle_avg_range = df['high'].iloc[i - 5:i + 1].mean() - df['low'].iloc[i - 5:i + 1].mean()
                impulse_strength = candle2_range / candle_avg_range if candle_avg_range > 0 else 1.0

                # 确保足够的推动力
                if impulse_strength >= sensitivity:
                    # 计算体积权重
                    volume_weight = 1.0
                    if use_volume and 'volume' in df.columns:
                        avg_volume = df['volume'].iloc[i - 5:i + 1].mean()
                        if avg_volume > 0:
                            volume_weight = candle2['volume'] / avg_volume

                    # 检查是否已被填补
                    filled = False
                    fill_price = None
                    fill_time = None

                    # 检查后续K线是否填补了FVG
                    for j in range(i + 3, len(df)):
                        if df['low'].iloc[j] <= candle3['high']:
                            filled = True
                            fill_price = candle3['high']
                            fill_time = j
                            break

                    # 记录FVG
                    fvg_data = {
                        'type': 'bullish',
                        'direction': 'UP',
                        'start_idx': i,
                        'end_idx': i + 2,
                        'upper_boundary': candle1['low'],
                        'lower_boundary': candle3['high'],
                        'gap_size': gap_size,
                        'gap_midpoint': candle3['high'] + gap_size / 2,
                        'impulse_strength': impulse_strength,
                        'volume_weight': volume_weight,
                        'is_filled': filled,
                        'fill_price': fill_price,
                        'fill_time': fill_time,
                        'age': len(df) - (i + 2),
                        'timeframe': timeframe,
                        'creation_time': df.index[i + 2] if hasattr(df.index, '__getitem__') else None
                    }

                    fvg_list.append(fvg_data)

        # 检测看跌FVG (Bearish FVG)
        if candle1['high'] < candle3['low']:
            gap_size = candle3['low'] - candle1['high']

            # 确保缺口足够大
            if gap_size >= min_gap_threshold:
                # 计算中间K线的爆发强度
                candle2_range = candle2['high'] - candle2['low']
                candle_avg_range = df['high'].iloc[i - 5:i + 1].mean() - df['low'].iloc[i - 5:i + 1].mean()
                impulse_strength = candle2_range / candle_avg_range if candle_avg_range > 0 else 1.0

                # 确保足够的推动力
                if impulse_strength >= sensitivity:
                    # 计算体积权重
                    volume_weight = 1.0
                    if use_volume and 'volume' in df.columns:
                        avg_volume = df['volume'].iloc[i - 5:i + 1].mean()
                        if avg_volume > 0:
                            volume_weight = candle2['volume'] / avg_volume

                    # 检查是否已被填补
                    filled = False
                    fill_price = None
                    fill_time = None

                    # 检查后续K线是否填补了FVG
                    for j in range(i + 3, len(df)):
                        if df['high'].iloc[j] >= candle3['low']:
                            filled = True
                            fill_price = candle3['low']
                            fill_time = j
                            break

                    # 记录FVG
                    fvg_data = {
                        'type': 'bearish',
                        'direction': 'DOWN',
                        'start_idx': i,
                        'end_idx': i + 2,
                        'upper_boundary': candle3['low'],
                        'lower_boundary': candle1['high'],
                        'gap_size': gap_size,
                        'gap_midpoint': candle1['high'] + gap_size / 2,
                        'impulse_strength': impulse_strength,
                        'volume_weight': volume_weight,
                        'is_filled': filled,
                        'fill_price': fill_price,
                        'fill_time': fill_time,
                        'age': len(df) - (i + 2),
                        'timeframe': timeframe,
                        'creation_time': df.index[i + 2] if hasattr(df.index, '__getitem__') else None
                    }

                    fvg_list.append(fvg_data)

    # 按照缺口大小排序
    fvg_list = sorted(fvg_list, key=lambda x: x['gap_size'] * x['impulse_strength'], reverse=True)

    # 打印检测结果
    if fvg_list:
        print_colored(f"检测到 {len(fvg_list)} 个FVG", Colors.INFO)
        for i, fvg in enumerate(fvg_list[:3]):  # 只显示前3个
            status = "已填补" if fvg['is_filled'] else "未填补"
            fvg_color = Colors.GREEN if fvg['direction'] == 'UP' else Colors.RED
            print_colored(
                f"{i + 1}. {fvg_color}{fvg['type']} FVG{Colors.RESET}: "
                f"大小={fvg['gap_size']:.6f}, 强度={fvg['impulse_strength']:.2f}, "
                f"状态={status}, 年龄={fvg['age']}根K线",
                Colors.INFO
            )
    else:
        print_colored("未检测到任何FVG", Colors.INFO)

    return fvg_list


def classify_gap_type(df: pd.DataFrame, fvg_data: List[Dict[str, Any]], trend_data: Dict[str, Any]) -> List[
    Dict[str, Any]]:
    """
    将检测到的FVG分类为breakaway gap, measuring gap, 等

    参数:
        df: 价格数据
        fvg_data: FVG列表
        trend_data: 趋势信息

    返回:
        添加了分类信息的FVG列表
    """
    if not fvg_data:
        return []

    trend_direction = trend_data.get('direction', 'NEUTRAL')
    trend_duration = trend_data.get('duration', 0)

    # 获取最近的摆动点
    swing_highs = []
    swing_lows = []
    if 'Swing_Highs' in df.columns:
        swing_highs = df['Swing_Highs'].dropna().tolist()
    if 'Swing_Lows' in df.columns:
        swing_lows = df['Swing_Lows'].dropna().tolist()

    for fvg in fvg_data:
        # 默认为未分类
        fvg['gap_classification'] = 'regular'

        # 获取FVG所在位置的趋势上下文
        fvg_idx = fvg['end_idx']
        recent_trend = 'NEUTRAL'

        # 计算FVG之前的短期趋势
        if fvg_idx > 10:
            pre_fvg_close = df['close'].iloc[fvg_idx - 10:fvg_idx]
            if len(pre_fvg_close) > 1:
                pre_trend = 'UP' if pre_fvg_close.iloc[-1] > pre_fvg_close.iloc[0] else 'DOWN'

                # 检查FVG之后的价格走势
                post_idx = min(fvg_idx + 10, len(df) - 1)
                if post_idx > fvg_idx:
                    post_fvg_close = df['close'].iloc[fvg_idx:post_idx]
                    if len(post_fvg_close) > 1:
                        post_trend = 'UP' if post_fvg_close.iloc[-1] > post_fvg_close.iloc[0] else 'DOWN'

                        # Breakaway Gap: 出现在盘整区域之后，标志着新趋势的开始
                        if pre_trend == 'NEUTRAL' and post_trend != 'NEUTRAL':
                            fvg['gap_classification'] = 'breakaway'

                        # Measuring Gap: 出现在已建立趋势的中途
                        elif pre_trend == post_trend and pre_trend != 'NEUTRAL':
                            fvg['gap_classification'] = 'measuring'

                        # Exhaustion Gap: 出现在趋势末端，通常是剧烈的价格变动
                        elif pre_trend != 'NEUTRAL' and post_trend != pre_trend:
                            fvg['gap_classification'] = 'exhaustion'

        # 检查是否为已填补的缺口
        if fvg['is_filled']:
            fvg['gap_classification'] = 'filled ' + fvg['gap_classification']

        # 检查是否为部分填补的缺口(tapped gap)
        elif fvg_idx < len(df) - 1:
            if fvg['direction'] == 'UP':
                min_post_price = df['low'].iloc[fvg_idx:].min()
                if min_post_price <= fvg['upper_boundary'] and min_post_price >= fvg['lower_boundary']:
                    fvg['gap_classification'] = 'tapped ' + fvg['gap_classification']
            else:  # DOWN direction
                max_post_price = df['high'].iloc[fvg_idx:].max()
                if max_post_price >= fvg['lower_boundary'] and max_post_price <= fvg['upper_boundary']:
                    fvg['gap_classification'] = 'tapped ' + fvg['gap_classification']

    return fvg_data


def detect_imbalance_patterns(df: pd.DataFrame, lookback: int = 20) -> Dict[str, Any]:
    """
    检测买入-卖出不平衡(BISI)和卖出-买入不平衡(SIBI)模式

    参数:
        df: 价格数据
        lookback: 回溯检查的K线数量

    返回:
        包含不平衡模式信息的字典
    """
    if df is None or len(df) < lookback + 5:
        return {"bisi": [], "sibi": [], "detected": False}

    # 初始化结果
    result = {
        "bisi": [],
        "sibi": [],
        "detected": False
    }

    try:
        # 计算每根K线的买卖压力
        bull_power = []
        bear_power = []

        for i in range(len(df) - lookback, len(df)):
            if i < 0:
                continue

            # 简单估计每根K线的买卖压力
            o, h, l, c = df[['open', 'high', 'low', 'close']].iloc[i]

            # 计算K线总长度
            candle_range = h - l
            if candle_range == 0:
                bull_pwr = 0.5
                bear_pwr = 0.5
            else:
                # 如果是阳线，买压大；如果是阴线，卖压大
                if c > o:  # 阳线
                    bull_pwr = ((c - o) / candle_range) * 0.8 + ((h - c) / candle_range) * 0.2
                    bear_pwr = 1 - bull_pwr
                else:  # 阴线
                    bear_pwr = ((o - c) / candle_range) * 0.8 + ((c - l) / candle_range) * 0.2
                    bull_pwr = 1 - bear_pwr

            bull_power.append(bull_pwr)
            bear_power.append(bear_pwr)

        # 检测BISI模式（先买后卖）
        for i in range(3, len(bull_power)):
            # 检查是否有3根以上连续的买压占优
            if all(bull_power[j] > 0.6 for j in range(i - 3, i)):
                # 然后检查是否出现2根以上连续的卖压占优
                if all(bear_power[j] > 0.6 for j in range(i, min(i + 2, len(bear_power)))):
                    # 找到BISI模式
                    start_idx = len(df) - lookback + i - 3
                    end_idx = len(df) - lookback + i + 1

                    if start_idx >= 0 and end_idx < len(df):
                        bisi_pattern = {
                            "start_idx": start_idx,
                            "end_idx": end_idx,
                            "strength": sum(bull_power[i - 3:i]) - sum(bear_power[i - 3:i]),
                            "price_range": (df['low'].iloc[start_idx], df['high'].iloc[end_idx])
                        }
                        result["bisi"].append(bisi_pattern)
                        result["detected"] = True

        # 检测SIBI模式（先卖后买）
        for i in range(3, len(bear_power)):
            # 检查是否有3根以上连续的卖压占优
            if all(bear_power[j] > 0.6 for j in range(i - 3, i)):
                # 然后检查是否出现2根以上连续的买压占优
                if all(bull_power[j] > 0.6 for j in range(i, min(i + 2, len(bull_power)))):
                    # 找到SIBI模式
                    start_idx = len(df) - lookback + i - 3
                    end_idx = len(df) - lookback + i + 1

                    if start_idx >= 0 and end_idx < len(df):
                        sibi_pattern = {
                            "start_idx": start_idx,
                            "end_idx": end_idx,
                            "strength": sum(bear_power[i - 3:i]) - sum(bull_power[i - 3:i]),
                            "price_range": (df['high'].iloc[start_idx], df['low'].iloc[end_idx])
                        }
                        result["sibi"].append(sibi_pattern)
                        result["detected"] = True

        # 打印检测结果
        if result["bisi"]:
            print_colored(f"检测到 {len(result['bisi'])} 个BISI模式", Colors.GREEN)
        if result["sibi"]:
            print_colored(f"检测到 {len(result['sibi'])} 个SIBI模式", Colors.RED)

    except Exception as e:
        print_colored(f"检测不平衡模式出错: {e}", Colors.ERROR)

    return result