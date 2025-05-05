"""
入场时机计算模块
计算最佳入场时机、入场条件和预期价格
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
from logger_utils import Colors, print_colored


def calculate_entry_timing(df: pd.DataFrame, signal: str,
                           quality_score: float,
                           current_price: float) -> Dict[str, Any]:
    """
    计算最佳入场时机、条件和预期价格
    整合FVG和市场状态分析

    参数:
        df: 包含所有指标的DataFrame
        signal: 交易信号 ('BUY' 或 'SELL')
        quality_score: 质量评分
        current_price: 当前价格

    返回:
        包含入场时机详细信息的字典
    """
    print_colored("⏱️ 开始计算入场时机...", Colors.BLUE + Colors.BOLD)

    # 默认结果
    result = {
        "should_wait": True,
        "entry_type": "LIMIT",  # 默认使用限价单
        "entry_conditions": [],
        "expected_entry_price": current_price,
        "max_wait_time": 60,  # 默认最多等待60分钟
        "confidence": 0.5,
        "immediate_entry": False
    }

    try:
        # 导入FVG和市场状态模块
        from fvg_module import detect_fair_value_gap, detect_imbalance_patterns
        from market_state_module import classify_market_state

        # 检测FVG
        fvg_data = detect_fair_value_gap(df)

        # 分析市场状态
        market_state = classify_market_state(df)

        # 获取趋势数据
        trend_data = get_smc_trend_and_duration(df, None, None)[2]  # 返回趋势信息字典

        # 根据市场状态调整策略
        market_condition = market_state["state"]
        trend_direction = market_state["trend"]

        # 检查FVG和入场机会
        if signal == "BUY":
            # 检查是否在未填补的看涨FVG附近
            bullish_fvgs = [fvg for fvg in fvg_data if fvg['direction'] == 'UP' and not fvg['is_filled']]

            for fvg in bullish_fvgs:
                # 如果当前价格在FVG区域内或接近上边界
                if (fvg['lower_boundary'] <= current_price <= fvg['upper_boundary'] or
                        abs(current_price - fvg['upper_boundary']) / current_price < 0.005):
                    result["entry_conditions"].append(f"价格位于看涨FVG区域内/附近")
                    result["immediate_entry"] = True
                    result["should_wait"] = False
                    result["entry_type"] = "MARKET"
                    result["confidence"] += 0.2
                    break

            # 检查是否在EMA支撑位附近
            if 'EMA50' in df.columns:
                ema50 = df['EMA50'].iloc[-1]
                if abs(current_price - ema50) / current_price < 0.01 and current_price > ema50:
                    result["entry_conditions"].append(f"价格接近EMA50支撑位")
                    result["immediate_entry"] = True
                    result["should_wait"] = False
                    result["entry_type"] = "MARKET"
                    result["confidence"] += 0.15

            # 检查BISI模式（买入-卖出-买入不平衡）
            imbalance = detect_imbalance_patterns(df)
            if imbalance["detected"] and imbalance["sibi"]:
                result["entry_conditions"].append(f"检测到SIBI模式（卖出-买入不平衡）")
                result["immediate_entry"] = True
                result["should_wait"] = False
                result["entry_type"] = "MARKET"
                result["confidence"] += 0.25

            # 强趋势市场中的连续性突破
            if market_condition == "STRONG_UPTREND" and trend_direction == "UP":
                if 'Supertrend_Direction' in df.columns and df['Supertrend_Direction'].iloc[-1] > 0:
                    result["entry_conditions"].append(f"强上升趋势中的超级趋势确认")
                    result["immediate_entry"] = True
                    result["should_wait"] = False
                    result["entry_type"] = "MARKET"
                    result["confidence"] += 0.3

            # 弱趋势或中性市场等待回调
            elif market_condition in ["WEAK_UPTREND", "NEUTRAL", "RANGING"]:
                # 等待回调至支撑位
                pullback_target = 0.0

                # 查找支撑位
                if 'BB_Lower' in df.columns:
                    bb_lower = df['BB_Lower'].iloc[-1]
                    if bb_lower < current_price:
                        pullback_target = max(pullback_target, bb_lower)

                if 'EMA20' in df.columns:
                    ema20 = df['EMA20'].iloc[-1]
                    if ema20 < current_price:
                        pullback_target = max(pullback_target, ema20)

                # 如果找到有效的回调目标
                if pullback_target > 0 and abs(pullback_target - current_price) / current_price > 0.005:
                    result["entry_conditions"].append(f"等待回调至支撑位 {pullback_target:.6f}")
                    result["expected_entry_price"] = pullback_target
                    result["confidence"] += 0.1

        elif signal == "SELL":
            # 检查是否在未填补的看跌FVG附近
            bearish_fvgs = [fvg for fvg in fvg_data if fvg['direction'] == 'DOWN' and not fvg['is_filled']]

            for fvg in bearish_fvgs:
                # 如果当前价格在FVG区域内或接近下边界
                if (fvg['lower_boundary'] <= current_price <= fvg['upper_boundary'] or
                        abs(current_price - fvg['lower_boundary']) / current_price < 0.005):
                    result["entry_conditions"].append(f"价格位于看跌FVG区域内/附近")
                    result["immediate_entry"] = True
                    result["should_wait"] = False
                    result["entry_type"] = "MARKET"
                    result["confidence"] += 0.2
                    break

            # 检查是否在EMA阻力位附近
            if 'EMA50' in df.columns:
                ema50 = df['EMA50'].iloc[-1]
                if abs(current_price - ema50) / current_price < 0.01 and current_price < ema50:
                    result["entry_conditions"].append(f"价格接近EMA50阻力位")
                    result["immediate_entry"] = True
                    result["should_wait"] = False
                    result["entry_type"] = "MARKET"
                    result["confidence"] += 0.15

            # 检查BISI模式（买入-卖出不平衡）
            imbalance = detect_imbalance_patterns(df)
            if imbalance["detected"] and imbalance["bisi"]:
                result["entry_conditions"].append(f"检测到BISI模式（买入-卖出不平衡）")
                result["immediate_entry"] = True
                result["should_wait"] = False
                result["entry_type"] = "MARKET"
                result["confidence"] += 0.25

            # 强趋势市场中的连续性突破
            if market_condition == "STRONG_DOWNTREND" and trend_direction == "DOWN":
                if 'Supertrend_Direction' in df.columns and df['Supertrend_Direction'].iloc[-1] < 0:
                    result["entry_conditions"].append(f"强下降趋势中的超级趋势确认")
                    result["immediate_entry"] = True
                    result["should_wait"] = False
                    result["entry_type"] = "MARKET"
                    result["confidence"] += 0.3

            # 弱趋势或中性市场等待反弹
            elif market_condition in ["WEAK_DOWNTREND", "NEUTRAL", "RANGING"]:
                # 等待反弹至阻力位
                bounce_target = float('inf')

                # 查找阻力位
                if 'BB_Upper' in df.columns:
                    bb_upper = df['BB_Upper'].iloc[-1]
                    if bb_upper > current_price:
                        bounce_target = min(bounce_target, bb_upper)

                if 'EMA20' in df.columns:
                    ema20 = df['EMA20'].iloc[-1]
                    if ema20 > current_price:
                        bounce_target = min(bounce_target, ema20)

                # 如果找到有效的反弹目标
                if bounce_target < float('inf') and abs(bounce_target - current_price) / current_price > 0.005:
                    result["entry_conditions"].append(f"等待反弹至阻力位 {bounce_target:.6f}")
                    result["expected_entry_price"] = bounce_target
                    result["confidence"] += 0.1

        # 高质量评分直接入场
        if quality_score >= 8.5:
            result["entry_conditions"].append(f"高质量评分: {quality_score:.2f}，直接入场")
            result["immediate_entry"] = True
            result["should_wait"] = False
            result["entry_type"] = "MARKET"
            result["confidence"] = max(result["confidence"], 0.9)

        # 计算预期入场时间
        import datetime
        current_time = datetime.datetime.now()

        if result["should_wait"]:
            # 根据波动性估计到达目标价格的时间
            if 'ATR' in df.columns:
                atr = df['ATR'].iloc[-1]
                atr_hourly = atr * 4  # 假设15分钟K线，转换为小时ATR
                price_diff = abs(result["expected_entry_price"] - current_price)

                # 估计所需时间（小时）
                if atr_hourly > 0:
                    hours_needed = price_diff / atr_hourly
                    expected_minutes = int(hours_needed * 60)
                    expected_minutes = max(5, min(result["max_wait_time"], expected_minutes))
                else:
                    expected_minutes = result["max_wait_time"]
            else:
                expected_minutes = result["max_wait_time"]

            expected_entry_time = current_time + datetime.timedelta(minutes=expected_minutes)
            result["expected_entry_minutes"] = expected_minutes
            result["expected_entry_time"] = expected_entry_time.strftime("%H:%M:%S")
        else:
            result["expected_entry_minutes"] = 0
            result["expected_entry_time"] = current_time.strftime("%H:%M:%S") + " (立即)"

        # 检查是否有条件，如果没有则添加默认条件
        if not result["entry_conditions"]:
            if result["immediate_entry"]:
                result["entry_conditions"].append("综合分析建议立即市价入场")
            else:
                result["entry_conditions"].append(f"等待价格达到 {result['expected_entry_price']:.6f}")

        # 日志输出
        condition_color = Colors.GREEN if result["immediate_entry"] else Colors.YELLOW
        print_colored("入场时机分析:", Colors.INFO)
        for i, condition in enumerate(result["entry_conditions"], 1):
            print_colored(f"{i}. {condition}", condition_color)

        wait_msg = "立即入场" if result["immediate_entry"] else f"等待 {result['expected_entry_minutes']} 分钟"
        print_colored(f"建议入场时间: {result['expected_entry_time']} ({wait_msg})", Colors.INFO)
        print_colored(f"预期入场价格: {result['expected_entry_price']:.6f}", Colors.INFO)
        print_colored(f"入场类型: {result['entry_type']}", Colors.INFO)
        print_colored(f"入场置信度: {result['confidence']:.2f}", Colors.INFO)

        return result

    except Exception as e:
        print_colored(f"❌ 计算入场时机失败: {e}", Colors.ERROR)
        result["error"] = str(e)
        result["entry_conditions"] = ["计算出错，建议采用默认市价入场策略"]
        import datetime
        result["expected_entry_time"] = datetime.datetime.now().strftime("%H:%M:%S") + " (立即)"
        return result


def detect_breakout_conditions(df: pd.DataFrame, lookback: int = 20) -> Dict[str, Any]:
    """
    检测价格突破条件

    参数:
        df: 价格数据DataFrame
        lookback: 回溯检查的K线数量

    返回:
        突破信息字典
    """
    print_colored("🔍 检测价格突破条件...", Colors.BLUE)

    try:
        # 确保数据足够
        if len(df) < lookback + 5:
            return {
                "has_breakout": False,
                "direction": "NONE",
                "strength": 0,
                "description": "数据不足，无法检测突破"
            }

        result = {
            "has_breakout": False,
            "direction": "NONE",
            "strength": 0,
            "description": "",
            "breakout_details": []
        }

        # 获取最新价格和成交量
        current_price = df['close'].iloc[-1]
        current_volume = df['volume'].iloc[-1] if 'volume' in df.columns else 0

        # 计算近期价格区间
        lookback_df = df.iloc[-lookback:-1]
        recent_high = lookback_df['high'].max()
        recent_low = lookback_df['low'].min()

        # 计算平均成交量
        avg_volume = lookback_df['volume'].mean() if 'volume' in df.columns else 0
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # 检查技术指标
        has_bb = all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle'])
        has_pivot = 'Classic_PP' in df.columns

        breakout_details = []

        # 1. 检查价格区间突破
        upside_breakout = current_price > recent_high
        downside_breakout = current_price < recent_low

        if upside_breakout:
            strength = (current_price - recent_high) / recent_high * 100
            breakout_details.append({
                "type": "price_range",
                "direction": "UP",
                "description": f"价格突破近期高点 {recent_high:.6f}",
                "strength": strength
            })
        elif downside_breakout:
            strength = (recent_low - current_price) / recent_low * 100
            breakout_details.append({
                "type": "price_range",
                "direction": "DOWN",
                "description": f"价格跌破近期低点 {recent_low:.6f}",
                "strength": strength
            })

        # 2. 检查布林带突破
        if has_bb:
            bb_upper = df['BB_Upper'].iloc[-1]
            bb_lower = df['BB_Lower'].iloc[-1]
            bb_width = (bb_upper - bb_lower) / df['BB_Middle'].iloc[-1]

            # 上轨突破
            if current_price > bb_upper:
                bb_breakout_strength = (current_price - bb_upper) / bb_upper * 100
                bb_width_factor = max(1, bb_width * 10)  # 窄的布林带突破更有意义
                bb_strength = bb_breakout_strength * bb_width_factor

                breakout_details.append({
                    "type": "bollinger_band",
                    "direction": "UP",
                    "description": f"价格突破布林带上轨 {bb_upper:.6f}",
                    "strength": bb_strength
                })

            # 下轨突破
            elif current_price < bb_lower:
                bb_breakout_strength = (bb_lower - current_price) / bb_lower * 100
                bb_width_factor = max(1, bb_width * 10)
                bb_strength = bb_breakout_strength * bb_width_factor

                breakout_details.append({
                    "type": "bollinger_band",
                    "direction": "DOWN",
                    "description": f"价格跌破布林带下轨 {bb_lower:.6f}",
                    "strength": bb_strength
                })

        # 3. 检查支点突破
        if has_pivot:
            pivot = df['Classic_PP'].iloc[-1]
            r1 = df['Classic_R1'].iloc[-1]
            s1 = df['Classic_S1'].iloc[-1]

            # 阻力突破
            if df['close'].iloc[-2] <= r1 and current_price > r1:
                pivot_strength = (current_price - r1) / r1 * 100
                breakout_details.append({
                    "type": "pivot_point",
                    "direction": "UP",
                    "description": f"价格突破R1阻力位 {r1:.6f}",
                    "strength": pivot_strength
                })

            # 支撑跌破
            elif df['close'].iloc[-2] >= s1 and current_price < s1:
                pivot_strength = (s1 - current_price) / s1 * 100
                breakout_details.append({
                    "type": "pivot_point",
                    "direction": "DOWN",
                    "description": f"价格跌破S1支撑位 {s1:.6f}",
                    "strength": pivot_strength
                })

        # 4. 检查动量指标
        if 'RSI' in df.columns:
            rsi = df['RSI'].iloc[-1]
            prev_rsi = df['RSI'].iloc[-2]

            if prev_rsi < 30 and rsi > 30:
                breakout_details.append({
                    "type": "indicator",
                    "direction": "UP",
                    "description": f"RSI从超卖区反弹 ({prev_rsi:.1f} -> {rsi:.1f})",
                    "strength": (rsi - prev_rsi) / 2
                })
            elif prev_rsi > 70 and rsi < 70:
                breakout_details.append({
                    "type": "indicator",
                    "direction": "DOWN",
                    "description": f"RSI从超买区回落 ({prev_rsi:.1f} -> {rsi:.1f})",
                    "strength": (prev_rsi - rsi) / 2
                })

        # 汇总结果
        if breakout_details:
            # 过滤出强度最高的突破
            strongest_breakout = max(breakout_details, key=lambda x: x.get("strength", 0))
            result["has_breakout"] = True
            result["direction"] = strongest_breakout["direction"]
            result["strength"] = strongest_breakout["strength"]
            result["description"] = strongest_breakout["description"]
            result["breakout_details"] = breakout_details

            # 考虑成交量
            if volume_ratio > 1.5:
                result["strength"] *= 1.2
                result["description"] += f"，成交量放大({volume_ratio:.1f}倍)"

            print_colored(f"检测到{result['direction']}方向突破:",
                          Colors.GREEN if result['direction'] == 'UP' else Colors.RED)
            print_colored(f"描述: {result['description']}", Colors.INFO)
            print_colored(f"强度: {result['strength']:.2f}", Colors.INFO)

            for detail in breakout_details:
                detail_dir = detail["direction"]
                detail_color = Colors.GREEN if detail_dir == "UP" else Colors.RED
                print_colored(
                    f"- {detail['type']}: {detail_color}{detail['description']}{Colors.RESET}, 强度: {detail['strength']:.2f}",
                    Colors.INFO)
        else:
            print_colored("未检测到明显突破", Colors.YELLOW)

        return result
    except Exception as e:
        print_colored(f"❌ 检测突破条件失败: {e}", Colors.ERROR)
        return {
            "has_breakout": False,
            "direction": "NONE",
            "strength": 0,
            "description": f"检测出错: {str(e)}",
            "error": str(e)
        }


def estimate_entry_execution_price(current_price: float, signal: str,
                                   order_type: str, market_impact: float = 0.001) -> float:
    """
    估计实际入场执行价格，考虑市场冲击和滑点

    参数:
        current_price: 当前价格
        signal: 交易信号 ('BUY' 或 'SELL')
        order_type: 订单类型 ('MARKET' 或 'LIMIT')
        market_impact: 市场冲击系数

    返回:
        估计的执行价格
    """
    if order_type == "LIMIT":
        # 限价单通常以指定价格成交
        return current_price

    # 市价单会有滑点
    if signal == "BUY":
        # 买入时价格通常会略高于当前价
        execution_price = current_price * (1 + market_impact)
    else:  # SELL
        # 卖出时价格通常会略低于当前价
        execution_price = current_price * (1 - market_impact)

    return execution_price