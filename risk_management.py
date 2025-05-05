"""
风险管理模块
提供考虑杠杆的止损计算、高级SMC止损策略以及风险控制功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from logger_utils import Colors, print_colored
from indicators_module import (
    find_swing_points,
    calculate_fibonacci_retracements,
    get_smc_trend_and_duration
)

from indicators_module import get_smc_trend_and_duration, find_swing_points
from logger_utils import Colors, print_colored

def calculate_leveraged_stop_loss(entry_price: float, leverage: int,
                                  base_stop_loss_pct: float, side: str = "BUY") -> float:
    """
    考虑杠杆的止损计算

    参数:
        entry_price: 入场价格
        leverage: 杠杆倍数
        base_stop_loss_pct: 基础止损百分比 (小数形式，如0.03表示3%)
        side: 交易方向 ("BUY" 或 "SELL")

    返回:
        调整后的止损价格
    """
    # 杠杆越高，容忍度越低
    adjusted_stop_loss_pct = base_stop_loss_pct / (leverage ** 0.5)

    # 根据交易方向计算止损价格
    if side.upper() == "BUY":
        stop_loss_price = entry_price * (1 - adjusted_stop_loss_pct)
    else:  # SELL
        stop_loss_price = entry_price * (1 + adjusted_stop_loss_pct)

    print_colored("🔍 杠杆止损计算:", Colors.BLUE)
    print_colored(f"入场价格: {entry_price:.6f}", Colors.INFO)
    print_colored(f"交易方向: {side}", Colors.INFO)
    print_colored(f"杠杆: {leverage}倍", Colors.INFO)
    print_colored(f"基础止损: {base_stop_loss_pct * 100:.2f}%", Colors.INFO)
    print_colored(f"调整后止损: {adjusted_stop_loss_pct * 100:.2f}%", Colors.INFO)
    print_colored(f"止损价格: {stop_loss_price:.6f}", Colors.INFO)

    return stop_loss_price


def calculate_dynamic_take_profit(entry_price: float, stop_loss: float,
                                  min_risk_reward: float = 2.0, side: str = "BUY") -> float:
    """
    基于风险回报比计算动态止盈位

    参数:
        entry_price: 入场价格
        stop_loss: 止损价格
        min_risk_reward: 最小风险回报比，默认2.0
        side: 交易方向 ("BUY" 或 "SELL")

    返回:
        止盈价格
    """
    # 计算风险（基于实际价格，而非百分比）
    if side.upper() == "BUY":
        risk = entry_price - stop_loss
        # 根据风险回报比计算止盈
        take_profit = entry_price + (risk * min_risk_reward)
    else:  # SELL
        risk = stop_loss - entry_price
        # 根据风险回报比计算止盈
        take_profit = entry_price - (risk * min_risk_reward)

    print_colored("📊 动态止盈计算:", Colors.BLUE)
    print_colored(f"入场价格: {entry_price:.6f}", Colors.INFO)
    print_colored(f"止损价格: {stop_loss:.6f}", Colors.INFO)
    print_colored(f"风险金额: {risk:.6f}", Colors.INFO)
    print_colored(f"风险回报比: {min_risk_reward:.1f}", Colors.INFO)
    print_colored(f"止盈价格: {take_profit:.6f}", Colors.INFO)

    return take_profit


def advanced_smc_stop_loss(df: pd.DataFrame, entry_price: float, leverage: int,
                           side: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    SMC增强止损策略，结合市场结构、杠杆和趋势

    参数:
        df: 价格数据
        entry_price: 入场价格
        leverage: 杠杆倍数
        side: 交易方向
        config: 配置参数

    返回:
        包含止损、止盈价格和其他信息的字典
    """
    print_colored("⚙️ 计算SMC增强止损策略", Colors.BLUE + Colors.BOLD)

    try:
        # 确保df包含足够数据
        if df is None or len(df) < 20:
            print_colored("⚠️ 数据不足，无法使用SMC止损策略", Colors.WARNING)
            # 使用默认止损（基于杠杆）
            default_stop_pct = 0.03  # 默认3%止损
            stop_loss = calculate_leveraged_stop_loss(entry_price, leverage, default_stop_pct, side)
            take_profit = calculate_dynamic_take_profit(entry_price, stop_loss, 2.0, side)

            return {
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "method": "default_leveraged",
                "risk_reward_ratio": 2.0
            }

        # 市场结构止损 - 使用摆动点
        swing_highs, swing_lows = find_swing_points(df)

        # 趋势分析
        trend, _, trend_info = get_smc_trend_and_duration(df)

        # 斐波那契回撤位
        fib_levels = calculate_fibonacci_retracements(df)

        # 当前价格
        current_price = df['close'].iloc[-1]

        # 确定基础止损位
        if side.upper() == "BUY":
            # 做多止损策略
            # 1. 尝试使用最近的摆动低点
            relevant_swings = [low for low in swing_lows if low < entry_price]
            structure_stop = max(relevant_swings) if relevant_swings else None

            # 2. 尝试使用斐波那契回撤位
            relevant_fibs = [level for level in fib_levels if level < entry_price]
            fib_stop = max(relevant_fibs) if relevant_fibs else None

            # 3. 默认百分比止损
            default_stop = entry_price * 0.97  # 默认3%止损

            # 选择最合适的止损
            if structure_stop and structure_stop > entry_price * 0.90:  # 不要让止损太远
                base_stop = structure_stop
                stop_method = "structure"
            elif fib_stop and fib_stop > entry_price * 0.90:
                base_stop = fib_stop
                stop_method = "fibonacci"
            else:
                base_stop = default_stop
                stop_method = "percentage"

        else:  # SELL
            # 做空止损策略
            # 1. 尝试使用最近的摆动高点
            relevant_swings = [high for high in swing_highs if high > entry_price]
            structure_stop = min(relevant_swings) if relevant_swings else None

            # 2. 尝试使用斐波那契回撤位
            relevant_fibs = [level for level in fib_levels if level > entry_price]
            fib_stop = min(relevant_fibs) if relevant_fibs else None

            # 3. 默认百分比止损
            default_stop = entry_price * 1.03  # 默认3%止损

            # 选择最合适的止损
            if structure_stop and structure_stop < entry_price * 1.10:  # 不要让止损太远
                base_stop = structure_stop
                stop_method = "structure"
            elif fib_stop and fib_stop < entry_price * 1.10:
                base_stop = fib_stop
                stop_method = "fibonacci"
            else:
                base_stop = default_stop
                stop_method = "percentage"

        # 计算止损百分比
        stop_loss_pct = abs(base_stop - entry_price) / entry_price

        # 根据杠杆调整止损
        leveraged_stop_loss = calculate_leveraged_stop_loss(
            entry_price,
            leverage,
            stop_loss_pct,
            side
        )

        # 根据趋势置信度调整风险回报比
        if trend_info["confidence"] == "高":
            risk_reward_ratio = 3.0
        elif trend_info["confidence"] == "中高":
            risk_reward_ratio = 2.5
        elif trend_info["confidence"] == "中":
            risk_reward_ratio = 2.0
        else:
            risk_reward_ratio = 1.5

        # 计算止盈
        take_profit = calculate_dynamic_take_profit(
            entry_price,
            leveraged_stop_loss,
            risk_reward_ratio,
            side
        )

        # 构建结果
        result = {
            "stop_loss": leveraged_stop_loss,
            "take_profit": take_profit,
            "method": stop_method,
            "base_stop": base_stop,
            "stop_loss_pct": stop_loss_pct * 100,  # 转为百分比显示
            "risk_reward_ratio": risk_reward_ratio,
            "trend": trend,
            "trend_confidence": trend_info["confidence"]
        }

        print_colored(f"SMC止损方法: {stop_method}", Colors.INFO)
        print_colored(f"基础止损价格: {base_stop:.6f} ({stop_loss_pct * 100:.2f}%)", Colors.INFO)
        print_colored(f"杠杆调整后止损: {leveraged_stop_loss:.6f}", Colors.INFO)
        print_colored(f"止盈价格: {take_profit:.6f}", Colors.INFO)
        print_colored(f"风险回报比: {risk_reward_ratio:.1f}", Colors.INFO)

        return result
    except Exception as e:
        print_colored(f"❌ 计算SMC止损失败: {e}", Colors.ERROR)
        # 使用默认止损（基于杠杆）
        default_stop_pct = 0.03  # 默认3%止损
        stop_loss = calculate_leveraged_stop_loss(entry_price, leverage, default_stop_pct, side)
        take_profit = calculate_dynamic_take_profit(entry_price, stop_loss, 2.0, side)

        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "method": "default_leveraged",
            "risk_reward_ratio": 2.0,
            "error": str(e)
        }


def calculate_trailing_stop_params(quality_score: float, trend: str,
                                   market_conditions: Dict[str, Any]) -> Dict[str, float]:
    """
    根据质量评分和市场情况计算适合的移动止损参数

    参数:
        quality_score: 质量评分 (0-10)
        trend: 市场趋势 ("UP", "DOWN", "NEUTRAL")
        market_conditions: 市场环境信息

    返回:
        包含移动止损参数的字典
    """
    # 基础激活百分比
    if quality_score >= 8.0:
        activation_pct = 2.0  # 高质量信号，快速激活移动止损
    elif quality_score >= 6.0:
        activation_pct = 3.0  # 中等质量信号
    else:
        activation_pct = 4.0  # 较低质量信号，需要更多确认

    # 基础回调百分比
    if quality_score >= 8.0:
        callback_pct = 1.0  # 高质量信号，紧密跟踪
    elif quality_score >= 6.0:
        callback_pct = 1.5  # 中等质量信号
    else:
        callback_pct = 2.0  # 较低质量信号，更宽松的跟踪

    # 根据趋势调整
    if trend == "UP" or trend == "DOWN":
        # 明确趋势，可以更紧密地跟踪
        callback_pct *= 0.8
    else:
        # 中性趋势，需要更宽松的跟踪
        callback_pct *= 1.2
        activation_pct *= 1.2

    # 根据市场条件调整
    if "environment" in market_conditions:
        env = market_conditions["environment"]
        if env == 'trending':
            # 趋势市场，可以更紧密地跟踪
            callback_pct *= 0.8
        elif env == 'ranging':
            # 震荡市场，需要更宽松的跟踪
            callback_pct *= 1.5
            activation_pct *= 1.3
        elif env == 'breakout':
            # 突破市场，快速激活但宽松跟踪
            activation_pct *= 0.7
            callback_pct *= 1.2
        elif env == 'extreme_volatility':
            # 极端波动市场，非常宽松的跟踪
            callback_pct *= 2.0
            activation_pct *= 1.5

    # 确保值在合理范围内
    activation_pct = max(1.0, min(10.0, activation_pct))
    callback_pct = max(0.5, min(5.0, callback_pct))

    print_colored("🔄 移动止损参数:", Colors.BLUE)
    print_colored(f"激活比例: {activation_pct:.1f}%", Colors.INFO)
    print_colored(f"回撤比例: {callback_pct:.1f}%", Colors.INFO)

    return {
        "activation_pct": activation_pct,
        "callback_pct": callback_pct,
        "quality_score": quality_score,
        "trend": trend
    }


def calculate_position_size(account_balance: float, entry_price: float, stop_loss: float,
                            max_risk_percent: float = 2.0, leverage: int = 1) -> Dict[str, float]:
    """
    计算基于风险的仓位大小

    参数:
        account_balance: 账户余额
        entry_price: 入场价格
        stop_loss: 止损价格
        max_risk_percent: 最大风险比例（占账户的百分比）
        leverage: 杠杆倍数

    返回:
        包含仓位信息的字典
    """
    # 每单位的风险（价格差）
    unit_risk = abs(entry_price - stop_loss)

    # 账户可承受的风险金额
    max_risk_amount = account_balance * (max_risk_percent / 100)

    # 计算仓位规模（单位）
    position_size = max_risk_amount / unit_risk

    # 考虑杠杆
    leveraged_position_size = position_size * leverage

    # 计算仓位价值
    position_value = leveraged_position_size * entry_price

    # 计算实际风险
    actual_risk_amount = unit_risk * (position_value / entry_price / leverage)
    actual_risk_percent = (actual_risk_amount / account_balance) * 100

    print_colored("📊 仓位规模计算:", Colors.BLUE)
    print_colored(f"账户余额: {account_balance:.2f}", Colors.INFO)
    print_colored(f"入场价格: {entry_price:.6f}", Colors.INFO)
    print_colored(f"止损价格: {stop_loss:.6f}", Colors.INFO)
    print_colored(f"单位风险: {unit_risk:.6f}", Colors.INFO)
    print_colored(f"最大风险: {max_risk_percent:.1f}% (金额: {max_risk_amount:.2f})", Colors.INFO)
    print_colored(f"杠杆: {leverage}倍", Colors.INFO)
    print_colored(f"仓位规模: {leveraged_position_size:.6f} 单位", Colors.INFO)
    print_colored(f"仓位价值: {position_value:.2f}", Colors.INFO)
    print_colored(f"实际风险: {actual_risk_percent:.2f}% (金额: {actual_risk_amount:.2f})", Colors.INFO)

    return {
        "position_size": leveraged_position_size,
        "position_value": position_value,
        "risk_amount": actual_risk_amount,
        "risk_percent": actual_risk_percent,
        "unit_risk": unit_risk,
        "leverage": leverage
    }


def adaptive_risk_management(df: pd.DataFrame, account_balance: float, quality_score: float,
                             side: str, leverage: int = 1) -> Dict[str, Any]:
    """
    自适应风险管理系统，根据市场条件、质量评分和账户规模调整仓位和止损

    参数:
        df: 价格数据
        account_balance: 账户余额
        quality_score: 质量评分 (0-10)
        side: 交易方向 ("BUY" 或 "SELL")
        leverage: 杠杆倍数

    返回:
        完整风险管理参数和建议
    """
    print_colored("🛡️ 自适应风险管理分析", Colors.BLUE + Colors.BOLD)

    try:
        # 当前价格
        current_price = df['close'].iloc[-1]

        # 市场趋势分析
        trend, _, trend_info = get_smc_trend_and_duration(df)

        # 基于质量评分调整风险 - 增加风险百分比
        if quality_score >= 8.0:
            max_risk_percent = 3.0  # 高质量信号，可接受更高风险 (从2.0改为3.0)
        elif quality_score >= 6.0:
            max_risk_percent = 2.5  # 中等质量信号 (从1.5改为2.5)
        else:
            max_risk_percent = 2.0  # 低质量信号，降低风险 (从1.0改为2.0)

        # 基于趋势调整风险
        if trend_info["confidence"] == "高":
            max_risk_percent *= 1.2  # 高置信度趋势，增加风险
        elif trend_info["confidence"] == "低":
            max_risk_percent *= 0.8  # 低置信度趋势，降低风险

        # 考虑Vortex指标调整风险
        vortex_adjustment = 1.0
        if 'VI_plus' in df.columns and 'VI_minus' in df.columns:
            vi_plus = df['VI_plus'].iloc[-1]
            vi_minus = df['VI_minus'].iloc[-1]
            vi_diff = abs(df['VI_diff'].iloc[-1]) if 'VI_diff' in df.columns else abs(vi_plus - vi_minus)

            # 计算趋势一致性
            vortex_trend = 1 if vi_plus > vi_minus else -1
            trade_trend = 1 if side.upper() == "BUY" else -1

            # 方向一致时增加风险接受度
            if vortex_trend == trade_trend:
                strength = vi_diff * 10  # 放大差值用于评估强度
                if strength > 1.5:
                    vortex_adjustment = 1.2  # 强趋势增加20%风险接受度
                    print_colored(f"Vortex指标显示强烈趋势与交易方向一致，风险调整: +20%", Colors.GREEN)
                elif strength > 0.8:
                    vortex_adjustment = 1.1  # 中等趋势增加10%风险接受度
                    print_colored(f"Vortex指标与交易方向一致，风险调整: +10%", Colors.GREEN)
            # 方向不一致时降低风险接受度
            else:
                vortex_adjustment = 0.8  # 降低20%风险接受度
                print_colored(f"Vortex指标与交易方向不一致，风险调整: -20%", Colors.WARNING)

            # 检查是否有交叉信号
            cross_up = df['Vortex_Cross_Up'].iloc[-1] if 'Vortex_Cross_Up' in df.columns else 0
            cross_down = df['Vortex_Cross_Down'].iloc[-1] if 'Vortex_Cross_Down' in df.columns else 0

            if (cross_up and side.upper() == "BUY") or (cross_down and side.upper() == "SELL"):
                vortex_adjustment *= 1.1  # 交叉信号再增加10%
                print_colored(f"Vortex交叉信号与交易方向一致，额外风险调整: +10%", Colors.GREEN)

        # 应用Vortex调整到风险百分比
        max_risk_percent *= vortex_adjustment

        # 计算止损点
        stop_loss_result = advanced_smc_stop_loss(df, current_price, leverage, side)
        stop_loss = stop_loss_result["stop_loss"]
        take_profit = stop_loss_result["take_profit"]

        # 计算仓位规模
        position_result = calculate_position_size(
            account_balance,
            current_price,
            stop_loss,
            max_risk_percent,
            leverage
        )

        # 新增：确保名义价值足够
        min_position_value = 50.0  # 最小50美元
        if position_result["position_value"] < min_position_value:
            # 调整仓位大小确保至少达到最小名义价值
            position_size = min_position_value / current_price
            position_value = min_position_value

            # 更新仓位信息
            position_result["position_size"] = position_size
            position_result["position_value"] = position_value

            print_colored(f"⚠️ 仓位价值过小，已调整为最小值: {min_position_value} USDC", Colors.WARNING)

        # 计算移动止损参数
        market_conditions = {"environment": "trending" if trend != "NEUTRAL" else "ranging"}
        trailing_stop_params = calculate_trailing_stop_params(quality_score, trend, market_conditions)

        # 风险状态评估
        risk_level = "低" if position_result["risk_percent"] <= 1.0 else "中" if position_result[
                                                                                     "risk_percent"] <= 2.0 else "高"

        # 汇总结果
        result = {
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size": position_result["position_size"],
            "position_value": position_result["position_value"],
            "max_risk_percent": max_risk_percent,
            "actual_risk_percent": position_result["risk_percent"],
            "risk_level": risk_level,
            "leverage": leverage,
            "risk_reward_ratio": stop_loss_result.get("risk_reward_ratio", 0),
            "trailing_stop": trailing_stop_params,
            "quality_score": quality_score,
            "trend": trend,
            "trend_confidence": trend_info["confidence"],
            "vortex_adjustment": vortex_adjustment
        }

        # 判断是否应该执行交易
        if risk_level == "高" and quality_score < 7.0:
            result["recommendation"] = "AVOID"
            result["recommendation_reason"] = "风险较高但质量评分不足"
        elif leverage > 10 and quality_score < 8.0:
            result["recommendation"] = "REDUCE_LEVERAGE"
            result["recommendation_reason"] = "杠杆过高但质量评分不足，建议降低杠杆"
        elif position_result["position_value"] < 10.0:  # 仓位价值过小
            result["recommendation"] = "INCREASE_SIZE"
            result["recommendation_reason"] = "仓位价值过小，建议增加仓位或选择其他交易机会"
        else:
            result["recommendation"] = "PROCEED"
            result["recommendation_reason"] = "风险参数合理，可以执行交易"

        # 打印结果摘要
        print_colored(f"风险等级: {risk_level}", Colors.INFO)
        print_colored(f"最大风险: {max_risk_percent:.2f}%, 实际风险: {position_result['risk_percent']:.2f}%",
                      Colors.INFO)
        print_colored(f"建议: {result['recommendation']}, 原因: {result['recommendation_reason']}", Colors.INFO)

        return result
    except Exception as e:
        print_colored(f"❌ 风险管理分析失败: {e}", Colors.ERROR)
        return {
            "error": str(e),
            "recommendation": "AVOID",
            "recommendation_reason": "风险分析失败，建议避免交易"
        }


# 添加到文件: risk_management.py
# 文件末尾添加以下函数:

def calculate_reversal_based_take_profit(df: pd.DataFrame, entry_price: float, direction: str, stop_loss: float,
                                         fvg_data: List[Dict[str, Any]], trend_data: Dict[str, Any],
                                         market_state: Dict[str, Any], min_reward_ratio: float = 1.5) -> Dict[str, Any]:
    """
    基于反转检测的动态止盈系统

    参数:
        df: 价格数据
        entry_price: 入场价格
        direction: 交易方向 ("LONG" 或 "SHORT")
        stop_loss: 止损价格
        fvg_data: FVG信息
        trend_data: 趋势信息
        market_state: 市场状态信息
        min_reward_ratio: 最小风险回报比

    返回:
        止盈信息字典
    """
    from market_state_module import detect_market_reversal

    # 计算基础的ATR止盈目标作为参考
    atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else (df['high'].mean() - df['low'].mean()) / 10

    # 根据市场状态调整ATR倍数
    if market_state["state"].startswith("STRONG_"):
        atr_multiplier = 2.5  # 强趋势使用更大倍数
    elif market_state["state"].startswith("VOLATILE_"):
        atr_multiplier = 2.0  # 波动性强使用中等倍数
    else:
        atr_multiplier = 1.5  # 其他情况使用较小倍数

    basic_target_distance = atr_multiplier * atr

    if direction == "LONG":
        basic_target = entry_price + basic_target_distance
    else:
        basic_target = entry_price - basic_target_distance

    # 确保至少有最小的风险回报比
    sl_distance = abs(entry_price - stop_loss)
    min_target_distance = sl_distance * min_reward_ratio

    if direction == "LONG":
        min_target = entry_price + min_target_distance
        basic_target = max(basic_target, min_target)
    else:
        min_target = entry_price - min_target_distance
        basic_target = min(basic_target, min_target)

    # 如果有FVG作为目标，调整目标位置
    fvg_target = None
    if fvg_data:
        if direction == "LONG":
            # 找到上方的看跌FVG作为目标
            bearish_fvgs = [fvg for fvg in fvg_data if fvg['direction'] == 'DOWN' and
                            not fvg['is_filled'] and fvg['lower_boundary'] > entry_price]

            if bearish_fvgs:
                # 选择最近的一个
                nearest_fvg = min(bearish_fvgs, key=lambda x: abs(x['lower_boundary'] - entry_price))
                fvg_target = nearest_fvg['lower_boundary']  # 使用下边界作为目标
        else:
            # 找到下方的看涨FVG作为目标
            bullish_fvgs = [fvg for fvg in fvg_data if fvg['direction'] == 'UP' and
                            not fvg['is_filled'] and fvg['upper_boundary'] < entry_price]

            if bullish_fvgs:
                # 选择最近的一个
                nearest_fvg = min(bullish_fvgs, key=lambda x: abs(x['upper_boundary'] - entry_price))
                fvg_target = nearest_fvg['upper_boundary']  # 使用上边界作为目标

    # 如果FVG目标有效，且满足最小风险回报比，则使用FVG目标
    if fvg_target is not None:
        fvg_reward_ratio = abs(fvg_target - entry_price) / sl_distance
        if fvg_reward_ratio >= min_reward_ratio:
            basic_target = fvg_target

    # 获取当前反转检测结果
    reversal_data = detect_market_reversal(df, fvg_data, trend_data, market_state)

    # 激活反转提前退出的阈值
    reversal_threshold = 0.65  # 需要较强的反转信号

    return {
        'basic_target': basic_target,
        'reversal_probability': reversal_data['probability'],
        'reversal_strength': reversal_data['strength'],
        'reversal_signals': reversal_data['signals'],
        'use_reversal_exit': reversal_data['probability'] >= reversal_threshold,
        'min_reward_ratio_target': min_target,
        'atr_multiplier': atr_multiplier,
        'atr_value': atr,
        'fvg_target': fvg_target
    }


def manage_take_profit(position: Dict[str, Any], current_price: float, df: pd.DataFrame,
                       fvg_data: List[Dict[str, Any]], trend_data: Dict[str, Any],
                       market_state: Dict[str, Any], min_reward_ratio: float = 1.5) -> Dict[str, Any]:
    """
    管理止盈逻辑，结合固定目标和反转检测

    参数:
        position: 持仓信息
        current_price: 当前价格
        df: 价格数据
        fvg_data: FVG信息
        trend_data: 趋势信息
        market_state: 市场状态信息
        min_reward_ratio: 最小风险回报比

    返回:
        止盈决策字典
    """
    # 获取持仓信息
    symbol = position['symbol']
    entry_price = position['entry_price']
    direction = position['position_side']
    stop_loss = position['current_stop_level']

    # 计算当前利润
    if direction == "LONG":
        current_profit_pct = (current_price - entry_price) / entry_price
    else:
        current_profit_pct = (entry_price - current_price) / entry_price

    # 获取反转检测结果
    tp_data = calculate_reversal_based_take_profit(
        df, entry_price, direction, stop_loss, fvg_data, trend_data, market_state, min_reward_ratio
    )

    # 止盈决策逻辑
    take_profit = False
    reason = ""

    # 情况1: 达到基本目标且有反转信号
    if ((direction == "LONG" and current_price >= tp_data['basic_target']) or
            (direction == "SHORT" and current_price <= tp_data['basic_target'])):
        if tp_data['reversal_probability'] >= 0.4:  # 较低的反转阈值
            take_profit = True
            reason = f"达到基本目标且检测到{tp_data['reversal_strength']}反转信号"

    # 情况2: 强烈反转信号出现
    elif tp_data['use_reversal_exit']:
        take_profit = True
        reason = f"检测到{tp_data['reversal_strength']}反转信号: {', '.join(tp_data['reversal_signals'][:2]) if tp_data['reversal_signals'] else '综合反转指标'}"

    # 情况3: 达到最小风险回报比，但接近支撑/阻力
    elif ((direction == "LONG" and current_price >= tp_data['min_reward_ratio_target']) or
          (direction == "SHORT" and current_price <= tp_data['min_reward_ratio_target'])):
        # 检查是否接近支撑/阻力
        is_near_sr = False

        # 检查是否接近EMA200
        if 'EMA200' in df.columns:
            ema200 = df['EMA200'].iloc[-1]
            distance_to_ema = abs(current_price - ema200) / current_price
            if distance_to_ema < 0.005:  # 0.5%以内
                is_near_sr = True
                reason = f"达到最小风险回报比({min_reward_ratio}R)且接近EMA200"

        # 检查是否接近摆动点
        swing_highs, swing_lows = find_swing_points(df)
        if direction == "LONG":
            for high in swing_highs:
                if abs(current_price - high) / current_price < 0.01:  # 1%以内
                    is_near_sr = True
                    reason = f"达到最小风险回报比({min_reward_ratio}R)且接近历史高点"
                    break
        else:
            for low in swing_lows:
                if abs(current_price - low) / current_price < 0.01:  # 1%以内
                    is_near_sr = True
                    reason = f"达到最小风险回报比({min_reward_ratio}R)且接近历史低点"
                    break

        if is_near_sr:
            take_profit = True

    return {
        'take_profit': take_profit,
        'reason': reason,
        'current_profit_pct': current_profit_pct,
        'current_reward_ratio': abs(current_price - entry_price) / abs(entry_price - stop_loss),
        'reversal_probability': tp_data['reversal_probability'],
        'reversal_signals': tp_data['reversal_signals'],
        'basic_target': tp_data['basic_target'],
        'atr_target_multiplier': tp_data['atr_multiplier'],
        'atr_value': tp_data['atr_value'],
        'min_reward_ratio': min_reward_ratio,
        'fvg_target': tp_data['fvg_target']
    }


def optimize_entry_timing(df: pd.DataFrame, fvg_data: List[Dict[str, Any]],
                          market_state: Dict[str, Any], signal: str, quality_score: float,
                          current_price: float, timeframe: str = "15m") -> Dict[str, Any]:
    """
    基于15分钟K线优化入场时机

    参数:
        df: 价格数据
        fvg_data: FVG信息
        market_state: 市场状态信息
        signal: 交易信号 ('BUY' 或 'SELL')
        quality_score: 质量评分
        current_price: 当前价格
        timeframe: 时间框架

    返回:
        入场时机信息字典
    """
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
            from fvg_module import detect_imbalance_patterns
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
            from fvg_module import detect_imbalance_patterns
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
        print_colored(f"优化入场时机出错: {e}", Colors.ERROR)
        result["error"] = str(e)
        result["entry_conditions"] = ["计算出错，建议采用默认市价入场策略"]
        import datetime
        result["expected_entry_time"] = datetime.datetime.now().strftime("%H:%M:%S") + " (立即)"
        return result