import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# 添加项目路径以便导入自定义模块
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def ensure_output_directory(output_dir):
    """确保输出目录存在，如果不存在则创建"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ 创建输出目录: {output_dir}")
    return output_dir


def prepare_stock_data(csv_file_path, stock_code):
    """
    准备股票数据，转换为Kronos模型需要的格式

    参数:
    csv_file_path: CSV文件路径
    stock_code: 股票代码，用于显示信息

    返回:
    df: 处理后的DataFrame
    """
    print(f"正在加载和预处理股票 {stock_code} 数据...")

    # 读取CSV文件
    df = pd.read_csv(csv_file_path, encoding='utf-8-sig')

    # 检查数据列名并重命名为标准格式
    column_mapping = {
        '日期': 'timestamps',
        '开盘价': 'open',
        '最高价': 'high',
        '最低价': 'low',
        '收盘价': 'close',
        '成交量': 'volume',
        '成交额': 'amount'
    }

    # 只重命名存在的列
    actual_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=actual_mapping)

    # 确保时间戳列存在并转换为datetime格式
    if 'timestamps' not in df.columns:
        # 如果数据有日期索引，重置索引
        if df.index.name == '日期':
            df = df.reset_index()
            df = df.rename(columns={'日期': 'timestamps'})

    df['timestamps'] = pd.to_datetime(df['timestamps'])

    # 按时间排序
    df = df.sort_values('timestamps').reset_index(drop=True)

    # 去除收盘价为0或NaN的异常行（停牌日等）
    if 'close' in df.columns:
        before_count = len(df)
        df = df[df['close'].notna() & (df['close'] > 0)].reset_index(drop=True)
        removed = before_count - len(df)
        if removed > 0:
            print(f"⚠️  移除了 {removed} 条无效记录（收盘价为0或缺失）")

    print(f"✅ 数据加载完成，共 {len(df)} 条记录")
    print(f"时间范围: {df['timestamps'].min()} 到 {df['timestamps'].max()}")
    print(f"数据列: {df.columns.tolist()}")

    return df


def calculate_prediction_parameters(df, target_days=100):
    """
    根据目标预测天数计算合适的参数

    参数:
    df: 股票数据DataFrame
    target_days: 目标预测天数（自然日）

    返回:
    lookback: 回看期数
    pred_len: 预测期数
    """
    # 计算平均交易日数量（考虑节假日）
    total_days = (df['timestamps'].max() - df['timestamps'].min()).days
    trading_days = len(df)
    trading_ratio = trading_days / total_days if total_days > 0 else 0.7  # 交易日比例

    # 计算目标预测的交易日数量
    pred_trading_days = int(target_days * trading_ratio)

    # 设置回看期数为预测期数的2-3倍，但不超过数据总量的70%
    max_lookback = int(len(df) * 0.7)
    lookback = min(pred_trading_days * 2, max_lookback, len(df) - pred_trading_days)
    pred_len = min(pred_trading_days, len(df) - lookback)

    print(f"📊 参数计算:")
    print(f"  目标预测天数: {target_days} 天（自然日）")
    print(f"  预计交易日数量: {pred_trading_days} 天")
    print(f"  回看期数 (lookback): {lookback}")
    print(f"  预测期数 (pred_len): {pred_len}")

    return lookback, pred_len


def generate_future_dates_with_holidays(last_date, pred_len):
    """
    生成未来的交易日日期，考虑中国节假日

    参数:
    last_date: 最后一个历史数据的日期
    pred_len: 预测期数

    返回:
    future_dates: 未来的交易日日期列表
    """
    # 中国主要节假日（需要根据实际情况调整）
    holidays_2025 = [
        # 2025年国庆
