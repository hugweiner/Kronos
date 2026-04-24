import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import os
import time
import random


def get_stock_market(stock_code):
    """
    根据股票代码判断市场类型
    返回: 市场前缀 '0'-深交所, '1'-上交所
    """
    if stock_code.startswith(('0', '2', '3')):
        return '0'  # 深交所
    elif stock_code.startswith(('6', '9')):
        return '1'  # 上交所
    else:
        return '1'  # 默认上交所


def get_stock_data_eastmoney_all_history(stock_code="002354"):
    """
    使用东方财富网API获取股票所有历史数据
    """
    try:
        print(f"正在从东方财富网获取股票 {stock_code} 的全部历史数据...")

        # 获取市场类型
        market = get_stock_market(stock_code)
        secid = f"{market}.{stock_code}"

        # 使用东方财富API获取所有历史数据
        url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"

        # 设置足够早的起始日期（中国股市从1990年开始）
        start_date = "19900101"
        end_date = datetime.now().strftime('%Y%m%d')

        params = {
            'secid': secid,
            'fields1': 'f1,f2,f3,f4,f5,f6',
            'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
            'klt': '101',  # 日线
            'fqt': '1',  # 前复权
            'beg': start_date,
            'end': end_date,
            'lmt': '100000',  # 增大限制以确保获取完整历史数据（原50000可能不够）
            'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
            'cb': f'jQuery{random.randint(1000000, 9999999)}_{int(time.time() * 1000)}'
        }

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
            'Referer': 'https://quote.eastmoney.com/',
            'Accept': '*/*',
        }

        # 适当增加随机延迟，避免请求过于频繁被限流
        time.sleep(random.uniform(1.5, 3))

        response = requests.get(url, params=params, headers=headers, timeout=15)

        print(f"API响应状态码: {response.status_code}")

        if response.status_code == 200:
            # 处理JSONP响应
            response_text = response.text

            # 提取JSON数据（处理JSONP格式）
            if response_text.startswith('/**/'):
                response_text = response_text[4:]

            # 查找JSON数据的开始和结束位置
            start_idx = response_text.find('(')
            end_idx = response_text.rfind(')')

            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx + 1:end_idx]
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    print("❌ JSON解析失败，尝试直接解析...")
                    return parse_kline_data_directly_all_history(response_text, stock_code)
            else:
                print("❌ 无法找到JSON数据边界")
                return None

            print(f"API返回数据状态: {data.get('rc', 'N/A')}")

            if data and data.get('data') is not None:
                klines = data['data'].get('klines', [])
                print(f"获取到 {len(klines)} 条历史K线数据")

                if not klines:
                    print("⚠️ K线数据为空")
                    return None

                # 解析数据
            
