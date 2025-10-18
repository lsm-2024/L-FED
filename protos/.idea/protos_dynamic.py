import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


class DynamicThresholdStrategy:
    def __init__(self, initial_capital=10000, fee_rate=0.001, slippage=0.0005, coin_name="BTC"):
        """
        初始化动态阈值策略
        :param initial_capital: 初始资金
        :param fee_rate: 手续费率
        :param slippage: 滑点率
        :param coin_name: 币种名称
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.coin_name = coin_name
        self.position = 0  # 持仓数量
        self.trade_records = []  # 交易记录
        self.daily_balances = []  # 每日资产余额
        self.mape = None  # 平均绝对百分比误差

    def load_data(self, file_path):
        """
        从CSV文件读取历史数据和预测数据
        :param file_path: CSV文件路径
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 检查必要的列是否存在
            required_columns = ['date', 'close', 'original_preds']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"CSV文件中缺少必要的列: {col}")

            # 确保日期列格式正确
            df['date'] = pd.to_datetime(df['date'])

            # 计算MAPE (平均绝对百分比误差)
            # 排除真实值为0的情况，避免除零错误
            non_zero_mask = df['close'] != 0
            absolute_percent_errors = np.abs(
                (df.loc[non_zero_mask, 'close'] - df.loc[non_zero_mask, 'original_preds']) /
                df.loc[non_zero_mask, 'close'])
            self.mape = np.mean(absolute_percent_errors)

            # 重命名列以保持一致性
            df = df.rename(columns={
                'close': f'{self.coin_name}_actual',
                'original_preds': f'{self.coin_name}_pred'
            })

            # 添加MAPE列
            df[f'{self.coin_name}_mape'] = self.mape

            print(f"成功加载数据: {len(df)} 条记录")
            print(f"计算得到的MAPE: {self.mape:.2%}")

            return df
        except Exception as e:
            print(f"加载数据时出错: {str(e)}")
            raise

    def calculate_threshold(self):
        """根据MAPE计算动态阈值"""
        # 阈值设置为MAPE的1.5倍，留有一定安全边际
        return self.mape * 1.5

    def calculate_confidence(self):
        """根据MAPE计算预测置信度"""
        #return 1 - self.mape
        return 1 - 1.5*self.mape    #降低交易仓位，降低风险
    def execute_trade(self, date, action, price, amount, confidence):
        """
        执行交易
        :param date: 交易日期
        :param action: 操作类型：'buy'或'sell'
        :param price: 实际交易价格
        :param amount: 交易数量
        :param confidence: 预测置信度
        """
        # 计算包含手续费和滑点的实际价格
        if action == 'buy':
            # 买入时，实际价格 = 价格 * (1 + 手续费 + 滑点)
            actual_price = price * (1 + self.fee_rate + self.slippage)
            cost = amount * actual_price

            # 检查资金是否足够
            if cost > self.current_capital:
                return False  # 资金不足，交易失败

            # 更新资金和持仓
            self.current_capital -= cost
            self.position += amount

            # 记录交易
            self.trade_records.append({
                'date': date,
                'coin': self.coin_name,
                'action': action,
                'price': actual_price,
                'amount': amount,
                'cost': cost,
                'confidence': confidence,
                'profit': 0  # 买入时暂不计算利润
            })

        elif action == 'sell':
            # 卖出时，实际价格 = 价格 * (1 - 手续费 - 滑点)
            actual_price = price * (1 - self.fee_rate - self.slippage)
            revenue = amount * actual_price

            # 检查持仓是否足够
            if amount > self.position:
                return False  # 持仓不足，交易失败

            # 更新资金和持仓
            self.current_capital += revenue
            self.position -= amount

            # 计算利润（找到对应的买入记录）
            profit = 0
            remaining_amount = amount

            # 遍历交易记录，FIFO原则计算利润
            for record in reversed(self.trade_records):
                if record['coin'] == self.coin_name and record['action'] == 'buy' and record['profit'] == 0:
                    if remaining_amount <= 0:
                        break

                    sell_amount = min(remaining_amount, record['amount'])
                    buy_cost = sell_amount * record['price']
                    sell_revenue = sell_amount * actual_price
                    trade_profit = sell_revenue - buy_cost

                    profit += trade_profit
                    record['profit'] += trade_profit
                    remaining_amount -= sell_amount

            # 记录交易
            self.trade_records.append({
                'date': date,
                'coin': self.coin_name,
                'action': action,
                'price': actual_price,
                'amount': amount,
                'revenue': revenue,
                'confidence': confidence,
                'profit': profit
            })

        return True

    def run_backtest(self, data):
        """运行回测"""
        # 遍历每一天的数据
        for i in range(len(data) - 1):  # 最后一天无法预测下一天
            current_day = data.iloc[i]
            date = current_day['date']

            # 获取当前价格、预测价格
            current_price = current_day[f'{self.coin_name}_actual']
            predicted_price = current_day[f'{self.coin_name}_pred']

            # 计算预测涨跌幅
            predicted_change = (predicted_price - current_price) / current_price

            # 计算阈值和置信度
            threshold = self.calculate_threshold()
            confidence = self.calculate_confidence()

            # 确定操作信号
            if predicted_change > threshold:
                # 预测涨幅超过阈值，买入
                # 根据置信度确定仓位比例
                position_ratio = confidence
                investment_amount = self.current_capital * position_ratio
                # 计算可购买数量
                buy_amount = investment_amount / current_price

                # 执行买入
                self.execute_trade(
                    date=date,
                    action='buy',
                    price=current_price,
                    amount=buy_amount,
                    confidence=confidence
                )

            elif predicted_change < -threshold:
                # 预测跌幅超过阈值，卖出所有持仓
                if self.position > 0:
                    self.execute_trade(
                        date=date,
                        action='sell',
                        price=current_price,
                        amount=self.position,
                        confidence=confidence
                    )

            # 记录每日资产余额（现金 + 持仓市值）
            coin_value = self.position * current_price
            total_balance = self.current_capital + coin_value
            self.daily_balances.append({
                'date': date,
                'balance': total_balance
            })

        # 回测结束时，卖出所有持仓
        final_date = data.iloc[-1]['date']
        final_price = data.iloc[-1][f'{self.coin_name}_actual']

        if self.position > 0:
            self.execute_trade(
                date=final_date,
                action='sell',
                price=final_price,
                amount=self.position,
                confidence=1.0
            )

        # 计算最终资产
        final_balance = self.current_capital
        self.daily_balances.append({
            'date': final_date,
            'balance': final_balance
        })

        return self.analyze_results()

    def analyze_results(self):
        """分析回测结果"""
        # 过滤掉买入记录（只统计卖出的利润）
        sell_trades = [t for t in self.trade_records if t['action'] == 'sell']

        # 计算总交易次数
        total_trades = len(sell_trades)

        # 计算盈利交易次数和胜率
        profitable_trades = [t for t in sell_trades if t['profit'] > 0]
        win_rate = len(profitable_trades) / total_trades if total_trades > 0 else 0

        # 计算总收益和平均单次收益
        total_profit = sum(t['profit'] for t in sell_trades)
        avg_profit_per_trade = total_profit / total_trades if total_trades > 0 else 0

        # 计算最终回报率
        final_return = (self.current_capital - self.initial_capital) / self.initial_capital * 100
        balance_df = pd.DataFrame(self.daily_balances)
        balance_df['cum_max'] = balance_df['balance'].cummax()
        balance_df['drawdown'] = (balance_df['balance'] - balance_df['cum_max']) / balance_df['cum_max']
        max_drawdown = balance_df['drawdown'].min() * 100
        results = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_profit_per_trade': avg_profit_per_trade,
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'return_rate': final_return,
            'max_drawdown': max_drawdown,
            'mape': self.mape
        }

        return results

    def plot_results(self):
        """绘制回测结果图表"""
        # 转换为数据框以便绘图
        balance_df = pd.DataFrame(self.daily_balances)
        balance_df['date'] = pd.to_datetime(balance_df['date'])

        # 绘制资产曲线
        plt.figure(figsize=(12, 6))
        plt.plot(balance_df['date'], balance_df['balance'], label='资产总值')
        plt.axhline(y=self.initial_capital, color='r', linestyle='--', label='初始资金')
        plt.title(f'{self.coin_name}动态阈值策略资产曲线')
        plt.xlabel('日期')
        plt.ylabel('资产价值 (USDT)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 如果有交易记录，绘制收益分布
        sell_trades = [t for t in self.trade_records if t['action'] == 'sell']
        if sell_trades:
            profits = [t['profit'] for t in sell_trades]
            plt.figure(figsize=(12, 6))
            plt.hist(profits, bins=20, alpha=0.7)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title(f'{self.coin_name}交易收益分布')
            plt.xlabel('收益 (USDT)')
            plt.ylabel('交易次数')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    # 配置参数
    INITIAL_CAPITAL = 10000  # 初始资金
    FEE_RATE = 0.001  # 手续费率 0.1%
    SLIPPAGE = 0.0005  # 滑点率 0.05%
    COIN_NAME = "BTC"  # 币种名称
    CSV_FILE_PATH = "prediction_data.csv"  # CSV文件路径

    # 创建策略实例
    strategy = DynamicThresholdStrategy(
        initial_capital=INITIAL_CAPITAL,
        fee_rate=FEE_RATE,
        slippage=SLIPPAGE,
        coin_name=COIN_NAME
    )

    try:
        # 加载CSV数据
        data = strategy.load_data(CSV_FILE_PATH)

        # 运行回测
        results = strategy.run_backtest(data)

        # 输出回测结果
        print("\n===== 回测结果 =====")
        print(f"币种: {COIN_NAME}")
        print(f"初始资金: {results['initial_capital']:.2f} USDT")
        print(f"最终资金: {results['final_capital']:.2f} USDT")
        print(f"总收益率: {results['return_rate']:.2f}%")
        print(f"最大回撤: {results['max_drawdown']:.2f}%")
        print(f"MAPE: {results['mape']:.2%}")
        print(f"总交易次数: {results['total_trades']} 次")
        print(f"胜率: {results['win_rate']:.2%}")
        print(f"总收益: {results['total_profit']:.2f} USDT")
        print(f"平均单次收益: {results['avg_profit_per_trade']:.2f} USDT")

        # 绘制结果图表
        strategy.plot_results()
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
