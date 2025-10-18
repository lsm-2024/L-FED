import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


class CrossCurrencyArbitrage:
    def __init__(self, initial_capital=10000, fee_rate=0.001, slippage=0.0005,
                 correlation_window=30, correlation_threshold=0.1):
        """
        初始化跨币种套利策略
        :param initial_capital: 初始资金
        :param fee_rate: 手续费率
        :param slippage: 滑点率
        :param correlation_window: 计算相关性的窗口大小
        :param correlation_threshold: 相关性偏离阈值
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.correlation_window = correlation_window  # 计算相关性的窗口大小
        self.correlation_threshold = correlation_threshold  # 相关性偏离阈值

        # 持仓信息
        self.positions = {
            'BTC': 0,  # 比特币持仓数量
            'ETH': 0  # 以太坊持仓数量
        }

        # 交易记录和每日资产
        self.trade_records = []
        self.daily_balances = []

        # 策略相关数据
        self.combined_data = None
        self.avg_correlation = None
        self.volatility = {
            'BTC': None,
            'ETH': None
        }

    def load_data(self, btc_file_path, eth_file_path):
        """
        从CSV文件加载BTC和ETH的历史数据和预测数据
        :param btc_file_path: BTC数据CSV文件路径
        :param eth_file_path: ETH数据CSV文件路径
        """
        try:
            # 读取两个币种的数据
            btc_df = pd.read_csv(btc_file_path)
            eth_df = pd.read_csv(eth_file_path)

            # 检查必要的列是否存在
            required_columns = ['date', 'close', 'original_preds']
            for df, coin in [(btc_df, 'BTC'), (eth_df, 'ETH')]:
                for col in required_columns:
                    if col not in df.columns:
                        raise ValueError(f"{coin}数据文件中缺少必要的列: {col}")

            # 处理日期格式
            btc_df['date'] = pd.to_datetime(btc_df['date'])
            eth_df['date'] = pd.to_datetime(eth_df['date'])

            # 重命名列以便区分
            btc_df = btc_df.rename(columns={
                'close': 'BTC_close',
                'original_preds': 'BTC_pred'
            })

            eth_df = eth_df.rename(columns={
                'close': 'ETH_close',
                'original_preds': 'ETH_pred'
            })

            # 合并数据，按日期对齐
            combined_df = pd.merge(btc_df[['date', 'BTC_close', 'BTC_pred']],
                                   eth_df[['date', 'ETH_close', 'ETH_pred']],
                                   on='date', how='inner')

            # 计算收益率
            combined_df['BTC_return'] = combined_df['BTC_close'].pct_change()
            combined_df['ETH_return'] = combined_df['ETH_close'].pct_change()

            # 计算滚动相关性
            combined_df['correlation'] = combined_df['BTC_return'].rolling(
                window=self.correlation_window).corr(combined_df['ETH_return'])

            # 计算平均相关性（作为基准）
            self.avg_correlation = combined_df['correlation'].mean()

            # 计算年化波动率（基于252个交易日）
            self.volatility['BTC'] = combined_df['BTC_return'].std() * np.sqrt(252)
            self.volatility['ETH'] = combined_df['ETH_return'].std() * np.sqrt(252)

            print(f"成功加载数据: {len(combined_df)} 条共同日期记录")
            print(f"BTC与ETH的平均相关性: {self.avg_correlation:.4f}")
            print(f"BTC年化波动率: {self.volatility['BTC']:.2%}")
            print(f"ETH年化波动率: {self.volatility['ETH']:.2%}")

            self.combined_data = combined_df
            return combined_df

        except Exception as e:
            print(f"加载数据时出错: {str(e)}")
            raise

    def calculate_position_ratio(self):
        """根据波动率计算仓位配比"""
        # 仓位比 = BTC波动率 / ETH波动率
        ratio = self.volatility['BTC'] / self.volatility['ETH']
        return ratio

    def execute_trade(self, date, btc_price, eth_price, signal):
        """
        执行套利交易
        :param date: 交易日期
        :param btc_price: BTC当前价格
        :param eth_price: ETH当前价格
        :param signal: 交易信号: 1表示做多ETH做空BTC, -1表示做多BTC做空ETH
        """
        # 计算仓位配比
        position_ratio = self.calculate_position_ratio()

        # 确定总投资金额（使用当前资金的80%进行套利，留20%作为缓冲）
        investment_amount = self.current_capital * 0.8

        # 根据信号分配仓位
        if signal == 1:
            # 做多ETH，做空BTC
            eth_invest = investment_amount / (1 + position_ratio)
            btc_invest = investment_amount - eth_invest

            # 计算交易数量
            eth_amount = eth_invest / eth_price
            btc_amount = btc_invest / btc_price

            # 执行交易：买入ETH，卖出BTC（如果有持仓）
            # 先平掉现有相反持仓
            if self.positions['BTC'] > 0:
                self.sell('BTC', date, btc_price, self.positions['BTC'])
            if self.positions['ETH'] < 0:
                self.buy('ETH', date, eth_price, abs(self.positions['ETH']))

            # 建立新持仓（做空BTC用负数表示）
            self.buy('ETH', date, eth_price, eth_amount)
            self.sell('BTC', date, btc_price, btc_amount)  # 做空操作

        elif signal == -1:
            # 做多BTC，做空ETH
            btc_invest = investment_amount / (1 + position_ratio)
            eth_invest = investment_amount - btc_invest

            # 计算交易数量
            btc_amount = btc_invest / btc_price
            eth_amount = eth_invest / eth_price

            # 执行交易：买入BTC，卖出ETH（如果有持仓）
            # 先平掉现有相反持仓
            if self.positions['ETH'] > 0:
                self.sell('ETH', date, eth_price, self.positions['ETH'])
            if self.positions['BTC'] < 0:
                self.buy('BTC', date, btc_price, abs(self.positions['BTC']))

            # 建立新持仓（做空ETH用负数表示）
            self.buy('BTC', date, btc_price, btc_amount)
            self.sell('ETH', date, eth_price, eth_amount)  # 做空操作

        elif signal == 0:
            # 平仓信号，平掉所有持仓
            if self.positions['BTC'] != 0:
                if self.positions['BTC'] > 0:
                    self.sell('BTC', date, btc_price, self.positions['BTC'])
                else:
                    self.buy('BTC', date, btc_price, abs(self.positions['BTC']))

            if self.positions['ETH'] != 0:
                if self.positions['ETH'] > 0:
                    self.sell('ETH', date, eth_price, self.positions['ETH'])
                else:
                    self.buy('ETH', date, eth_price, abs(self.positions['ETH']))

    def buy(self, coin, date, price, amount):
        """买入操作（包括平空仓）"""
        # 买入时，实际价格 = 价格 * (1 + 手续费 + 滑点)
        actual_price = price * (1 + self.fee_rate + self.slippage)
        cost = amount * actual_price

        # 检查资金是否足够
        if cost > self.current_capital:
            return False  # 资金不足，交易失败

        # 更新资金和持仓
        self.current_capital -= cost

        # 如果是平空仓（原持仓为负）
        if self.positions[coin] < 0:
            cover_amount = min(amount, abs(self.positions[coin]))
            self.positions[coin] += cover_amount
            remaining_amount = amount - cover_amount

            # 记录平仓交易
            self.trade_records.append({
                'date': date,
                'coin': coin,
                'action': 'cover',  # 平仓
                'price': actual_price,
                'amount': cover_amount,
                'cost': cover_amount * actual_price,
                'profit': 0
            })

            # 如果还有剩余资金，建立多仓
            if remaining_amount > 0:
                self.positions[coin] += remaining_amount
                self.trade_records.append({
                    'date': date,
                    'coin': coin,
                    'action': 'buy',  # 买入
                    'price': actual_price,
                    'amount': remaining_amount,
                    'cost': remaining_amount * actual_price,
                    'profit': 0
                })
        else:
            # 直接建立多仓
            self.positions[coin] += amount
            self.trade_records.append({
                'date': date,
                'coin': coin,
                'action': 'buy',
                'price': actual_price,
                'amount': amount,
                'cost': cost,
                'profit': 0
            })

        return True

    def sell(self, coin, date, price, amount):
        """卖出操作（包括建立空仓）"""
        # 卖出时，实际价格 = 价格 * (1 - 手续费 - 滑点)
        actual_price = price * (1 - self.fee_rate - self.slippage)

        # 如果是平多仓（原持仓为正）
        if self.positions[coin] > 0:
            sell_amount = min(amount, self.positions[coin])
            revenue = sell_amount * actual_price

            # 计算利润
            profit = 0
            remaining_amount = sell_amount

            # 按FIFO原则计算利润
            for record in reversed(self.trade_records):
                if record['coin'] == coin and record['action'] in ['buy', 'cover'] and record['profit'] == 0:
                    if remaining_amount <= 0:
                        break

                    close_amount = min(remaining_amount, record['amount'])
                    buy_cost = close_amount * record['price']
                    sell_revenue = close_amount * actual_price
                    trade_profit = sell_revenue - buy_cost

                    profit += trade_profit
                    record['profit'] += trade_profit
                    remaining_amount -= close_amount

            # 更新资金和持仓
            self.current_capital += revenue
            self.positions[coin] -= sell_amount

            # 记录交易
            self.trade_records.append({
                'date': date,
                'coin': coin,
                'action': 'sell',
                'price': actual_price,
                'amount': sell_amount,
                'revenue': revenue,
                'profit': profit
            })

            # 如果还有剩余需要卖出的量，建立空仓（用负数表示）
            short_amount = amount - sell_amount
            if short_amount > 0:
                self.positions[coin] -= short_amount
                self.trade_records.append({
                    'date': date,
                    'coin': coin,
                    'action': 'short',  # 做空
                    'price': actual_price,
                    'amount': short_amount,
                    'revenue': short_amount * actual_price,  # 做空获得的资金
                    'profit': 0
                })
                self.current_capital += short_amount * actual_price  # 做空获得资金
        else:
            # 直接建立空仓
            self.positions[coin] -= amount
            revenue = amount * actual_price
            self.current_capital += revenue
            self.trade_records.append({
                'date': date,
                'coin': coin,
                'action': 'short',
                'price': actual_price,
                'amount': amount,
                'revenue': revenue,
                'profit': 0
            })

        return True

    def run_backtest(self):
        """运行回测"""
        if self.combined_data is None:
            raise ValueError("请先加载数据")

        # 遍历每一天的数据
        for i in range(self.correlation_window, len(self.combined_data) - 1):
            current_day = self.combined_data.iloc[i]
            date = current_day['date']
            btc_price = current_day['BTC_close']
            eth_price = current_day['ETH_close']
            current_correlation = current_day['correlation']

            # 确定交易信号
            if pd.isna(current_correlation):
                signal = 0  # 相关性数据不足，不交易
            elif current_correlation > self.avg_correlation + self.correlation_threshold:
                signal = 1  # 相关性过高，做多ETH，做空BTC
            elif current_correlation < self.avg_correlation - self.correlation_threshold:
                signal = -1  # 相关性过低，做多BTC，做空ETH
            else:
                signal = 0  # 相关性正常，不交易或平仓

            # 执行交易
            if signal != 0:
                self.execute_trade(date, btc_price, eth_price, signal)
            elif self.positions['BTC'] != 0 or self.positions['ETH'] != 0:
                # 如果相关性回到正常范围且有持仓，则平仓
                self.execute_trade(date, btc_price, eth_price, 0)

            # 记录每日资产余额
            btc_value = self.positions['BTC'] * btc_price
            eth_value = self.positions['ETH'] * eth_price
            total_balance = self.current_capital + btc_value + eth_value
            self.daily_balances.append({
                'date': date,
                'balance': total_balance,
                'correlation': current_correlation,
                'signal': signal
            })

        # 回测结束时，平掉所有持仓
        final_day = self.combined_data.iloc[-1]
        final_date = final_day['date']
        final_btc_price = final_day['BTC_close']
        final_eth_price = final_day['ETH_close']

        self.execute_trade(final_date, final_btc_price, final_eth_price, 0)

        # 记录最终资产
        final_balance = self.current_capital + self.positions['BTC'] * final_btc_price + self.positions[
            'ETH'] * final_eth_price
        self.daily_balances.append({
            'date': final_date,
            'balance': final_balance,
            'correlation': self.combined_data.iloc[-1]['correlation'],
            'signal': 0
        })

        return self.analyze_results()

    def analyze_results(self):
        """分析回测结果"""
        # 过滤出产生利润的交易记录
        profitable_trades = [t for t in self.trade_records if t['action'] in ['sell', 'cover'] and t['profit'] != 0]

        # 计算总交易次数
        total_trades = len(profitable_trades)

        # 计算盈利交易次数和胜率
        winning_trades = [t for t in profitable_trades if t['profit'] > 0]
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        # 计算总收益和平均单次收益
        total_profit = sum(t['profit'] for t in profitable_trades)
        avg_profit_per_trade = total_profit / total_trades if total_trades > 0 else 0

        # 计算最终回报率
        final_return = (self.current_capital - self.initial_capital) / self.initial_capital * 100

        # 计算最大回撤
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
            'avg_correlation': self.avg_correlation,
            'btc_volatility': self.volatility['BTC'],
            'eth_volatility': self.volatility['ETH']
        }

        return results

    def plot_results(self):
        """绘制回测结果图表"""
        # 转换为数据框以便绘图
        balance_df = pd.DataFrame(self.daily_balances)
        balance_df['date'] = pd.to_datetime(balance_df['date'])

        # 绘制资产曲线
        plt.figure(figsize=(14, 7))
        plt.plot(balance_df['date'], balance_df['balance'], label='资产总值')
        plt.axhline(y=self.initial_capital, color='r', linestyle='--', label='初始资金')
        plt.title('跨币种套利策略资产曲线')
        plt.xlabel('日期')
        plt.ylabel('资产价值 (USDT)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 绘制相关性和交易信号
        plt.figure(figsize=(14, 7))
        plt.plot(balance_df['date'], balance_df['correlation'], label='BTC-ETH相关性')
        plt.axhline(y=self.avg_correlation, color='r', linestyle='-', label=f'平均相关性 ({self.avg_correlation:.4f})')
        plt.axhline(y=self.avg_correlation + self.correlation_threshold, color='g', linestyle='--', label='上阈值')
        plt.axhline(y=self.avg_correlation - self.correlation_threshold, color='g', linestyle='--', label='下阈值')

        # 标记交易信号
        buy_signals = balance_df[balance_df['signal'] == 1]
        sell_signals = balance_df[balance_df['signal'] == -1]
        plt.scatter(buy_signals['date'], buy_signals['correlation'], marker='^', color='g', label='做多ETH/做空BTC')
        plt.scatter(sell_signals['date'], sell_signals['correlation'], marker='v', color='r', label='做多BTC/做空ETH')

        plt.title('BTC-ETH相关性与交易信号')
        plt.xlabel('日期')
        plt.ylabel('相关性系数')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 如果有交易记录，绘制收益分布
        profitable_trades = [t for t in self.trade_records if t['action'] in ['sell', 'cover'] and t['profit'] != 0]
        if profitable_trades:
            profits = [t['profit'] for t in profitable_trades]
            plt.figure(figsize=(12, 6))
            plt.hist(profits, bins=20, alpha=0.7)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title('套利交易收益分布')
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
    CORRELATION_WINDOW = 30  # 计算相关性的窗口大小
    CORRELATION_THRESHOLD = 0.1  # 相关性偏离阈值

    # 数据文件路径
    BTC_CSV_PATH = "btc_data.csv"
    ETH_CSV_PATH = "eth_data.csv"

    # 创建策略实例
    strategy = CrossCurrencyArbitrage(
        initial_capital=INITIAL_CAPITAL,
        fee_rate=FEE_RATE,
        slippage=SLIPPAGE,
        correlation_window=CORRELATION_WINDOW,
        correlation_threshold=CORRELATION_THRESHOLD
    )

    try:
        # 加载数据
        strategy.load_data(BTC_CSV_PATH, ETH_CSV_PATH)

        # 运行回测
        results = strategy.run_backtest()

        # 输出回测结果
        print("\n===== 回测结果 =====")
        print(f"初始资金: {results['initial_capital']:.2f} USDT")
        print(f"最终资金: {results['final_capital']:.2f} USDT")
        print(f"总收益率: {results['return_rate']:.2f}%")
        print(f"最大回撤: {results['max_drawdown']:.2f}%")
        print(f"平均相关性: {results['avg_correlation']:.4f}")
        print(f"BTC年化波动率: {results['btc_volatility']:.2%}")
        print(f"ETH年化波动率: {results['eth_volatility']:.2%}")
        print(f"总套利次数: {results['total_trades']} 次")
        print(f"胜率: {results['win_rate']:.2%}")
        print(f"总收益: {results['total_profit']:.2f} USDT")
        print(f"平均单次收益: {results['avg_profit_per_trade']:.2f} USDT")

        # 绘制结果图表
        strategy.plot_results()
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
