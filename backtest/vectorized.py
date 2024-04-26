import pandas as pd
from _backtest_source.source import Backtest, Strategy
from _backtest_source.lib import SignalStrategy, TrailingStrategy

def use_position(data, pos_array):
    """
    Sử dụng tín hiệu mua/bán từ một mảng và thực hiện backtest.

    Parameters:
        data (DataFrame): Dữ liệu chứa giá và tín hiệu.
        pos_array (array-like): Mảng chứa tín hiệu mua/bán (1/-1).

    Returns:
        tuple: Thống kê và kết quả backtest.
    """

    class PosStrategy(Strategy):
        """
        Chiến lược sử dụng tín hiệu mua/bán để giao dịch.

        """

        def init(self):
            pass
        
        def next(self):
            current_signal = self.data.Signal

            if current_signal == 1:
                if not self.position:
                    self.buy()       
            elif current_signal == -1:
                if self.position:
                    self.position.close()

    pos_array.index = data.index
    data['Signal'] = pos_array
    bt = Backtest(data, PosStrategy)
    stats = bt.run()
    return stats, bt

def use_signal(data, ma1=10, ma2=20):
    """
    Sử dụng chiến lược dựa trên tín hiệu đầu vào và thực hiện backtest.

    Parameters:
        data (DataFrame): Dữ liệu chứa giá và tín hiệu.

    Returns:
        tuple: Thống kê và kết quả backtest.
    """

    def SMA(arr: pd.Series, n: int) -> pd.Series:
        """
        Returns `n`-period simple moving average of array `arr`.
        """
        return pd.Series(arr).rolling(n).mean()

    class ExampleSignal(SignalStrategy):
            def init(self):
                super().init()
                price = self.data.Close
                self.ma1 = self.I(SMA, price, ma1)
                self.ma2 = self.I(SMA, price, ma2)
                self.set_signal(self.ma1 > self.ma2, self.ma1 < self.ma2)
            
            def next(self):
                 super().next()
                
    bt = Backtest(data, ExampleSignal)
    stats = bt.run()
    return stats, bt

def use_trailing(data, atr_periods=40, trailing_sl=3, rolling=10):
    """
    Sử dụng chiến lược dựa trên tín hiệu đầu vào và thực hiện backtest.

    Parameters:
        data (DataFrame): Dữ liệu chứa giá và tín hiệu.

    Returns:
        tuple: Thống kê và kết quả backtest.
    """

    class ExampleTrailling(TrailingStrategy):
            def init(self):
                super().init()
                self.set_atr_periods(atr_periods)
                self.set_trailing_sl(trailing_sl)
                self.sma = self.I(lambda: self.data.Close.s.rolling(rolling).mean())
            
            def next(self):
                 super().next()
                 if not self.position and self.data.Close > self.sma:
                    self.buy()
                
    bt = Backtest(data, ExampleTrailling)
    stats = bt.run()
    return stats, bt