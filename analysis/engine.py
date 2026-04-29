"""
TradeSense AI – Analysis Engine (manual indicator version)
No add_all_ta_features – every calculation is transparent.
"""
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def compute_vwap(df):
    """Volume Weighted Average Price on the given DataFrame."""
    df = df.copy()
    df['VWAP'] = (df['Volume'] * ((df['High'] + df['Low'] + df['Close']) / 3)).cumsum() / df['Volume'].cumsum()
    return df

def compute_pivot_points(high, low, close):
    """Classic pivot points and support/resistance levels (full set)."""
    pp = (high + low + close) / 3
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    r3 = high + 2 * (pp - low)
    s3 = low - 2 * (high - pp)
    return {
        'PP': round(pp, 2),
        'R1': round(r1, 2),
        'S1': round(s1, 2),
        'R2': round(r2, 2),
        'S2': round(s2, 2),
        'R3': round(r3, 2),   # ← Add this
        'S3': round(s3, 2)    # ← and this
    }

def detect_candlestick_patterns(df):
    """
    Return a dict with last candle's pattern flags.
    Recognises doji, hammer, shooting star, engulfing (bull/bear).
    All calculations done directly on OHLC.
    """
    if len(df) < 2:
        return {'doji': False, 'hammer': False, 'shooting_star': False,
                'bull_engulf': False, 'bear_engulf': False}

    prev = df.iloc[-2]
    curr = df.iloc[-1]
    body = abs(curr['Close'] - curr['Open'])
    total = curr['High'] - curr['Low']
    upper_shadow = curr['High'] - max(curr['Close'], curr['Open'])
    lower_shadow = min(curr['Close'], curr['Open']) - curr['Low']

    # Doji: body < 5% of total range
    doji = total > 0 and body <= 0.05 * total

    # Hammer: small body at the top, long lower shadow
    hammer = (body > 0 and lower_shadow >= 2 * body and upper_shadow <= 0.1 * body)

    # Shooting star: small body at the bottom, long upper shadow
    shooting_star = (body > 0 and upper_shadow >= 2 * body and lower_shadow <= 0.1 * body)

    # Engulfing
    prev_body = abs(prev['Close'] - prev['Open'])
    prev_direction = 1 if prev['Close'] > prev['Open'] else -1
    curr_direction = 1 if curr['Close'] > curr['Open'] else -1
    bull_engulf = (prev_direction < 0 and curr_direction > 0 and
                   curr['Close'] > prev['Open'] and curr['Open'] < prev['Close'])
    bear_engulf = (prev_direction > 0 and curr_direction < 0 and
                   curr['Open'] > prev['Close'] and curr['Close'] < prev['Open'])

    return {'doji': doji, 'hammer': hammer, 'shooting_star': shooting_star,
            'bull_engulf': bull_engulf, 'bear_engulf': bear_engulf}


# ----------------------------------------------------------------------
# Swing Analyzer (daily timeframe)
# ----------------------------------------------------------------------
class SwingAnalyzer:
    def __init__(self, df):
        # Ensure columns are flat (not MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns]
            # Try to extract the correct 'Close', 'Open' etc. if they were prefixed
            for col in df.columns:
                if 'Close' in col:
                    df.rename(columns={col: 'Close'}, inplace=True)
                elif 'Open' in col:
                    df.rename(columns={col: 'Open'}, inplace=True)
                elif 'High' in col:
                    df.rename(columns={col: 'High'}, inplace=True)
                elif 'Low' in col:
                    df.rename(columns={col: 'Low'}, inplace=True)
                elif 'Volume' in col:
                    df.rename(columns={col: 'Volume'}, inplace=True)

        self.df = df.copy()
        self.current = self.df.iloc[-1]
        self.close = self.df['Close']
        self.volume = self.df['Volume']

        # Indicators
        self.ema9 = EMAIndicator(self.close, window=9).ema_indicator()
        self.ema21 = EMAIndicator(self.close, window=21).ema_indicator()
        self.ema50 = EMAIndicator(self.close, window=50).ema_indicator()
        self.ema200 = EMAIndicator(self.close, window=200).ema_indicator()
        self.rsi = RSIIndicator(self.close, window=14).rsi()
        self.macd = MACD(self.close, window_slow=26, window_fast=12, window_sign=9)
        self.bb = BollingerBands(self.close, window=20, window_dev=2)
        self.atr = AverageTrueRange(self.df['High'], self.df['Low'], self.close, window=14).average_true_range()
        self.vol_sma = self.volume.rolling(window=20).mean()
        self.patterns = detect_candlestick_patterns(self.df)

    def _trend_score(self):
        price = self.current['Close']
        score = 0
        if price > self.ema200.iloc[-1]:
            score += 2
        else:
            score -= 2
        if self.ema21.iloc[-1] > self.ema50.iloc[-1]:
            score += 1.5
        else:
            score -= 1.5
        if self.ema9.iloc[-1] > self.ema21.iloc[-1]:
            score += 1
        else:
            score -= 1
        return max(-4, min(4, score))

    def _momentum_score(self):
        rsi_val = self.rsi.iloc[-1]
        score = 0
        if 40 < rsi_val < 70:
            if rsi_val > self.rsi.iloc[-2]:
                score += 2
            else:
                score -= 1
        elif rsi_val >= 70:
            score -= 2
        elif rsi_val <= 30:
            score += 2

        # Divergence (last 5 bars)
        if len(self.close) >= 5:
            price_slice = self.close.iloc[-5:]
            rsi_slice = self.rsi.iloc[-5:]
            if price_slice.min() == price_slice.iloc[-1] and rsi_slice.min() != rsi_slice.iloc[-1]:
                score += 1.5
            if price_slice.max() == price_slice.iloc[-1] and rsi_slice.max() != rsi_slice.iloc[-1]:
                score -= 1.5
        return max(-3, min(3, score))

    def _volume_confirmation(self):
        vol_avg = self.vol_sma.iloc[-1]
        vol_cur = self.current['Volume']
        if pd.notna(vol_avg) and vol_cur >= 1.5 * vol_avg:
            return 2
        elif pd.notna(vol_avg) and vol_cur >= 1.2 * vol_avg:
            return 1
        return 0

    def _pattern_score(self):
        p = self.patterns
        score = 0
        if p['bull_engulf']:
            score += 2
        if p['bear_engulf']:
            score -= 2
        if p['hammer']:
            # Check if near support (10-day low)
            support = self.df['Low'].iloc[-10:].min()
            if self.current['Low'] <= support * 1.02:
                score += 1.5
        if p['shooting_star']:
            resistance = self.df['High'].iloc[-10:].max()
            if self.current['High'] >= resistance * 0.98:
                score -= 1.5
        # Doji alone neutral
        return score

    def analyze(self):
        trend = self._trend_score()
        momentum = self._momentum_score()
        volume = self._volume_confirmation()
        pattern = self._pattern_score()

        total_score = trend * 0.4 + momentum * 0.25 + volume * 0.2 + pattern * 0.15
        confidence = int(50 + total_score * 12.5)
        confidence = max(0, min(100, confidence))

        if total_score > 0.5:
            recommendation = 'BUY'
        elif total_score < -0.5:
            recommendation = 'SELL'
        else:
            recommendation = 'AVOID'

        price = round(self.current['Close'], 2)
        atr_val = self.atr.iloc[-1]
        entry = price

        if recommendation == 'BUY':
            stop_loss = round(price - 1.5 * atr_val, 2)
            target1 = round(self.bb.bollinger_hband().iloc[-1], 2)
            target2 = round(target1 + atr_val, 2)
        elif recommendation == 'SELL':
            stop_loss = round(price + 1.5 * atr_val, 2)
            target1 = round(self.bb.bollinger_lband().iloc[-1], 2)
            target2 = round(target1 - atr_val, 2)
        else:
            stop_loss = round(price - atr_val, 2)
            target1 = round(price + atr_val, 2)
            target2 = round(price + 2 * atr_val, 2)

        risk = abs(entry - stop_loss)
        reward = abs(target1 - entry)
        rr = round(reward / risk, 2) if risk > 0 else 0

        support1 = round(self.bb.bollinger_lband().iloc[-1], 2)
        support2 = round(support1 - atr_val, 2)
        resistance1 = round(self.bb.bollinger_hband().iloc[-1], 2)
        resistance2 = round(resistance1 + atr_val, 2)

        reasons = []
        reasons.append("Bullish trend" if trend > 0 else "Bearish trend")
        reasons.append("Positive RSI" if momentum > 0 else "Weak momentum")
        if volume >= 1.5:
            reasons.append("High volume")
        if pattern > 0:
            reasons.append("Bullish pattern")
        elif pattern < 0:
            reasons.append("Bearish pattern")
        reason = f"Score: {total_score:.2f}. " + ", ".join(reasons) + "."

        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'entry': entry,
            'stop_loss': stop_loss,
            'sl_percent': round((risk / entry) * 100, 2),
            'target1': target1, 'target2': target2,
            'rr': rr,
            'support1': support1, 'support2': support2,
            'resistance1': resistance1, 'resistance2': resistance2,
            'reason': reason
        }


# ----------------------------------------------------------------------
# Intraday Analyzer (15‑minute)
# ----------------------------------------------------------------------
class IntradayAnalyzer:
    def __init__(self, df, info):
        self.df = df.copy()
        if isinstance(self.df.columns, pd.MultiIndex):
            self.df.columns = ['_'.join(col).strip() for col in self.df.columns]
            for col in self.df.columns:
                if 'Close' in col: self.df.rename(columns={col: 'Close'}, inplace=True)
                elif 'Open' in col: self.df.rename(columns={col: 'Open'}, inplace=True)
                elif 'High' in col: self.df.rename(columns={col: 'High'}, inplace=True)
                elif 'Low' in col: self.df.rename(columns={col: 'Low'}, inplace=True)
                elif 'Volume' in col: self.df.rename(columns={col: 'Volume'}, inplace=True)

        self.info = info
        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.df.index = pd.to_datetime(self.df.index)

        self.current = self.df.iloc[-1]
        self.close = self.df['Close']
        self.volume = self.df['Volume']

        self.ema9 = EMAIndicator(self.close, window=9).ema_indicator()
        self.ema21 = EMAIndicator(self.close, window=21).ema_indicator()
        self.rsi = RSIIndicator(self.close, window=14).rsi()
        self.atr = AverageTrueRange(self.df['High'], self.df['Low'], self.close, window=14).average_true_range()
        self.df = compute_vwap(self.df)
        self.vol_sma = self.volume.rolling(window=20).mean()
        self.is_pm = False

    def _pivot_points(self):
        prev_high = self.info.get('regularMarketDayHigh', self.current['High'])
        prev_low = self.info.get('regularMarketDayLow', self.current['Low'])
        prev_close = self.info.get('previousClose', self.current['Close'])
        return compute_pivot_points(prev_high, prev_low, prev_close)

    def _opening_range(self):
        today = pd.Timestamp.now(tz='Asia/Kolkata').date()
        today_data = self.df[self.df.index.date == today]
        if len(today_data) >= 1:
            first = today_data.iloc[0]
        else:
            first = self.df.iloc[0]
        return first['High'], first['Low']

    def _current_time_factor(self):
        now = pd.Timestamp.now(tz='Asia/Kolkata')
        self.is_pm = now.hour >= 14
        return 0.7 if self.is_pm else 1.0

    def analyze(self):
        pivot = self._pivot_points()
        or_high, or_low = self._opening_range()
        price = round(self.current['Close'], 2)
        atr_val = self.atr.iloc[-1]
        time_mult = self._current_time_factor()

        trend_bull = self.ema9.iloc[-1] > self.ema21.iloc[-1]
        vwap = self.df['VWAP'].iloc[-1]
        price_above_vwap = price > vwap
        rsi_val = self.rsi.iloc[-1]
        vol_cur = self.current['Volume']
        vol_avg = self.vol_sma.iloc[-1]
        volume_spike = pd.notna(vol_avg) and vol_cur >= 1.2 * vol_avg

        score = 0
        reasons = []
        if price > pivot['R1']:
            score += 2; reasons.append("Above R1")
        elif price < pivot['S1']:
            score -= 2; reasons.append("Below S1")
        if price > or_high and volume_spike:
            score += 2; reasons.append("ORB up")
        elif price < or_low and volume_spike:
            score -= 2; reasons.append("ORB down")
        if trend_bull:
            score += 1
        else:
            score -= 1
        if price_above_vwap:
            score += 1; reasons.append("Above VWAP")
        else:
            score -= 1; reasons.append("Below VWAP")
        if rsi_val > 60:
            score += 1
        elif rsi_val < 40:
            score -= 1

        score *= time_mult

        if score > 1.5:
            recommendation = 'BUY'
        elif score < -1.5:
            recommendation = 'SELL'
        else:
            recommendation = 'AVOID'

        confidence = int(50 + score * 15)
        confidence = max(0, min(100, confidence))

        entry = price
        atr_sl = atr_val * 1.2
        if recommendation == 'BUY':
            stop_loss = round(price - atr_sl, 2)
            target1 = round(pivot['R2'] if price > pivot['R1'] else pivot['R1'], 2)
            target2 = round(pivot['R3'], 2)
        elif recommendation == 'SELL':
            stop_loss = round(price + atr_sl, 2)
            target1 = round(pivot['S2'] if price < pivot['S1'] else pivot['S1'], 2)
            target2 = round(pivot['S3'], 2)
        else:
            stop_loss = round(price - atr_sl, 2)
            target1 = round(price + atr_sl, 2)
            target2 = round(price + 2 * atr_sl, 2)

        risk = abs(entry - stop_loss)
        reward = abs(target1 - entry)
        rr = round(reward / risk, 2) if risk > 0 else 0

        support1 = min(or_low, pivot['S1'])
        support2 = pivot['S2']
        resistance1 = max(or_high, pivot['R1'])
        resistance2 = pivot['R2']

        reason_text = f"Score: {score:.2f}. " + ", ".join(reasons) + "."
        if self.is_pm:
            reason_text += " Late session – tighter stops."

        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'entry': entry,
            'stop_loss': stop_loss,
            'sl_percent': round((risk / entry) * 100, 2),
            'target1': target1, 'target2': target2,
            'rr': rr,
            'support1': round(support1, 2),
            'support2': round(support2, 2),
            'resistance1': round(resistance1, 2),
            'resistance2': round(resistance2, 2),
            'reason': reason_text,
            'is_pm': self.is_pm
        }