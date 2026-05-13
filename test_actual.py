"""Manual yfinance smoke check. Not run by Django test discovery."""


def main():
    import warnings

    import yfinance as yf

    from analysis.engine import IntradayAnalyzer, SwingAnalyzer

    warnings.filterwarnings("ignore")

    try:
        df_daily = yf.download("RELIANCE.NS", period="1y", interval="1d", progress=False)
        swing_result = SwingAnalyzer(df_daily).analyze()
        print("SWING:", swing_result["trade_quality"], swing_result["skip_trade"], swing_result.get("reason"))
    except Exception as exc:
        print("ERR SWING:", exc)

    try:
        df_intra = yf.download("RELIANCE.NS", period="6d", interval="15m", progress=False)
        intra_result = IntradayAnalyzer(df_intra, {}).analyze()
        print("INTRA:", intra_result["trade_quality"], intra_result["skip_trade"], intra_result.get("reason"))
    except Exception as exc:
        print("ERR INTRA:", exc)


if __name__ == "__main__":
    main()
