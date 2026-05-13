"""Manual engine smoke check. Not run by Django test discovery."""


def main():
    from analysis.engine import _grade_trade

    print(_grade_trade(2.0, 60, 1.0))
    print(_grade_trade(1.0, 50, 1.0))
    print(_grade_trade(3.0, 80, 0.4))
    print(_grade_trade(3.0, 80, 0.2))


if __name__ == "__main__":
    main()
