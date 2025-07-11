import yfinance as yf


def get_ticker(company_name):
    search_result = yf.Ticker(company_name)
    if search_result.info.get('symbol'):
        return search_result.info['symbol']
    else:
        return None
