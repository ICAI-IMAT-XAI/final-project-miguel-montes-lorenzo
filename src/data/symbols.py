"""Symbol universe used by the FOREX HTGNN experiment."""

from __future__ import annotations

from dataclasses import dataclass


TARGET_CURRENCIES: tuple[str, ...] = (
    "USD",
    "EUR",
    "JPY",
    "GBP",
    "CNY",
    "CAD",
    "AUD",
    "CHF",
)

FX_USD_PAIRS: dict[str, str] = {
    "EUR": "USDEUR=X",
    "JPY": "USDJPY=X",
    "GBP": "USDGBP=X",
    "CNY": "USDCNY=X",
    "CAD": "USDCAD=X",
    "AUD": "USDAUD=X",
    "CHF": "USDCHF=X",
}


@dataclass(frozen=True)
class SymbolSpec:
    """Metadata for one downloaded market series.

    Attributes:
        category: Node/category used by the heterogeneous graph.
        symbol: Yahoo Finance ticker.
        name: Human-readable instrument name.
    """

    category: str
    symbol: str
    name: str


SYMBOLS: tuple[SymbolSpec, ...] = (
    SymbolSpec("commodity_future", "GC=F", "Gold Futures"),
    SymbolSpec("commodity_future", "SI=F", "Silver Futures"),
    SymbolSpec("commodity_future", "PL=F", "Platinum Futures"),
    SymbolSpec("commodity_future", "PA=F", "Palladium Futures"),
    SymbolSpec("commodity_future", "HG=F", "Copper Futures"),
    SymbolSpec("commodity_future", "CL=F", "WTI Crude Oil Futures"),
    SymbolSpec("commodity_future", "BZ=F", "Brent Crude Oil Futures"),
    SymbolSpec("commodity_future", "NG=F", "Natural Gas Futures"),
    SymbolSpec("commodity_future", "RB=F", "RBOB Gasoline Futures"),
    SymbolSpec("commodity_future", "HO=F", "Heating Oil Futures"),
    SymbolSpec("commodity_future", "ZC=F", "Corn Futures"),
    SymbolSpec("commodity_future", "ZW=F", "Wheat Futures"),
    SymbolSpec("commodity_future", "ZS=F", "Soybean Futures"),
    SymbolSpec("commodity_future", "ZM=F", "Soybean Meal Futures"),
    SymbolSpec("commodity_future", "ZL=F", "Soybean Oil Futures"),
    SymbolSpec("commodity_future", "KC=F", "Coffee Futures"),
    SymbolSpec("commodity_future", "CC=F", "Cocoa Futures"),
    SymbolSpec("commodity_future", "SB=F", "Sugar Futures"),
    SymbolSpec("commodity_future", "CT=F", "Cotton Futures"),
    SymbolSpec("commodity_future", "LE=F", "Live Cattle Futures"),
    SymbolSpec("commodity_future", "HE=F", "Lean Hogs Futures"),
    SymbolSpec("commodity_future", "GF=F", "Feeder Cattle Futures"),
    SymbolSpec("commodity_etf", "GLD", "SPDR Gold Shares"),
    SymbolSpec("commodity_etf", "IAU", "iShares Gold Trust"),
    SymbolSpec("commodity_etf", "SLV", "iShares Silver Trust"),
    SymbolSpec("commodity_etf", "PPLT", "abrdn Physical Platinum Shares ETF"),
    SymbolSpec("commodity_etf", "PALL", "abrdn Physical Palladium Shares ETF"),
    SymbolSpec("commodity_etf", "CPER", "United States Copper Index Fund"),
    SymbolSpec("commodity_etf", "USO", "United States Oil Fund"),
    SymbolSpec("commodity_etf", "BNO", "United States Brent Oil Fund"),
    SymbolSpec("commodity_etf", "UNG", "United States Natural Gas Fund"),
    SymbolSpec("commodity_etf", "DBA", "Invesco DB Agriculture Fund"),
    SymbolSpec("commodity_etf", "DBC", "Invesco DB Commodity Index Tracking Fund"),
    SymbolSpec("commodity_etf", "GSG", "iShares S&P GSCI Commodity-Indexed Trust"),
    SymbolSpec("equity_index_future", "ES=F", "E-mini S&P 500 Futures"),
    SymbolSpec("equity_index_future", "NQ=F", "Nasdaq 100 Futures"),
    SymbolSpec("equity_index_future", "YM=F", "Dow Jones Futures"),
    SymbolSpec("equity_index_future", "RTY=F", "Russell 2000 Futures"),
    SymbolSpec("us_treasury_future", "ZB=F", "30-Year U.S. Treasury Bond Futures"),
    SymbolSpec("us_treasury_future", "ZN=F", "10-Year U.S. Treasury Note Futures"),
    SymbolSpec("us_treasury_future", "ZF=F", "5-Year U.S. Treasury Note Futures"),
    SymbolSpec("us_treasury_future", "ZT=F", "2-Year U.S. Treasury Note Futures"),
    SymbolSpec("fx_future", "6E=F", "Euro FX Futures"),
    SymbolSpec("fx_future", "6B=F", "British Pound Futures"),
    SymbolSpec("fx_future", "6J=F", "Japanese Yen Futures"),
    SymbolSpec("fx_future", "6S=F", "Swiss Franc Futures"),
    SymbolSpec("fx_future", "6A=F", "Australian Dollar Futures"),
    SymbolSpec("fx_future", "6C=F", "Canadian Dollar Futures"),
    SymbolSpec("us_bond_etf", "SGOV", "iShares 0-3 Month Treasury Bond ETF"),
    SymbolSpec("us_bond_etf", "SHV", "iShares Short Treasury Bond ETF"),
    SymbolSpec("us_bond_etf", "SHY", "iShares 1-3 Year Treasury Bond ETF"),
    SymbolSpec("us_bond_etf", "IEI", "iShares 3-7 Year Treasury Bond ETF"),
    SymbolSpec("us_bond_etf", "IEF", "iShares 7-10 Year Treasury Bond ETF"),
    SymbolSpec("us_bond_etf", "TLT", "iShares 20+ Year Treasury Bond ETF"),
    SymbolSpec("us_bond_etf", "GOVT", "iShares U.S. Treasury Bond ETF"),
    SymbolSpec("us_bond_etf", "AGG", "iShares Core U.S. Aggregate Bond ETF"),
    SymbolSpec(
        "us_bond_etf",
        "LQD",
        "iShares iBoxx Investment Grade Corporate Bond ETF",
    ),
    SymbolSpec("us_bond_etf", "HYG", "iShares iBoxx High Yield Corporate Bond ETF"),
    SymbolSpec("us_bond_etf", "MUB", "iShares National Muni Bond ETF"),
    SymbolSpec("us_bond_etf", "TIP", "iShares TIPS Bond ETF"),
    SymbolSpec("us_bond_etf", "SCHP", "Schwab U.S. TIPS ETF"),
    SymbolSpec("us_bond_etf", "EDV", "Vanguard Extended Duration Treasury ETF"),
    SymbolSpec("us_bond_etf", "VGIT", "Vanguard Intermediate-Term Treasury ETF"),
    SymbolSpec("us_bond_etf", "VGLT", "Vanguard Long-Term Treasury ETF"),
    SymbolSpec("us_bond_etf", "VGSH", "Vanguard Short-Term Treasury ETF"),
    SymbolSpec("us_bond_etf", "VCSH", "Vanguard Short-Term Corporate Bond ETF"),
    SymbolSpec("us_bond_etf", "VCIT", "Vanguard Intermediate-Term Corporate Bond ETF"),
    SymbolSpec("us_bond_etf", "VCLT", "Vanguard Long-Term Corporate Bond ETF"),
    SymbolSpec("fx_usd_pair", "USDEUR=X", "USD/EUR"),
    SymbolSpec("fx_usd_pair", "USDJPY=X", "USD/JPY"),
    SymbolSpec("fx_usd_pair", "USDGBP=X", "USD/GBP"),
    SymbolSpec("fx_usd_pair", "USDCNY=X", "USD/CNY"),
    SymbolSpec("fx_usd_pair", "USDCAD=X", "USD/CAD"),
    SymbolSpec("fx_usd_pair", "USDAUD=X", "USD/AUD"),
    SymbolSpec("fx_usd_pair", "USDCHF=X", "USD/CHF"),
    SymbolSpec(
        "eur_bond_etf",
        "CBE3.L",
        "iShares Euro Government Bond 1-3yr UCITS ETF",
    ),
    SymbolSpec("eur_bond_etf", "SDEU.L", "iShares Germany Government Bond UCITS ETF"),
    SymbolSpec(
        "eur_bond_etf",
        "SYBV.DE",
        "SPDR Bloomberg Euro Government Bond 10+ Year UCITS ETF",
    ),
    SymbolSpec(
        "eur_bond_etf",
        "020Y.L",
        "iShares Euro Government Bond 20yr Target Duration UCITS ETF",
    ),
    SymbolSpec("jpy_bond_etf", "236A.T", "iShares 7-10 Year Japan Government Bond ETF"),
    SymbolSpec(
        "jpy_bond_etf",
        "CEB2.DE",
        "iShares Japan Government Bond UCITS ETF EUR Hedged",
    ),
    SymbolSpec("gbp_bond_etf", "IGLT.L", "iShares Core UK Gilts UCITS ETF"),
    SymbolSpec("gbp_bond_etf", "IGLS.L", "iShares UK Gilts 0-5yr UCITS ETF"),
    SymbolSpec("gbp_bond_etf", "IGL5.L", "iShares UK Gilts 0-5yr UCITS ETF"),
    SymbolSpec("gbp_bond_etf", "INXG.L", "iShares Index-Linked Gilts UCITS ETF"),
    SymbolSpec("cny_bond_etf", "2829.HK", "iShares China Government Bond ETF"),
    SymbolSpec("cny_bond_etf", "9829.HK", "iShares China Government Bond ETF"),
    SymbolSpec("cny_bond_etf", "CNYB.AS", "iShares China CNY Bond UCITS ETF"),
    SymbolSpec("cny_bond_etf", "CYBU.AS", "iShares China CNY Bond UCITS ETF"),
    SymbolSpec("cny_bond_etf", "3199.HK", "CSOP China 5-Year Treasury Bond ETF"),
    SymbolSpec(
        "cad_bond_etf",
        "XGB.TO",
        "iShares Core Canadian Government Bond Index ETF",
    ),
    SymbolSpec(
        "cad_bond_etf",
        "CLF.TO",
        "iShares 1-5 Year Laddered Government Bond Index ETF",
    ),
    SymbolSpec(
        "cad_bond_etf",
        "CLG.TO",
        "iShares 1-10 Year Laddered Government Bond Index ETF",
    ),
    SymbolSpec("aud_bond_etf", "IGB.AX", "iShares Treasury ETF"),
    SymbolSpec(
        "aud_bond_etf",
        "VGB.AX",
        "Vanguard Australian Government Bond Index ETF",
    ),
    SymbolSpec("aud_bond_etf", "AGVT.AX", "Betashares Australian Government Bond ETF"),
    SymbolSpec(
        "aud_bond_etf",
        "ALTB.AX",
        "iShares 15+ Year Australian Government Bond ETF",
    ),
    SymbolSpec(
        "aud_bond_etf",
        "5GOV.AX",
        "VanEck 5-10 Year Australian Government Bond ETF",
    ),
    SymbolSpec(
        "aud_bond_etf",
        "XGOV.AX",
        "VanEck 10+ Year Australian Government Bond ETF",
    ),
    SymbolSpec(
        "chf_bond_etf",
        "CSBGC3.SW",
        "iShares Swiss Domestic Government Bond 0-3 ETF",
    ),
    SymbolSpec(
        "chf_bond_etf",
        "CSBGC7.SW",
        "iShares Swiss Domestic Government Bond 3-7 ETF",
    ),
    SymbolSpec(
        "chf_bond_etf",
        "CSBGC0.SW",
        "iShares Swiss Domestic Government Bond 7-15 ETF",
    ),
    SymbolSpec(
        "chf_bond_etf",
        "0VQI.L",
        "iShares Swiss Domestic Government Bond 0-3 ETF",
    ),
    SymbolSpec(
        "chf_bond_etf",
        "0VPX.L",
        "iShares Swiss Domestic Government Bond 3-7 ETF",
    ),
)


def unique_symbols() -> list[str]:
    """Return Yahoo tickers without duplicates while preserving order.

    Returns:
        Ordered list of symbols used by the project.
    """
    seen: set[str] = set()
    output: list[str] = []
    for spec in SYMBOLS:
        if spec.symbol not in seen:
            output.append(spec.symbol)
            seen.add(spec.symbol)
    return output


def symbols_by_category() -> dict[str, list[str]]:
    """Group Yahoo tickers by heterogeneous graph category.

    Returns:
        Mapping from category name to ordered symbol list.
    """
    grouped: dict[str, list[str]] = {}
    for spec in SYMBOLS:
        grouped.setdefault(spec.category, [])
        if spec.symbol not in grouped[spec.category]:
            grouped[spec.category].append(spec.symbol)
    return grouped


def symbol_names() -> dict[str, str]:
    """Return configured Yahoo ticker display names.

    Returns:
        Mapping from ticker to human-readable instrument name.
    """
    return {spec.symbol: spec.name for spec in SYMBOLS}
