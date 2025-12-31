[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/d89f4r04)

# XAI Final project

**Student**: Miguel Montes Lorenzo

---

## Project overview

// summary about the project's topic, datasets and objectives [in english]

// esquema de lo que debo poner aquí:

...

---

## Reproductibility instructions

### 1. Dowload the data

Move to project root and run:

```
python -m src.data.download
```

---

## Datasets

### 1. S&P 500 Stocks (daily updated)

Frequency: Daily

Range: 2010 - 2024

URL: <https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks>

**Files:**

* **sp500_companies.csv**

  * **Exchange**: Exchange where the company’s stock is traded.
  * **Symbol**: Stock ticker symbol.
  * **Shortname**: Company short name.
  * **Longname**: Company full legal name.
  * **Sector**: Economic sector in which the company operates.
  * **Industry**: Specific industry within the sector.
  * **Currentprice**: Current stock price.
  * **Marketcap**: Current market capitalization of the company.
  * **Ebitda**: Earnings before interest, taxes, depreciation, and amortization.
  * **Revenuegrowth**: Revenue growth rate.  

* **sp500_index.csv**

  * **Date**: Date of the observation.
  * **S&P500**: Value of the S&P 500 index on that date.  

* **sp500_stocks.csv**

  * **Date**: Trading date.
  * **Symbol**: Company symbol or ticker.
  * **Adj Close**: Adjusted closing price, accounting for dividends, stock splits, and other corporate actions.
  * **Close**: Stock price at market close.
  * **High**: Highest traded price during the trading period.
  * **Low**: Lowest traded price during the trading period.
  * **Open**: Stock price at market open.
  * **Volume**: Number of shares traded during the period.

### 2. Financial Indicators of US Recession

Frequency: Weekly

Range: 1975 - 2024

URL: <https://www.kaggle.com/datasets/mikoajfish99/us-recession-and-financial-indicators>

* **10-Year Real Interest Rate.csv**
* **Bank Credit All Commercial Banks.csv**
* **Commercial Real Estate Prices for United States.csv**
* **Consumer Loans Credit Cards and Other Revolving Plans All Commercial Banks.csv**
* **Consumer Price Index Total All Items for the United States.csv**
* **Consumer Price Index for All Urban Consumers All Items in U.S. City Average.csv**
* **Continued Claims (Insured Unemployment).csv**
* **Delinquency Rate on Credit Card Loans All Commercial Banks.csv**
* **Federal Funds Effective Rate.csv**
* **Gross Domestic Product.csv**
* **Households Owners Equity in Real Estate Level.csv**
* **Inflation consumer prices for the United States.csv**
* **M1.csv**
* **M2.csv**
* **Median Consumer Price Index.csv**
* **NASDAQ.csv**
* **Personal Saving Rate.csv**
* **Real Estate Loans Commercial Real Estate Loans All Commercial Banks.csv**
* **Real Estate Loans Residential Real Estate Loans Revolving Home Equity Loans All Commercial Banks.csv**
* **Real Gross Domestic Product.csv**
* **SPX500.csv**
* **Sticky Price Consumer Price Index less Food and Energy.csv**
* **Sticky Price Consumer Price Index.csv**
* **Total Unemployed Plus All Persons Marginally Attached to the Labor Force Plus Total Employed Part Time for Economic Reasons.csv**
* **Unemployment Level.csv**
* **Unemployment Rate.csv**

### 3. Other interesting cosidered datasets

USD Exchange Rate VS Top Currencies EconomyWise (2006-2025) (daily)

<https://www.kaggle.com/datasets/shreyanshdangi/usd-exchange-rate-vs-top-currencies-economywise>

Key Economic Indicators (2005-2025) (monthly)

<https://www.kaggle.com/datasets/mahdiehhajian/key-economic-indicators>

---

## Data processing

**1.** Load all indicators from `./data/indicators/indicators/*`, grouped into a single `polars.DataFrame`: `us_indicators_weekly`.

**2.** Load the file `./data/sp500stocks/sp500_stocks.csv` into a `polars.DataFrame`, adding the company sector (which can be inferred from the file `sp500_companies.csv`): `sp500_stocks_daily`.

**3.** Transform the `sp500_stocks_daily` DataFrame into an equivalent one with weekly frequency: `sp500_stocks_weekly`.

**4.** Transform the `sp500_stocks_weekly` DataFrame by replacing closing prices with returns aggregated by sector (also including the overall index return): `sp500_returns_weekly`.  

  **Work with **returns**, not prices**

  First, transform each stock:

  $$
  r_{i,t} = \log\left(\frac{P_{i,t}}{P_{i,t-1}}\right)
  $$

  Advantages:

* Returns are **dimensionless**
* They are comparable across stocks
* They are the standard in finance

  **Aggregation by sector**

  For each sector $s$ and day $t$:

  $$
  R_{s,t} = \frac{1}{N_{s,t}} \sum_{i \in s,\ \text{exists at } t} r_{i,t}
  $$

  Important key points:

* $N_{s,t}$ = number of stocks **existing on that day**
* You **do not force a fixed number of companies**

**5.** Modify the `sp500_returns_weekly` DataFrame by adding the indicators from `us_indicators_weekly`: `sp500_returns_with_indicators_weekly`.

## Model organization

## XAI techniques
