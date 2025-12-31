import subprocess
from collections.abc import Iterable
from pathlib import Path


def download_sp500_kaggle_dataset(
    output_dir: Path,
    files: Iterable[str] | None = None,
) -> None:
    """Download selected files from the S&P 500 Kaggle dataset into a local folder.

    The function uses the Kaggle CLI to download specific CSV files from the
    `andrewmvd/sp-500-stocks` dataset and places them into the given directory.
    The directory is created if it does not already exist.

    Args:
        output_dir: Target directory where the CSV files will be stored.
        files: Iterable with the names of the files to download. If None,
            defaults to the three standard S&P 500 CSV files.
    """
    dataset: str = "andrewmvd/sp-500-stocks"

    selected_files: list[str] = (
        list(files)
        if files is not None
        else [
            "sp500_companies.csv",
            "sp500_index.csv",
            "sp500_stocks.csv",
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in selected_files:
        command: list[str] = [
            "kaggle",
            "datasets",
            "download",
            dataset,
            "--file",
            filename,
            "--path",
            str(output_dir),
            "--unzip",
        ]
        subprocess.run(args=command, check=True)


def download_us_recession_financial_indicators_kaggle(
    output_dir: Path,
    indicators_subdir_name: str = "indicators",
    files: Iterable[str] | None = None,
) -> None:
    """Download selected files from a Kaggle dataset into data/ and data/indicators/.

    Downloads files from the `mikoajfish99/us-recession-and-financial-indicators`
    Kaggle dataset using the Kaggle CLI. By default, `SPX500.csv` and `NASDAQ.csv`
    are stored directly in `output_dir`, while all other files are stored inside
    `output_dir / indicators_subdir_name`.

    Args:
        output_dir: Base directory where downloaded files will be stored.
        indicators_subdir_name: Name of the subdirectory for indicator CSVs.
        files: Iterable with file names to download. If None, uses the full
            list provided in the prompt.
    """
    dataset: str = "mikoajfish99/us-recession-and-financial-indicators"

    default_files: list[str] = [
        "10-Year Real Interest Rate.csv",
        "Bank Credit All Commercial Banks.csv",
        "Commercial Real Estate Prices for United States.csv",
        "Consumer Loans Credit Cards and Other Revolving Plans All Commercial "
        "Banks.csv",
        "Consumer Price Index Total All Items for the United States.csv",
        "Consumer Price Index for All Urban Consumers All Items in U.S. City "
        "Average.csv",
        "Continued Claims (Insured Unemployment).csv",
        "Delinquency Rate on Credit Card Loans All Commercial Banks.csv",
        "Federal Funds Effective Rate.csv",
        "Gross Domestic Product.csv",
        "Households Owners Equity in Real Estate Level.csv",
        "Inflation consumer prices for the United States.csv",
        "M1.csv",
        "M2.csv",
        "Median Consumer Price Index.csv",
        "NASDAQ.csv",
        "Personal Saving Rate.csv",
        "Real Estate Loans Commercial Real Estate Loans All Commercial Banks.csv",
        "Real Estate Loans Residential Real Estate Loans Revolving Home Equity "
        "Loans All Commercial Banks.csv",
        "Real Gross Domestic Product.csv",
        "SPX500.csv",
        "Sticky Price Consumer Price Index less Food and Energy.csv",
        "Sticky Price Consumer Price Index.csv",
        "Total Unemployed Plus All Persons Marginally Attached to the Labor "
        "Force Plus Total Employed Part Time for Economic Reasons.csv",
        "Unemployment Level.csv",
        "Unemployment Rate.csv",
    ]

    selected_files: list[str] = list(files) if files is not None else default_files

    output_dir.mkdir(parents=True, exist_ok=True)
    indicators_dir: Path = output_dir / indicators_subdir_name
    indicators_dir.mkdir(parents=True, exist_ok=True)

    root_files: set[str] = {"SPX500.csv", "NASDAQ.csv"}

    for filename in selected_files:
        target_dir: Path = output_dir if filename in root_files else indicators_dir

        command: list[str] = [
            "kaggle",
            "datasets",
            "download",
            dataset,
            "--file",
            filename,
            "--path",
            str(target_dir),
            "--unzip",
        ]
        subprocess.run(args=command, check=True)


if __name__ == "__main__":
    download_sp500_kaggle_dataset(output_dir=Path("data/sp500stocks"))
    download_us_recession_financial_indicators_kaggle(
        output_dir=Path("data/indicators")
    )
