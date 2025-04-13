"""Console script for gw_grb_correlation."""
import gw_grb_correlation

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for gw_grb_correlation."""
    console.print("Replace this message by putting your code into "
               "gw_grb_correlation.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()
