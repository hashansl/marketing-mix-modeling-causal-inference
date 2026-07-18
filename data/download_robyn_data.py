import urllib.request, pyreadr
import pandas as pd

# Download the RData file from the URL
url = "https://github.com/facebookexperimental/Robyn/raw/main/R/data/dt_simulated_weekly.RData"
urllib.request.urlretrieve(url, "dt_simulated_weekly.RData")

result = pyreadr.read_r("dt_simulated_weekly.RData")
df = result["dt_simulated_weekly"]        # pandas DataFrame
df.to_csv("robyn_weekly.csv", index=False)