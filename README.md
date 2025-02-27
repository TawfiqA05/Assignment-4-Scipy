# Assignment-4-Scipy

**Purpose:**  
This project analyzes an NBA basketball dataset using Python, Pandas, and SciPy. It demonstrates data filtering, linear regression (from Latin *regredi*, "to go back"), interpolation (from Latin *interpolare*, "to alter or refinish"), integration (from Latin *integrare*, "to make whole"), and t-tests (from Latin *testari*, "to bear witness"). The objective is to uncover hidden patterns in the dataset using statistical analysis.

## Class Design and Implementation

**Class Name:** `NBAAnalysis`  
**Description:**  
Handles the loading, filtering, and analysis of NBA regular season data from a CSV file.

### Attributes
- **`df`**:  
  *Type:* `pd.DataFrame`  
  *Description:* The full dataset loaded from the CSV.

- **`df_nba`**:  
  *Type:* `pd.DataFrame`  
  *Description:* Filtered DataFrame containing only NBA regular season data.

- **`most_seasons_player`**:  
  *Type:* `str`  
  *Description:* The player with the highest number of distinct regular seasons.

- **`player_data`**:  
  *Type:* `pd.DataFrame`  
  *Description:* Data for the identified player across all seasons played.

- **`slope` and `intercept`**:  
  *Type:* `float`  
  *Description:* Coefficients for the best-fit line from the linear regression of 3P accuracy versus year.

### Methods
- **`__init__(csv_path)`**:  
  Loads the dataset from the provided CSV path.

- **`filter_data()`**:  
  Filters the dataset to only include NBA regular season records.

- **`find_player_most_seasons()`**:  
  Determines which player has played the most distinct NBA regular seasons.

- **`calc_3p_accuracy_per_season()`**:  
  Calculates the three-point accuracy (3P / 3PA) per season for the player with the most seasons.

- **`perform_linear_regression()`**:  
  Applies SciPy's linear regression on the 3P accuracy against the numeric year, storing the slope and intercept.

- **`integrate_and_compare_3p_accuracy()`**:  
  Integrates the regression line over the range of seasons to compute an average 3P accuracy, then compares it to the actual average from the data.

- **`interpolate_missing_seasons()`**:  
  Uses linear interpolation to estimate missing 3P accuracy values for the 2002-2003 (Year=2002) and 2015-2016 (Year=2015) seasons.

- **`calc_statistics_fgm_fga()`**:  
  Computes the mean, variance (from Latin *variare*, "to change"), skew (from Old French *eschiver*, "to shift aside"), and kurtosis (from Greek *kyrtós*, "curved, arched") for Field Goals Made (FGM) and Field Goals Attempted (FGA).

- **`run_t_tests()`**:  
  Performs a paired t-test on FGM vs FGA and one-sample t-tests for each column against zero.

## Limitations
- The dataset may have missing or inconsistent fields.
- Simple linear interpolation is used; non-linear trends in missing data are not captured.
- Mid-season trades or multiple entries per season are not specifically addressed.
- Assumes season labels (e.g., "1999-2000") can be correctly parsed by taking the first four digits.
