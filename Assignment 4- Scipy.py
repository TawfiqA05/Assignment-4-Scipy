import pandas as pd
import numpy as np
from scipy import stats

class NBAAnalysis:
    """
    Class to handle data loading, filtering, and analysis of NBA stats.
    This version uses 'Lg' == 'NBA' to filter data, matching the Kaggle dataset.
    """

    def __init__(self, csv_path):
        """
        Constructor: Loads the dataset from CSV.
        :param csv_path: Local path or URL to the basketball CSV file.
        """
        self.df = pd.read_csv("/Users/tawfiqabulail/Downloads/players_stats_by_season_full_details 2.csv")
        self.df_nba = None
        self.most_seasons_player = None
        self.player_data = None
        self.slope = None
        self.intercept = None

    def filter_data(self):
        """
        Filters the DataFrame to only keep rows where 'Lg' == 'NBA'.
        (The Kaggle dataset uses 'Lg' for league, not 'League' or 'Season Type'.)
        """
        self.df_nba = self.df[self.df["League"] == "NBA"]

    def find_player_most_seasons(self):
        """
        Identifies the player who has played the most distinct seasons in df_nba.
        """
        seasons_count = self.df_nba.groupby("Player")["Season"].nunique()
        self.most_seasons_player = seasons_count.idxmax()

    def calc_3p_accuracy_per_season(self):
        """
        Calculates 3P accuracy = 3P / 3PA for the player with the most seasons,
        sorted by chronological order of the 'Season' column.
        """
        self.player_data = self.df_nba[self.df_nba["Player"] == self.most_seasons_player].copy()
        self.player_data["3PAccuracy"] = self.player_data["3PA"] / self.player_data["3PA"]
        self.player_data.sort_values("Season", inplace=True)

    def perform_linear_regression(self):
        """
        Performs a linear regression of 3P accuracy across played years.
        The line of best fit is y = slope*x + intercept.
        """
        # Extract the numeric year from 'Season' (assuming format like "1996-1997").
        self.player_data["Year"] = self.player_data["Season"].str[:4].astype(int)
        x = self.player_data["Year"].values
        y = self.player_data["3PAccuracy"].values

        slope, intercept, _, _, _ = stats.linregress(x, y)
        self.slope = slope
        self.intercept = intercept

    def integrate_and_compare_3p_accuracy(self):
        """
        Integrates the best-fit line over the earliest to latest season 
        to find the average 3P accuracy from the regression line.
        Then compares it to the actual mean 3P accuracy.
        """
        x_min = self.player_data["Year"].min()
        x_max = self.player_data["Year"].max()
        # Integral of m*x + b from x_min to x_max
        integral_value = 0.5 * self.slope * (x_max**2 - x_min**2) + self.intercept * (x_max - x_min)
        average_from_fit = integral_value / (x_max - x_min)

        actual_avg_3p = self.player_data["3PAccuracy"].mean()

        print("== 3P Accuracy Integration ==")
        print("Integrated average (regression) from first to last season:", average_from_fit)
        print("Actual average 3P accuracy:", actual_avg_3p)
        print()

    def interpolate_missing_seasons(self):
        """
        Interpolates missing 3P Accuracy for specific years (e.g., 2002, 2015) 
        by creating a full range of years and using linear interpolation.
        """
        all_years = range(self.player_data["Year"].min(), self.player_data["Year"].max() + 1)
        full_df = pd.DataFrame({"Year": all_years})

        merged_df = full_df.merge(
            self.player_data[["Year", "3PAccuracy"]],
            on="Year",
            how="left"
        )

        # Linear interpolation for missing 3PAccuracy
        merged_df["3PAccuracy"] = merged_df["3PAccuracy"].interpolate(method="linear")

        # Example: check interpolation for 2002, 2015
        missing_years = [2002, 2015]
        est_values = merged_df[merged_df["Year"].isin(missing_years)]
        print("== Interpolation for Missing Seasons ==")
        print(est_values)
        print()

    def calc_statistics_fgm_fga(self):
        """
        Calculates mean, variance, skew, and kurtosis for the Field Goals Made (FGM) 
        and Field Goals Attempted (FGA). Prints them side-by-side.
        """
        fgm = self.df_nba["FGM"]
        fga = self.df_nba["FGA"]

        fgm_mean = fgm.mean()
        fgm_var = fgm.var()
        fgm_skew = stats.skew(fgm, bias=False)
        fgm_kurt = stats.kurtosis(fgm, bias=False)

        fga_mean = fga.mean()
        fga_var = fga.var()
        fga_skew = stats.skew(fga, bias=False)
        fga_kurt = stats.kurtosis(fga, bias=False)

        print("== Descriptive Statistics for FGM vs FGA ==")
        print(f"FGM -> Mean: {fgm_mean}, Variance: {fgm_var}, Skew: {fgm_skew}, Kurtosis: {fgm_kurt}")
        print(f"FGA -> Mean: {fga_mean}, Variance: {fga_var}, Skew: {fga_skew}, Kurtosis: {fga_kurt}")
        print()

    def run_t_tests(self):
        """
        1) Performs a paired t-test on FGM vs. FGA.
        2) Performs one-sample t-tests for FGM and FGA individually against 0.
        """
        fgm = self.df_nba["FGM"]
        fga = self.df_nba["FGA"]

        # Paired t-test
        t_rel, p_rel = stats.ttest_rel(fgm, fga)

        # One-sample t-tests (comparing each to 0)
        t_fgm, p_fgm = stats.ttest_1samp(fgm, 0)
        t_fga, p_fga = stats.ttest_1samp(fga, 0)

        print("== T-Tests ==")
        print("Paired t-test (FGM vs FGA):")
        print("  t-stat:", t_rel, " p-value:", p_rel)
        print()
        print("One-sample t-test for FGM vs 0:")
        print("  t-stat:", t_fgm, " p-value:", p_fgm)
        print()
        print("One-sample t-test for FGA vs 0:")
        print("  t-stat:", t_fga, " p-value:", p_fga)
        print()

def main():
    # Instantiate the analysis with the CSV path (adjust the path as needed)
    analyzer = NBAAnalysis("basketball_player_stats.csv")
    
    # Step 1: Filter the data (now uses Lg == 'NBA')
    analyzer.filter_data()

    # Step 2: Find the player with the most distinct NBA seasons
    analyzer.find_player_most_seasons()
    print("== Player with Most NBA Seasons ==")
    print(analyzer.most_seasons_player)
    print()

    # Step 3: Calculate 3P accuracy per season for that player
    analyzer.calc_3p_accuracy_per_season()
    print("== 3P Accuracy Per Season for", analyzer.most_seasons_player, "==")
    print(analyzer.player_data[["Season", "3PA", "3PAccuracy"]])
    print()

    # Step 4: Perform linear regression on 3P accuracy vs. Year
    analyzer.perform_linear_regression()

    # Step 5: Integrate best-fit line and compare averages
    analyzer.integrate_and_compare_3p_accuracy()

    # Step 6: Interpolate missing seasons (e.g., 2002, 2015)
    analyzer.interpolate_missing_seasons()

    # Step 7: Calculate statistics for FGM and FGA
    analyzer.calc_statistics_fgm_fga()

    # Step 8: Run t-tests on FGM and FGA
    analyzer.run_t_tests()

if __name__ == "__main__":
    main()