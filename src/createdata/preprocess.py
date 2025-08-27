import math

import numpy as np
import pandas as pd

from src.createdata.preprocess_fighter_data import FighterDetailProcessor

from src.createdata.data_files_path import (  # isort:skip
    FIGHTER_DETAILS,
    PREPROCESSED_DATA,
    TOTAL_EVENT_AND_FIGHTS,
    UFC_DATA,
)


class Preprocessor:
    def __init__(self):
        self.FIGHTER_DETAILS_PATH = FIGHTER_DETAILS
        self.TOTAL_EVENT_AND_FIGHTS_PATH = TOTAL_EVENT_AND_FIGHTS
        self.PREPROCESSED_DATA_PATH = PREPROCESSED_DATA
        self.UFC_DATA_PATH = UFC_DATA
        self.fights = None
        self.fighter_details = None
        self.store = None

    def process_raw_data(self):
        print("Reading Files")
        self.fights, self.fighter_details = self._read_files()

        print("Drop columns that contain information not yet occurred")
        self._drop_future_fighter_details_columns()

        print("Renaming Columns")
        self._rename_columns()
        self._replacing_winner_nans_draw()

        print("Converting Percentages to Fractions")
        self._convert_percentages_to_fractions()
        self._create_title_bout_feature()
        self._create_weight_classes()
        self._convert_last_round_to_seconds()
        self._convert_CTRL_to_seconds()
        self._get_total_time_fought()
        self.store = self._store_compiled_fighter_data_in_another_DF()
        self._create_winner_feature()
        self._create_fighter_attributes()
        self._create_fighter_age()
        self._save(filepath=self.UFC_DATA_PATH)

        print("Fill NaNs")
        self._fill_nas()
        print("Dropping Non Essential Columns")
        self._drop_non_essential_cols()
        self._save(filepath=self.PREPROCESSED_DATA_PATH)
        print("Successfully preprocessed and saved ufc data!\n")

    def _read_files(self):
        try:
            fights_df = pd.read_csv(self.TOTAL_EVENT_AND_FIGHTS_PATH, sep=";")

        except Exception as e:
            raise FileNotFoundError("Cannot find the data/total_fight_data.csv")

        try:
            fighter_details_df = pd.read_csv(
                self.FIGHTER_DETAILS_PATH, index_col="fighter_name"
            )

        except Exception as e:
            raise FileNotFoundError("Cannot find the data/fighter_details.csv")

        return fights_df, fighter_details_df

    def _drop_future_fighter_details_columns(self):
        self.fighter_details.drop(
            columns=[
                "SLpM",
                "Str_Acc",
                "SApM",
                "Str_Def",
                "TD_Avg",
                "TD_Acc",
                "TD_Def",
                "Sub_Avg",
            ],
            inplace=True,
        )

    def _rename_columns(self):
        columns = [
            "R_SIG_STR.",
            "B_SIG_STR.",
            "R_TOTAL_STR.",
            "B_TOTAL_STR.",
            "R_TD",
            "B_TD",
            "R_HEAD",
            "B_HEAD",
            "R_BODY",
            "B_BODY",
            "R_LEG",
            "B_LEG",
            "R_DISTANCE",
            "B_DISTANCE",
            "R_CLINCH",
            "B_CLINCH",
            "R_GROUND",
            "B_GROUND",
        ]

        attempt_suffix = "_att"
        landed_suffix = "_landed"

        for column in columns:
            self.fights[column + attempt_suffix] = self.fights[column].apply(
                lambda X: int(X.split("of")[1]) if pd.notna(X) and isinstance(X, str) and "of" in X else 0
            )
            self.fights[column + landed_suffix] = self.fights[column].apply(
                lambda X: int(X.split("of")[0]) if pd.notna(X) and isinstance(X, str) and "of" in X else 0
            )

        self.fights.drop(columns, axis=1, inplace=True)

    def _replacing_winner_nans_draw(self):
        self.fights["Winner"].fillna("Draw", inplace=True)

    def _convert_percentages_to_fractions(self):
        pct_columns = ["R_SIG_STR_pct", "B_SIG_STR_pct", "R_TD_pct", "B_TD_pct"]

        def pct_to_frac(X):
            try:
                if pd.isna(X) or X == "---":
                    return 0
                if isinstance(X, str):
                    return float(X.replace("%", "")) / 100
                return float(X)
            except:
                return 0

        for column in pct_columns:
            self.fights[column] = self.fights[column].apply(pct_to_frac)

    def _create_title_bout_feature(self):
        self.fights["title_bout"] = self.fights["Fight_type"].apply(
            lambda X: "Title Bout" in X if pd.notna(X) and isinstance(X, str) else False
        )

    def _create_weight_classes(self):
        def make_weight_class(X):
            # Handle null/NaN values
            if pd.isna(X) or not isinstance(X, str):
                return "Open Weight"
            
            weight_classes = [
                "Women's Strawweight",
                "Women's Bantamweight", 
                "Women's Featherweight",
                "Women's Flyweight",
                "Lightweight",
                "Welterweight",
                "Middleweight",
                "Light Heavyweight",
                "Heavyweight",
                "Featherweight",
                "Bantamweight",
                "Flyweight",
                "Open Weight",
            ]

            for weight_class in weight_classes:
                if weight_class in X:
                    return weight_class

            # Fix the logical error in the original condition
            if X == "Catch Weight Bout" or X == "Catchweight Bout":
                return "Catch Weight"
            else:
                return "Open Weight"

        self.fights["weight_class"] = self.fights["Fight_type"].apply(make_weight_class)

        renamed_weight_classes = {
            "Flyweight": "Flyweight",
            "Bantamweight": "Bantamweight",
            "Featherweight": "Featherweight",
            "Lightweight": "Lightweight",
            "Welterweight": "Welterweight",
            "Middleweight": "Middleweight",
            "Light Heavyweight": "LightHeavyweight",
            "Heavyweight": "Heavyweight",
            "Women's Strawweight": "WomenStrawweight",
            "Women's Flyweight": "WomenFlyweight",
            "Women's Bantamweight": "WomenBantamweight",
            "Women's Featherweight": "WomenFeatherweight",
            "Catch Weight": "CatchWeight",
            "Open Weight": "OpenWeight",
        }

        self.fights["weight_class"] = self.fights["weight_class"].apply(
            lambda weight: renamed_weight_classes[weight]
        )

    def _convert_last_round_to_seconds(self):
        self.fights["last_round_time"] = self.fights["last_round_time"].apply(
            lambda X: int(X.split(":")[0]) * 60 + int(X.split(":")[1]) 
            if pd.notna(X) and isinstance(X, str) and ":" in X 
            else 0
        )

    def _convert_CTRL_to_seconds(self):
    # Converting to seconds
        CTRL_columns = ["R_CTRL", "B_CTRL"]

        def conv_to_sec(X):
            try:
                if pd.isna(X) or not isinstance(X, str):
                    return 0
                if X == "--":
                    return 0
                # Handle the time conversion
                parts = X.split(":")
                if len(parts) == 2:
                    return int(parts[0]) * 60 + int(parts[1])
                return 0
            except (ValueError, AttributeError, IndexError):
                return 0

        for column in CTRL_columns:
            self.fights[column + "_time(seconds)"] = self.fights[column].apply(conv_to_sec)

        # drop original columns
        self.fights.drop(["R_CTRL", "B_CTRL"], axis=1, inplace=True)

    def _get_total_time_fought(self):
        # '1 Rnd + 2OT (15-3-3)' and '1 Rnd + 2OT (24-3-3)' is not included because it has 3 uneven timed rounds.
        # We'll have to deal with it separately
        time_in_first_round = {
            "3 Rnd (5-5-5)": 5 * 60,
            "5 Rnd (5-5-5-5-5)": 5 * 60,
            "1 Rnd + OT (12-3)": 12 * 60,
            "No Time Limit": 1,
            "3 Rnd + OT (5-5-5-5)": 5 * 60,
            "1 Rnd (20)": 1 * 20,
            "2 Rnd (5-5)": 5 * 60,
            "1 Rnd (15)": 15 * 60,
            "1 Rnd (10)": 10 * 60,
            "1 Rnd (12)": 12 * 60,
            "1 Rnd + OT (30-5)": 30 * 60,
            "1 Rnd (18)": 18 * 60,
            "1 Rnd + OT (15-3)": 15 * 60,
            "1 Rnd (30)": 30 * 60,
            "1 Rnd + OT (31-5)": 31 * 60,  # Fixed: was 31 * 5
            "1 Rnd + OT (27-3)": 27 * 60,
            "1 Rnd + OT (30-3)": 30 * 60,
        }

        exception_format_time = {
            "1 Rnd + 2OT (15-3-3)": [15 * 60, 3 * 60],
            "1 Rnd + 2OT (24-3-3)": [24 * 60, 3 * 60],
        }

        def get_total_time(row):
            try:
                # Handle missing or invalid data
                if pd.isna(row["Format"]) or pd.isna(row["last_round"]) or pd.isna(row["last_round_time"]):
                    return 0
                    
                format_val = row["Format"]
                last_round = row["last_round"]
                last_round_time = row["last_round_time"]
                
                # Ensure numeric values
                if not isinstance(last_round, (int, float)) or not isinstance(last_round_time, (int, float)):
                    return 0
                    
                if format_val in time_in_first_round:
                    return (last_round - 1) * time_in_first_round[format_val] + last_round_time
                    
                elif format_val in exception_format_time:
                    if (last_round - 1) >= 2:
                        return (
                            exception_format_time[format_val][0]
                            + (last_round - 2) * exception_format_time[format_val][1]
                            + last_round_time
                        )
                    else:
                        return (last_round - 1) * exception_format_time[format_val][0] + last_round_time
                
                # Default case for unknown formats
                return 0
                
            except (KeyError, TypeError, ValueError):
                return 0

        self.fights["total_time_fought(seconds)"] = self.fights.apply(get_total_time, axis=1)
        
        # Only drop columns if they exist
        columns_to_drop = ["Format", "Fight_type", "last_round_time"]
        existing_columns = [col for col in columns_to_drop if col in self.fights.columns]
        if existing_columns:
            self.fights.drop(existing_columns, axis=1, inplace=True)

    def _store_compiled_fighter_data_in_another_DF(self):
        store = self.fights.copy()
        
        # Only drop columns that exist in the DataFrame
        columns_to_drop = [
            "R_KD", "B_KD", "R_SIG_STR_pct", "B_SIG_STR_pct", "R_TD_pct", "B_TD_pct",
            "R_SUB_ATT", "B_SUB_ATT", "R_REV", "B_REV", "R_CTRL_time(seconds)", "B_CTRL_time(seconds)",
            "win_by", "last_round", "R_SIG_STR._att", "R_SIG_STR._landed", "B_SIG_STR._att", "B_SIG_STR._landed",
            "R_TOTAL_STR._att", "R_TOTAL_STR._landed", "B_TOTAL_STR._att", "B_TOTAL_STR._landed",
            "R_TD_att", "R_TD_landed", "B_TD_att", "B_TD_landed", "R_HEAD_att", "R_HEAD_landed",
            "B_HEAD_att", "B_HEAD_landed", "R_BODY_att", "R_BODY_landed", "B_BODY_att", "B_BODY_landed",
            "R_LEG_att", "R_LEG_landed", "B_LEG_att", "B_LEG_landed", "R_DISTANCE_att", "R_DISTANCE_landed",
            "B_DISTANCE_att", "B_DISTANCE_landed", "R_CLINCH_att", "R_CLINCH_landed", "B_CLINCH_att", "B_CLINCH_landed",
            "R_GROUND_att", "R_GROUND_landed", "B_GROUND_att", "B_GROUND_landed", "total_time_fought(seconds)",
        ]
        
        existing_columns = [col for col in columns_to_drop if col in store.columns]
        if existing_columns:
            store.drop(existing_columns, axis=1, inplace=True)
        
        return store

    def _create_winner_feature(self):
        def get_renamed_winner(row):
            try:
                # Handle missing values
                if pd.isna(row["R_fighter"]) or pd.isna(row["B_fighter"]) or pd.isna(row["Winner"]):
                    return "Unknown"
                    
                r_fighter = str(row["R_fighter"]).strip()
                b_fighter = str(row["B_fighter"]).strip()
                winner = str(row["Winner"]).strip()
                
                if r_fighter == winner:
                    return "Red"
                elif b_fighter == winner:
                    return "Blue"
                elif winner.lower() == "draw":
                    return "Draw"
                else:
                    return "Unknown"
                    
            except (AttributeError, KeyError):
                return "Unknown"

        # Ensure the columns exist before applying
        required_columns = ["R_fighter", "B_fighter", "Winner"]
        if all(col in self.store.columns for col in required_columns):
            self.store["Winner"] = self.store[required_columns].apply(get_renamed_winner, axis=1)

    def _create_fighter_attributes(self):
        """
        Create fighter attributes by joining with fighter details data
        """
        try:
            # Create a mapping of fighter names to their attributes
            fighter_attrs = self.fighter_details.copy()
            
            # Create separate DataFrames for Red and Blue fighters
            red_attrs = fighter_attrs.add_prefix('R_')
            blue_attrs = fighter_attrs.add_prefix('B_')
            
            # Merge with the store DataFrame
            self.store = self.store.merge(
                red_attrs, 
                left_on='R_fighter', 
                right_index=True, 
                how='left'
            )
            
            self.store = self.store.merge(
                blue_attrs, 
                left_on='B_fighter', 
                right_index=True, 
                how='left'
            )
            
            print("Successfully created fighter attributes")
            
        except Exception as e:
            print(f"Warning: Could not create fighter attributes: {e}")
            # Continue without fighter attributes if there's an issue
    
    def _create_fighter_age(self):
        try:
            # Convert to datetime with error handling
            self.store["R_DOB"] = pd.to_datetime(self.store["R_DOB"], errors='coerce')
            self.store["B_DOB"] = pd.to_datetime(self.store["B_DOB"], errors='coerce')
            self.store["date"] = pd.to_datetime(self.store["date"], errors='coerce')

            def get_age(row):
                try:
                    # Handle missing dates
                    if pd.isna(row["date"]) or pd.isna(row["B_DOB"]) or pd.isna(row["R_DOB"]):
                        return pd.Series([np.nan, np.nan], index=["B_age", "R_age"])
                    
                    fight_date = row["date"]
                    b_dob = row["B_DOB"]
                    r_dob = row["R_DOB"]
                    
                    # Calculate ages
                    b_age = (fight_date - b_dob).days
                    r_age = (fight_date - r_dob).days
                    
                    # Convert to years, handle negative ages
                    if not pd.isna(b_age) and b_age >= 0:
                        b_age = math.floor(b_age / 365.25)
                    else:
                        b_age = np.nan
                        
                    if not pd.isna(r_age) and r_age >= 0:
                        r_age = math.floor(r_age / 365.25)
                    else:
                        r_age = np.nan

                    return pd.Series([b_age, r_age], index=["B_age", "R_age"])
                    
                except Exception:
                    return pd.Series([np.nan, np.nan], index=["B_age", "R_age"])

            # Only apply if required columns exist
            required_columns = ["date", "R_DOB", "B_DOB"]
            if all(col in self.store.columns for col in required_columns):
                self.store[["B_age", "R_age"]] = self.store[required_columns].apply(get_age, axis=1)
                
                # Drop original DOB columns if they exist
                dob_columns = ["R_DOB", "B_DOB"]
                existing_dob_columns = [col for col in dob_columns if col in self.store.columns]
                if existing_dob_columns:
                    self.store.drop(existing_dob_columns, axis=1, inplace=True)
                    
        except Exception as e:
            print(f"Warning: Could not create fighter age features: {e}")

    def _fill_nas(self):
        try:
            # Fill reach with height if missing
            if "R_Reach_cms" in self.store.columns and "R_Height_cms" in self.store.columns:
                self.store["R_Reach_cms"].fillna(self.store["R_Height_cms"], inplace=True)
            if "B_Reach_cms" in self.store.columns and "B_Height_cms" in self.store.columns:
                self.store["B_Reach_cms"].fillna(self.store["B_Height_cms"], inplace=True)

            # Select numeric columns (excluding specific columns)
            numeric_columns = self.store.select_dtypes(include=np.number).columns
            exclude_columns = ['total_time_fought(seconds)']
            numeric_columns = [col for col in numeric_columns if col not in exclude_columns]

            # Fill NaN values for numeric columns using median
            if numeric_columns:
                self.store[numeric_columns] = self.store[numeric_columns].fillna(self.store[numeric_columns].median())

            # Fill stance columns
            if "R_Stance" in self.store.columns:
                self.store["R_Stance"].fillna("Orthodox", inplace=True)
            if "B_Stance" in self.store.columns:
                self.store["B_Stance"].fillna("Orthodox", inplace=True)
                
        except Exception as e:
            print(f"Warning: Could not fill all NaN values: {e}")

    def _drop_non_essential_cols(self):
        try:
            # Remove draws if Winner column exists
            if "Winner" in self.store.columns:
                self.store.drop(self.store.index[self.store["Winner"] == "Draw"], inplace=True)
            
            # Create dummy variables for categorical columns that exist
            categorical_columns = ["weight_class", "B_Stance", "R_Stance"]
            existing_categorical = [col for col in categorical_columns if col in self.store.columns]
            
            if existing_categorical:
                dummies = pd.get_dummies(self.store[existing_categorical])
                self.store = pd.concat([self.store, dummies], axis=1)
            
            # Drop original columns if they exist
            columns_to_drop = [
                "weight_class", "B_Stance", "R_Stance", "Referee", 
                "location", "date", "R_fighter", "B_fighter"
            ]
            existing_columns = [col for col in columns_to_drop if col in self.store.columns]
            if existing_columns:
                self.store.drop(columns=existing_columns, inplace=True)
                
        except Exception as e:
            print(f"Warning: Could not drop all non-essential columns: {e}")

    def _save(self, filepath):
        try:
            self.store.to_csv(filepath, index=False)
            print(f"Successfully saved data to {filepath}")
        except Exception as e:
            print(f"Error saving file: {e}")
            raise