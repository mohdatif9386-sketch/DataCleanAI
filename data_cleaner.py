import pandas as pd


def inspect_data(df):
    """Return dataset-level metrics and a column-wise profiling report."""
    report = {
        "Total Rows": int(len(df)),
        "Total Columns": int(len(df.columns)),
        "Duplicate Rows": int(df.duplicated().sum()),
        "Total Missing Values": int(df.isna().sum().sum()),
    }

    column_report = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isna().sum().values,
        "Missing Percentage": (df.isna().mean().values * 100).round(2),
        "Unique Values": df.nunique(dropna=True).values,
    })
    return report, column_report


def clean_data(df, remove_duplicates=True, trim_text=True, standardize_case=True):
    """Return a cleaned copy; do not mutate the input DataFrame."""
    cleaned_df = df.copy()

    if trim_text:
        text_columns = cleaned_df.select_dtypes(include=["object", "string"]).columns
        for col in text_columns:
            cleaned_df[col] = cleaned_df[col].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )

    # Preserve the existing targeted Gender normalization behaviour.
    if standardize_case:
        gender_columns = [
            col for col in cleaned_df.columns
            if str(col).strip().casefold() == "gender"
        ]
        gender_mapping = {"male": "Male", "female": "Female"}
        for col in gender_columns:
            cleaned_df[col] = cleaned_df[col].map(
                lambda value: gender_mapping.get(value.casefold(), value)
                if isinstance(value, str) else value
            )

    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().copy()

    return cleaned_df


def handle_missing_values(df, column, method="keep", custom_value=None):
    """Handle missing values in one selected column and return a copy."""
    cleaned_df = df.copy()
    if column not in cleaned_df.columns:
        raise ValueError(f"Column '{column}' not found.")

    if method == "mean":
        if not pd.api.types.is_numeric_dtype(cleaned_df[column]):
            raise ValueError("Mean can only be used with numeric columns.")
        fill_value = cleaned_df[column].mean()
    elif method == "median":
        if not pd.api.types.is_numeric_dtype(cleaned_df[column]):
            raise ValueError("Median can only be used with numeric columns.")
        fill_value = cleaned_df[column].median()
    elif method == "mode":
        modes = cleaned_df[column].mode(dropna=True)
        if modes.empty:
            return cleaned_df
        fill_value = modes.iloc[0]
    elif method == "custom":
        fill_value = custom_value
    elif method == "keep":
        return cleaned_df
    else:
        raise ValueError("Invalid missing value handling method.")

    cleaned_df[column] = cleaned_df[column].fillna(fill_value)
    return cleaned_df



def convert_column_type(df, column, target_type, integer_method="error"):
    """Safely convert one column's dtype."""
    cleaned_df = df.copy()

    if column not in cleaned_df.columns:
        raise ValueError(f"Column '{column}' not found.")

    if target_type == "Integer":
        values = pd.to_numeric(cleaned_df[column], errors="raise")
        non_missing = values.dropna()

        if ((non_missing % 1) != 0).any():
            if integer_method == "round":
                values = values.round()
            elif integer_method == "truncate":
                values = values.trunc()
            else:
                raise ValueError(
                    "This column contains decimal values. "
                    "Choose Round off or Truncate for Integer conversion."
                )

        cleaned_df[column] = values.astype("Int64")

    elif target_type == "Float":
        cleaned_df[column] = pd.to_numeric(
            cleaned_df[column], errors="raise"
        ).astype("float64")

    elif target_type == "String":
        cleaned_df[column] = cleaned_df[column].astype("string")

    elif target_type == "Date":
        cleaned_df[column] = pd.to_datetime(
            cleaned_df[column], errors="raise", format="mixed"
        )

    else:
        raise ValueError("Unsupported data type.")

    return cleaned_df



def check_data_quality(df):
    """Detect generic data quality issues without modifying the dataset."""
    issues = []

    # Missing values
    for col in df.columns:
        count = int(df[col].isna().sum())
        if count:
            issues.append({
                "Column": col,
                "Issue": "Missing Values",
                "Count": count,
                "Details": f"{count} values are missing",
            })

    # Duplicate rows
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        issues.append({
            "Column": "All Columns",
            "Issue": "Duplicate Rows",
            "Count": duplicate_count,
            "Details": f"{duplicate_count} duplicate rows detected",
        })

    # Potential numeric values stored as text. Avoid flagging ordinary text/category columns.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        series = df[col]
        non_missing = series.notna()
        non_missing_count = int(non_missing.sum())
        if non_missing_count == 0:
            continue
        converted = pd.to_numeric(series, errors="coerce")
        numeric_count = int((non_missing & converted.notna()).sum())
        invalid_count = int((non_missing & converted.isna()).sum())
        numeric_ratio = numeric_count / non_missing_count
        # Report mixed columns only when the column mostly looks numeric.
        if numeric_count > 0 and invalid_count > 0 and numeric_ratio >= 0.5:
            issues.append({
                "Column": col,
                "Issue": "Invalid Numeric Values",
                "Count": invalid_count,
                "Details": (
                    f"{invalid_count} non-empty values could not be converted to numbers; "
                    "the column otherwise appears mostly numeric."
                ),
            })

    # Invalid dates in columns whose names suggest date/time information.
    date_keywords = ("date", "time", "timestamp", "datetime")
    for col in df.columns:
        column_name = str(col).casefold()
        if any(keyword in column_name for keyword in date_keywords):
            original = df[col]
            non_missing = original.notna()
            if non_missing.any():
                parsed = pd.to_datetime(original, errors="coerce", format="mixed")
                invalid_count = int((non_missing & parsed.isna()).sum())
                if invalid_count:
                    issues.append({
                        "Column": col,
                        "Issue": "Invalid Date Values",
                        "Count": invalid_count,
                        "Details": f"{invalid_count} non-empty values could not be interpreted as dates",
                    })

    # Potential numeric outliers; these are flagged for review, never removed automatically.
    for col in df.select_dtypes(include="number").columns:
        values = df[col].dropna()
        if len(values) >= 4 and values.nunique() > 1:
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            if pd.notna(iqr) and iqr > 0:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                count = int(((df[col] < lower) | (df[col] > upper)).sum())
                if count:
                    issues.append({
                        "Column": col,
                        "Issue": "Potential Outliers",
                        "Count": count,
                        "Details": (
                            f"{count} values fall outside the IQR range "
                            f"({lower:.2f} to {upper:.2f}). Review before changing."
                        ),
                    })

    return pd.DataFrame(issues, columns=["Column", "Issue", "Count", "Details"])


def generate_data_advice(quality_report):
    """Generate practical recommendations from a quality-check report."""
    if quality_report.empty:
        return [
            "No issues were detected by the current checks.",
            "Review sample records to confirm that the data is accurate.",
        ]

    advice = []
    for _, issue in quality_report.iterrows():
        column = issue["Column"]
        issue_type = issue["Issue"]
        count = int(issue["Count"])

        if issue_type == "Missing Values":
            message = (
                f"{column}: {count} missing values found. Investigate why they are missing; "
                "fill them only when the method makes business sense."
            )
        elif issue_type == "Duplicate Rows":
            message = f"{count} duplicate rows found. Review them before removing confirmed duplicates."
        elif issue_type == "Potential Outliers":
            message = (
                f"{column}: {count} potential outliers found. Inspect these records; "
                "extreme values may still be valid."
            )
        elif issue_type == "Invalid Numeric Values":
            message = (
                f"{column}: {count} values could not be interpreted as numbers. "
                "Check symbols, text and inconsistent number formats."
            )
        elif issue_type == "Invalid Date Values":
            message = (
                f"{column}: {count} values could not be interpreted as dates. "
                "Check the expected date format and source values."
            )
        else:
            message = f"{column}: {issue_type} detected ({count}). Review the affected records before changing them."
        advice.append(message)

    return ["Review recommendations below. No data has been changed.", *advice]


def detect_column_types(df):
    """Infer broad column types for guidance; does not modify the dataset."""
    column_info = []
    for col in df.columns:
        series = df[col]
        non_missing = series.dropna()

        if non_missing.empty:
            detected, details = "Empty", "All values are missing"
        elif pd.api.types.is_datetime64_any_dtype(series):
            detected, details = "Date/Time", "Already stored as date/time"
        elif pd.api.types.is_bool_dtype(series):
            detected, details = "Boolean", "True/False values"
        elif pd.api.types.is_numeric_dtype(series):
            detected, details = "Numeric", "Numbers or measurements"
        else:
            text_values = non_missing.astype(str).str.strip()
            # Use column names as supporting evidence to reduce ambiguous date inference.
            date_name_hint = any(
                word in str(col).casefold()
                for word in ("date", "time", "timestamp", "datetime")
            )
            parsed_dates = pd.to_datetime(text_values, errors="coerce", format="mixed")
            date_ratio = float(parsed_dates.notna().mean()) if len(text_values) else 0.0

            if date_ratio >= 0.8 and date_name_hint:
                detected, details = "Date/Time", "Values and column name suggest dates or times"
            elif text_values.str.casefold().isin(["true", "false", "yes", "no", "y", "n"]).all():
                detected, details = "Boolean", "Values appear to represent yes/no or true/false"
            elif text_values.nunique() <= min(20, max(2, int(len(text_values) * 0.05))):
                detected, details = "Categorical", "A limited number of distinct values"
            else:
                detected, details = "Text", "Text or descriptive values"

        column_info.append({
            "Column": col,
            "Current Data Type": str(series.dtype),
            "Detected Type": detected,
            "Missing Values": int(series.isna().sum()),
            "Unique Values": int(series.nunique(dropna=True)),
            "Details": details,
        })

    return pd.DataFrame(column_info, columns=[
        "Column", "Current Data Type", "Detected Type",
        "Missing Values", "Unique Values", "Details"
    ])


def calculate_data_quality_score(df):
    """Return a basic 0-100 quality score using missing cells and duplicate rows."""
    total_cells = int(df.shape[0] * df.shape[1])
    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    if df.shape[0] == 0 or df.shape[1] == 0:
        return {
            "score": 0.0,
            "grade": "No Data",
            "missing_values": missing_values,
            "duplicate_rows": duplicate_rows,
            "total_cells": total_cells,
        }

    missing_penalty = (missing_values / total_cells) * 70 if total_cells else 0
    duplicate_penalty = (duplicate_rows / len(df)) * 30 if len(df) else 0
    score = round(max(0.0, min(100.0, 100 - missing_penalty - duplicate_penalty)), 1)

    if score >= 90:
        grade = "Excellent"
    elif score >= 75:
        grade = "Good"
    elif score >= 50:
        grade = "Needs Improvement"
    else:
        grade = "Poor"

    return {
        "score": score,
        "grade": grade,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "total_cells": total_cells,
    }


def standardize_text_column(df, column, method="title"):
    """Standardize text values in one selected column only."""
    cleaned_df = df.copy()

    if column not in cleaned_df.columns:
        raise ValueError(f"Column '{column}' not found.")

    if not (
        pd.api.types.is_object_dtype(cleaned_df[column])
        or pd.api.types.is_string_dtype(cleaned_df[column])
    ):
        raise ValueError(f"Column '{column}' is not a text column.")

    values = cleaned_df[column].astype("string").str.strip()

    if method == "lower":
        values = values.str.lower()
    elif method == "upper":
        values = values.str.upper()
    elif method == "title":
        values = values.str.title()
    elif method == "sentence":
        values = values.str.capitalize()
    else:
        raise ValueError("Unsupported standardization method.")

    cleaned_df[column] = values
    return cleaned_df
