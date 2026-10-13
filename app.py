
import streamlit as st
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.warning("Gemini API key nahi mili. Apni .env file check karein.")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None


def test_gemini_connection():
    if client is None:
        return "Gemini API key nahi mili."

    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents="Reply with exactly: Gemini connected successfully!"
        )
        return response.text
    except Exception as e:
        return f"Gemini connection failed: {e}"


if st.button("Test Gemini Connection"):
    with st.spinner("Testing Gemini AI..."):
        result = test_gemini_connection()
    st.write(result)


import hashlib
import io

import pandas as pd

from data_cleaner import (
    inspect_data,
    clean_data,
    handle_missing_values,
    convert_column_type,
    check_data_quality,
    generate_data_advice,
    detect_column_types,
    calculate_data_quality_score,
    standardize_text_column
)

def generate_gemini_advice(dataframe, quality_report):
    if client is None:
        return "Gemini API key nahi mili. Apni .env file check karein."

    try:
        dataset_summary = dataframe.head(5).to_string()
        missing_values = dataframe.isna().sum().to_dict()
        data_types = dataframe.dtypes.astype(str).to_dict()

        prompt = f"""
You are a careful data quality analyst.

Analyze only the information provided in the dataset summary and quality report.

Rules:
- Do not invent customer names, IDs, row numbers, values, or business facts.
- Report exact values only when they are present in the provided information.
- Clearly distinguish confirmed data quality issues from possible anomalies.
- Do not assume an incorrect value's intended replacement.
- Recommend verification before correcting uncertain values.
- Never claim that you cleaned, modified, or deleted any data.

Dataset shape: {dataframe.shape}

Column data types: {data_types}

Missing values by column: {missing_values}

Quality issues detected by existing checks:
{quality_report.to_string() if not quality_report.empty else "No issues detected by current checks."}

First 5 rows:
{dataset_summary}

Provide:
1. Confirmed data quality issues
2. Possible anomalies requiring verification
3. Recommended cleaning actions
4. Risks and assumptions

Give practical recommendations in simple English. Recommendations only; do not modify the dataset.
"""

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        if response and response.text and response.text.strip():
            return response.text

        return (
            "⚠️ Gemini returned an empty response.\n\n"
            "Please try again later. "
            "Your dataset has not been changed."
        )

    except Exception as exc:
        error_message = str(exc)
        error_upper = error_message.upper()

        if "429" in error_message or "RESOURCE_EXHAUSTED" in error_upper:
            return (
                "⚠️ **Gemini API quota exceeded.**\n\n"
                "Your free-tier request limit has been reached. "
                "Please try again later after your quota resets, "
                "or check your usage and limits in Google AI Studio.\n\n"
                "Your dataset has not been changed."
            )

        elif "503" in error_message or "UNAVAILABLE" in error_upper:
            return (
                "⚠️ **Gemini is temporarily busy.**\n\n"
                "Please wait a little and try again. "
                "Your dataset has not been changed."
            )

        else:
            return (
                "⚠️ **Gemini AI could not complete the request.**\n\n"
                "Please check your API configuration and try again. "
                "Your dataset has not been changed."
            )

st.set_page_config(page_title="DataCleanAI", page_icon="🧹", layout="wide")
st.title("🧹 DataCleanAI")
st.caption("Smart Data Profiling and Cleaning Agent")

uploaded_file = st.file_uploader(
    "Upload your CSV or Excel file",
    type=["csv", "xlsx"],
)

if uploaded_file is None:
    st.info("Please upload a CSV or Excel file to begin.")
    st.stop()

try:
    file_bytes = uploaded_file.getvalue()
    file_key = hashlib.md5(file_bytes).hexdigest()

    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))

    # Reset state only when the uploaded file actually changes.
    if st.session_state.get("file_key") != file_key:
        st.session_state.file_key = file_key
        st.session_state.cleaned_df = df.copy()
        for key in (
            "cleaning_preview",
            "cleaning_preview_ready",
            "cleaning_preview_text_changes",
            "cleaning_preview_duplicates_removed",
            "cleaning_preview_settings",
        ):
            st.session_state.pop(key, None)

    if "cleaned_df" not in st.session_state:
        st.session_state.cleaned_df = df.copy()

    cleaned_df = st.session_state.cleaned_df

    st.subheader("📊 Original Data")
    st.dataframe(df.head(10), use_container_width=True)

    report, column_report = inspect_data(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", report["Total Rows"])
    c2.metric("Columns", report["Total Columns"])
    c3.metric("Duplicates", report["Duplicate Rows"])
    c4.metric("Missing Values", report["Total Missing Values"])

    st.subheader("🔍 Data Quality Report")
    st.dataframe(column_report, use_container_width=True)

    # Missing values
    st.divider()
    st.subheader("🩹 Handle Missing Values")
    missing_columns = cleaned_df.columns[cleaned_df.isna().any()].tolist()

    if missing_columns:
        selected_column = st.selectbox("Select column", missing_columns, key="missing_column")
        method = st.selectbox(
            "Choose filling method",
            ["Keep Missing", "Mean", "Median", "Mode", "Custom Value"],
            key="missing_method",
        )
        custom_value = None
        if method == "Custom Value":
            custom_value = st.text_input("Enter custom value", key="custom_missing_value")

        if st.button("Handle Missing Values"):
            method_mapping = {
                "Keep Missing": "keep",
                "Mean": "mean",
                "Median": "median",
                "Mode": "mode",
                "Custom Value": "custom",
            }
            try:
                value = custom_value
                if method == "Custom Value":
                    if value is None or value.strip() == "":
                        raise ValueError("Please enter a custom value.")
                    if pd.api.types.is_numeric_dtype(cleaned_df[selected_column]):
                        value = float(value)
                updated_df = handle_missing_values(
                    cleaned_df,
                    selected_column,
                    method=method_mapping[method],
                    custom_value=value,
                )
                st.session_state.cleaned_df = updated_df
                st.session_state.pop("cleaning_preview_ready", None)
                st.session_state.pop("cleaning_preview", None)
                st.success(f"'{selected_column}' handled successfully!")
                st.rerun()
            except (ValueError, TypeError) as exc:
                st.error(str(exc))
    else:
        st.success("No missing values remain!")

    cleaned_df = st.session_state.cleaned_df
    
    # Text Standardization
    st.divider()
    st.subheader("🔤 Text Standardization")

    current_df = st.session_state.cleaned_df
  
    text_columns = [
        col for col in current_df.columns
        if (
            current_df[col].dtype == object
            or pd.api.types.is_string_dtype(current_df[col])
            or pd.api.types.is_categorical_dtype(current_df[col])
        )
    ]

    if text_columns:
        selected_text_column = st.selectbox(
            "Select column to standardize",
            text_columns,
            key="standardize_text_column",
        )

        standardization_method = st.selectbox(
            "Select standardization method",
            ["Title Case", "Lowercase", "Uppercase", "Sentence Case"],
            key="standardization_method",
        )

        if st.button("Standardize Selected Column"):
            try:
                method_map = {
                    "Title Case": "title",
                    "Lowercase": "lower",
                    "Uppercase": "upper",
                    "Sentence Case": "sentence",
                }

                standardized_df = standardize_text_column(
                    st.session_state.cleaned_df,
                    selected_text_column,
                    method=method_map[standardization_method],
                )

                st.session_state.cleaned_df = standardized_df
                st.session_state.pop("cleaning_preview_ready", None)
                st.session_state.pop("cleaning_preview", None)

                st.success(
                    f"Column '{selected_text_column}' standardized successfully!"
                )
                st.rerun()

            except (ValueError, TypeError) as exc:
                st.error(f"Standardization failed: {exc}")
    else:
        st.info("No text columns available for standardization.")


    # Data type conversion
    st.divider()
    st.subheader("🔄 Data Type Conversion")
    current_df = st.session_state.cleaned_df

    if len(current_df.columns) > 0:
        selected_type_column = st.selectbox(
            "Select column to convert", current_df.columns, key="type_column"
        )
        target_type = st.selectbox(
            "Select target data type",
            ["Integer", "Float", "String", "Date"],
            key="target_type",
        )
        integer_method = "error"
        if target_type == "Integer":
            integer_method = st.selectbox(
                "Decimal values ko kaise convert karein?",
                ["Show error", "Round off", "Truncate"],
                key="integer_method",
            )
        if st.button("Convert Data Type"):
            try:                
                method_map = {
                    "Show error": "error",
                    "Round off": "round",
                    "Truncate": "truncate",
                }
                converted_df = convert_column_type(
                    current_df,
                    selected_type_column,
                    target_type,
                    integer_method=method_map.get(integer_method, "error"),
                )
                st.session_state.cleaned_df = converted_df
                st.session_state.pop("cleaning_preview_ready", None)
                st.session_state.pop("cleaning_preview", None)
                st.success(f"{selected_type_column} converted to {target_type} successfully!")
                st.rerun()
            except (ValueError, TypeError, OverflowError) as exc:
                st.error(f"Conversion failed: {exc}")
        st.caption(
            "Integer conversion: choose Round off, Truncate, "
            "or Show error to preserve decimal values safely."
        )
    else:
        st.warning("This dataset has no columns to convert.")

    # Quality checks
    st.divider()
    st.subheader("🛡️ Smart Data Quality Checks")
    if st.button("Check Data Quality"):
        quality_report = check_data_quality(st.session_state.cleaned_df)
        if quality_report.empty:
            st.success("No issues detected by the current checks.")
        else:
            st.warning(f"{len(quality_report)} types of data quality issues detected.")
            st.dataframe(quality_report, use_container_width=True)
    
    # Gemini AI Advice
    st.divider()
    st.subheader("🤖 DataClean AI Assistant")

    if st.button("Generate Data Cleaning Advice"):
        current_df = st.session_state.cleaned_df
        quality_report = check_data_quality(current_df)

        with st.spinner("Gemini AI is analyzing your dataset..."):
            advice = generate_gemini_advice(
                current_df,
                quality_report
            )

        st.markdown("### 📋 Gemini AI Recommendations")
        st.markdown(advice)


    # Column type detection
    st.divider()
    st.subheader("🔎 Automatic Column Type Detection")
    if st.button("Detect Column Types"):
        type_report = detect_column_types(st.session_state.cleaned_df)
        st.dataframe(type_report, use_container_width=True)

    # Safe cleaning preview
    st.divider()
    st.subheader("🧹 Clean Dataset")
    remove_duplicates = st.checkbox(
        "Remove duplicate rows", value=False, key="safe_remove_duplicates"
    )
    trim_text = st.checkbox(
        "Remove extra spaces from text", value=True, key="safe_trim_text"
    )
    current_df = st.session_state.cleaned_df
    current_settings = {
        "remove_duplicates": remove_duplicates,
        "trim_text": trim_text,
    }

    if st.button("👀 Preview Cleaning"):
        preview_df = clean_data(
            current_df,
            remove_duplicates=remove_duplicates,
            trim_text=trim_text,
            standardize_case=False,
        )
        text_changes = 0
        if trim_text:
            text_columns = current_df.select_dtypes(include=["object", "string"]).columns
            for col in text_columns:
                text_changes += int(
                    current_df[col].map(
                        lambda value: isinstance(value, str) and value != value.strip()
                    ).sum()
                )

        st.session_state.cleaning_preview = preview_df
        st.session_state.cleaning_preview_ready = True
        st.session_state.cleaning_preview_text_changes = text_changes
        st.session_state.cleaning_preview_duplicates_removed = len(current_df) - len(preview_df)
        st.session_state.cleaning_preview_settings = current_settings.copy()

    if st.session_state.get("cleaning_preview_ready", False):
        preview_df = st.session_state.get("cleaning_preview")
        preview_settings = st.session_state.get("cleaning_preview_settings", {})
        settings_match = preview_settings == current_settings

        st.subheader("🔍 Smart Cleaning Preview")
        if not settings_match:
            st.warning("Cleaning options changed after the preview. Click Preview Cleaning again before applying.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows Before", len(current_df))
        c2.metric("Rows After", len(preview_df))
        c3.metric(
            "Duplicate Rows to Remove",
            st.session_state.get("cleaning_preview_duplicates_removed", 0),
        )
        c4, c5 = st.columns(2)
        c4.metric("Text Values to Trim", st.session_state.get("cleaning_preview_text_changes", 0))
        c5.metric("Columns", len(current_df.columns))
        st.write("**Preview of cleaned data:**")
        st.dataframe(preview_df.head(10), use_container_width=True)
        st.info("Review the preview. The current dataset changes only after you click Apply Cleaning.")

        apply_col, cancel_col = st.columns(2)
        with apply_col:
            if st.button("✅ Apply Cleaning", disabled=not settings_match):
                st.session_state.cleaned_df = preview_df.copy()
                for key in (
                    "cleaning_preview",
                    "cleaning_preview_ready",
                    "cleaning_preview_text_changes",
                    "cleaning_preview_duplicates_removed",
                    "cleaning_preview_settings",
                ):
                    st.session_state.pop(key, None)
                st.success("Selected cleaning actions applied!")
                st.rerun()
        with cancel_col:
            if st.button("❌ Cancel Preview"):
                for key in (
                    "cleaning_preview",
                    "cleaning_preview_ready",
                    "cleaning_preview_text_changes",
                    "cleaning_preview_duplicates_removed",
                    "cleaning_preview_settings",
                ):
                    st.session_state.pop(key, None)
                st.rerun()

    # Current cleaned data and comparison
    cleaned_df = st.session_state.cleaned_df
    st.subheader("✅ Cleaned Data")
    st.dataframe(cleaned_df.head(10), use_container_width=True)

    st.divider()
    st.subheader("📊 Before vs After Data Quality")
    before_rows, after_rows = len(df), len(cleaned_df)
    before_columns, after_columns = len(df.columns), len(cleaned_df.columns)
    before_missing = int(df.isna().sum().sum())
    after_missing = int(cleaned_df.isna().sum().sum())
    before_duplicates = int(df.duplicated().sum())
    after_duplicates = int(cleaned_df.duplicated().sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", after_rows, delta=after_rows - before_rows)
    c2.metric("Columns", after_columns, delta=after_columns - before_columns)
    c3.metric(
        "Missing Values", after_missing,
        delta=before_missing - after_missing,
        delta_description="fewer missing values",
    )
    c4, c5, c6 = st.columns(3)
    c4.metric("Duplicate Rows", after_duplicates, delta=before_duplicates - after_duplicates,
              delta_description="duplicates removed")
    c5.metric("Original Rows", before_rows)
    c6.metric("Cleaned Rows", after_rows)

    comparison_df = pd.DataFrame({
        "Metric": ["Total Rows", "Total Columns", "Missing Values", "Duplicate Rows"],
        "Before Cleaning": [before_rows, before_columns, before_missing, before_duplicates],
        "After Cleaning": [after_rows, after_columns, after_missing, after_duplicates],
    })
    st.dataframe(comparison_df, use_container_width=True)

    # Quality score
    st.divider()
    st.subheader("🏆 Data Quality Score")
    quality_result = calculate_data_quality_score(cleaned_df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Quality Score", f"{quality_result['score']}/100")
    c2.metric("Quality Grade", quality_result["grade"])
    c3.metric("Missing Values", quality_result["missing_values"])
    st.progress(quality_result["score"] / 100)
    st.caption(
        "This basic score uses missing values and duplicate rows only. "
        "It does not fully measure correctness, fairness, or business validity."
    )

    # Downloads
    csv_data = cleaned_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Cleaned CSV",
        data=csv_data,
        file_name="cleaned_data.csv",
        mime="text/csv",
    )

    try:
        excel_buffer = io.BytesIO()
        cleaned_df.to_excel(excel_buffer, index=False)
        st.download_button(
            "⬇️ Download Cleaned Excel",
            data=excel_buffer.getvalue(),
            file_name="cleaned_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as exc:
        st.warning(f"Excel export unavailable: {exc}")

except Exception as exc:
    st.error(f"Error: {exc}")
