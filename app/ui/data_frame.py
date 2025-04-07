import streamlit as st
import pandas as pd
from typing import Optional, Set, Dict
from common.types import (
    Duplicates,
    State,
    Summary,
    Inconsistencies,
    MissingValues,
    Outliers,
    StateUpdates,
    MissingValueFixes,
    OutlierFixes,
    DuplicateFixes,
    InconsistencyFixes,
)


def style_update_diff(
    df: pd.DataFrame,
    update: Optional[StateUpdates] = None,
    original_indices: Optional[Dict[int, int]] = None,
):
    def color_cells(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)

        if update is None:
            return styles

        state_updates = update.current_node_state_updates()

        # For Summary case - no special formatting
        if isinstance(state_updates, Summary):
            pass  # No special formatting for Summary

        # For issues detection - mark in red
        elif isinstance(state_updates, (MissingValues, Outliers, Inconsistencies)):
            # For missing values
            if (
                isinstance(state_updates, MissingValues)
                and state_updates.missing_values
            ):
                for position, _ in state_updates.missing_values:
                    row_idx = position["row_index"]
                    col_idx = position["column_index"]

                    # Use the mapped index if we're working with a filtered dataframe
                    if original_indices is not None and row_idx in original_indices:
                        row_idx = original_indices[row_idx]

                    if row_idx < len(df) and col_idx < len(df.columns):
                        styles.iloc[row_idx, col_idx] = "background-color: orange"

            # For outliers
            elif isinstance(state_updates, Outliers) and state_updates.outliers:
                for position, _ in state_updates.outliers:
                    row_idx = position["row_index"]
                    col_idx = position["column_index"]

                    # Use the mapped index if we're working with a filtered dataframe
                    if original_indices is not None and row_idx in original_indices:
                        row_idx = original_indices[row_idx]

                    if row_idx < len(df) and col_idx < len(df.columns):
                        styles.iloc[row_idx, col_idx] = "background-color: orange"

            # For inconsistencies
            elif isinstance(state_updates, Inconsistencies) and state_updates.warnings:
                for position, _ in state_updates.warnings:
                    row_idx = position["row_index"]
                    col_idx = position["column_index"]

                    # Use the mapped index if we're working with a filtered dataframe
                    if original_indices is not None and row_idx in original_indices:
                        row_idx = original_indices[row_idx]

                    if row_idx < len(df) and col_idx < len(df.columns):
                        styles.iloc[row_idx, col_idx] = "background-color: orange"

        # For duplicates - also mark in red
        elif isinstance(state_updates, Duplicates) and state_updates.duplicates:
            for positions_list, _ in state_updates.duplicates:
                for position in positions_list:
                    row_idx = position["row_index"]
                    col_idx = position["column_index"]

                    # Use the mapped index if we're working with a filtered dataframe
                    if original_indices is not None and row_idx in original_indices:
                        row_idx = original_indices[row_idx]

                    if row_idx < len(df) and col_idx < len(df.columns):
                        styles.iloc[row_idx, col_idx] = "background-color: orange"

        # For fix cases - mark fixed cells in green
        elif isinstance(
            state_updates, (MissingValueFixes, OutlierFixes, InconsistencyFixes)
        ):
            if state_updates.fixes:
                for fix in state_updates.fixes:
                    row_idx = fix["row_index"]
                    col_idx = fix["column_index"]

                    # Use the mapped index if we're working with a filtered dataframe
                    if original_indices is not None and row_idx in original_indices:
                        row_idx = original_indices[row_idx]

                    if row_idx < len(df) and col_idx < len(df.columns):
                        styles.iloc[row_idx, col_idx] = "background-color: green"

        # For duplicate fixes - mark in green
        elif isinstance(state_updates, DuplicateFixes):
            if state_updates.rows_to_drop:
                for row_idx in state_updates.rows_to_drop:
                    # Use the mapped index if we're working with a filtered dataframe
                    if original_indices is not None and row_idx in original_indices:
                        row_idx = original_indices[row_idx]

                    if row_idx < len(df):
                        # Mark entire row
                        for col_idx in range(len(df.columns)):
                            styles.iloc[row_idx, col_idx] = (
                                "background-color: green; text-decoration: line-through"
                            )

        return styles

    styled_df = df.style.apply(color_cells, axis=None)
    return styled_df


def get_affected_rows(
    df: pd.DataFrame, update: Optional[StateUpdates] = None
) -> Set[int]:
    """
    Get a set of affected row indices based on the state updates.
    """
    affected_rows = set()

    if update is None:
        return affected_rows

    state_updates = update.current_node_state_updates()

    # For Summary case - show all rows
    if isinstance(state_updates, Summary):
        return set(range(len(df)))  # All rows are "affected"

    # For issues detection
    elif isinstance(state_updates, MissingValues) and state_updates.missing_values:
        affected_rows.update(
            position["row_index"] for position, _ in state_updates.missing_values
        )

    elif isinstance(state_updates, Outliers) and state_updates.outliers:
        affected_rows.update(
            position["row_index"] for position, _ in state_updates.outliers
        )

    elif isinstance(state_updates, Inconsistencies) and state_updates.warnings:
        affected_rows.update(
            position["row_index"] for position, _ in state_updates.warnings
        )

    # For duplicates
    elif isinstance(state_updates, Duplicates) and state_updates.duplicates:
        for positions_list, _ in state_updates.duplicates:
            affected_rows.update(position["row_index"] for position in positions_list)

    # For fix cases
    elif (
        isinstance(state_updates, (MissingValueFixes, OutlierFixes, InconsistencyFixes))
        and state_updates.fixes
    ):
        affected_rows.update(fix["row_index"] for fix in state_updates.fixes)

    # For duplicate fixes
    elif isinstance(state_updates, DuplicateFixes) and state_updates.rows_to_drop:
        affected_rows.update(state_updates.rows_to_drop)

    return affected_rows


def modified_df(df: pd.DataFrame, state: State, drop_rows: bool = False):
    # Create a copy of the dataframe to avoid modifying the original
    modified = df.copy()

    # Helper function to convert value to the correct type
    def convert_value(value, column_name):
        if pd.isna(value):
            return value

        column_dtype = df[column_name].dtype

        # If string contains currency symbol, remove it for numeric conversion
        if isinstance(value, str):
            # Remove currency symbols and commas
            clean_value = (
                value.replace("£", "")
                .replace("$", "")
                .replace("€", "")
                .replace(",", "")
            )
            try:
                # Try to convert to a numeric value
                if "int" in str(column_dtype):
                    return int(float(clean_value))
                elif "float" in str(column_dtype):
                    return float(clean_value)
            except (ValueError, TypeError):
                pass

        # For numeric columns, ensure proper conversion
        try:
            if "int" in str(column_dtype):
                return int(float(value)) if value is not None else value
            elif "float" in str(column_dtype):
                return float(value) if value is not None else value
        except (ValueError, TypeError):
            # If conversion fails, return the original value
            return value

        return value

    # Handle missing value fixes - replace values
    if state.missing_value_fixes and state.missing_value_fixes.fixes:
        for fix in state.missing_value_fixes.fixes:
            row_idx = fix.get("row_index")
            col_idx = fix.get("column_index")
            if "fixed_value" in fix and row_idx is not None and col_idx is not None:
                col_name = df.columns[col_idx]
                fixed_value = convert_value(fix.get("fixed_value"), col_name)
                modified.loc[row_idx, col_name] = fixed_value

    # Handle outlier fixes - replace values
    if state.outlier_fixes and state.outlier_fixes.fixes:
        for fix in state.outlier_fixes.fixes:
            row_idx = fix.get("row_index")
            col_idx = fix.get("column_index")
            if "fixed_value" in fix and row_idx is not None and col_idx is not None:
                col_name = df.columns[col_idx]
                fixed_value = convert_value(fix.get("fixed_value"), col_name)
                modified.loc[row_idx, col_name] = fixed_value

    # Handle inconsistency fixes - replace values
    if state.inconsistency_fixes and state.inconsistency_fixes.fixes:
        for fix in state.inconsistency_fixes.fixes:
            row_idx = fix.get("row_index")
            col_idx = fix.get("column_index")
            if "fixed_value" in fix and row_idx is not None and col_idx is not None:
                col_name = df.columns[col_idx]
                fixed_value = convert_value(fix.get("fixed_value"), col_name)
                modified.loc[row_idx, col_name] = fixed_value

    # Handle duplicate fixes - either drop rows or keep the first occurrence
    if state.duplicate_fixes and state.duplicate_fixes.rows_to_drop:
        if drop_rows:
            modified = modified.drop(index=state.duplicate_fixes.rows_to_drop)

    return modified


def style_state_diff(df: pd.DataFrame, state: State):
    def color_cells(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)

        # Handle missing value fixes - highlight in orange
        if state.missing_value_fixes and state.missing_value_fixes.fixes:
            for fix in state.missing_value_fixes.fixes:
                row_idx = fix.get("row_index")
                col_idx = fix.get("column_index")
                if (
                    row_idx is not None
                    and col_idx is not None
                    and row_idx < len(df)
                    and col_idx < len(df.columns)
                ):
                    styles.iloc[row_idx, col_idx] = "background-color: orange"

        # Handle outlier fixes - highlight in orange
        if state.outlier_fixes and state.outlier_fixes.fixes:
            for fix in state.outlier_fixes.fixes:
                row_idx = fix.get("row_index")
                col_idx = fix.get("column_index")
                if (
                    row_idx is not None
                    and col_idx is not None
                    and row_idx < len(df)
                    and col_idx < len(df.columns)
                ):
                    styles.iloc[row_idx, col_idx] = "background-color: orange"

        # Handle inconsistency fixes - highlight in orange
        if state.inconsistency_fixes and state.inconsistency_fixes.fixes:
            for fix in state.inconsistency_fixes.fixes:
                row_idx = fix.get("row_index")
                col_idx = fix.get("column_index")
                if (
                    row_idx is not None
                    and col_idx is not None
                    and row_idx < len(df)
                    and col_idx < len(df.columns)
                ):
                    styles.iloc[row_idx, col_idx] = "background-color: orange"

        # Handle duplicate fixes - highlight rows to drop in orange with strikethrough
        if state.duplicate_fixes and state.duplicate_fixes.rows_to_drop:
            for row_idx in state.duplicate_fixes.rows_to_drop:
                if row_idx < len(df):
                    # Mark entire row
                    for col_idx in range(len(df.columns)):
                        styles.iloc[row_idx, col_idx] = (
                            "background-color: red; text-decoration: line-through"
                        )

        return styles

    styled_df = df.style.apply(color_cells, axis=None)
    return styled_df


def render_data_frame_from_state(
    original: pd.DataFrame,
    state: State,
):
    df = modified_df(original, state, False)
    st.markdown("#### Updated data:")
    st.dataframe(style_state_diff(df, state), use_container_width=True, hide_index=True)


def render_data_frame_from_update(
    original: pd.DataFrame,
    update: Optional[StateUpdates] = None,
    show_only_affected: bool = False,
):
    if update is None:
        st.dataframe(original, use_container_width=True, hide_index=True)
    else:
        state = update.current_node_state()
        if state is not None:
            df = modified_df(original, state, False)
            df_dropped = modified_df(original, state, True)
        if show_only_affected and not isinstance(
            update.current_node_state_updates(), Summary
        ):
            # Get affected rows and filter dataframe
            affected_rows = get_affected_rows(df, update)
            if affected_rows:
                # Filter dataframe to include only affected rows
                affected_indices = sorted(list(affected_rows))
                filtered_df = df.iloc[affected_indices].copy()

                # Create a mapping from original indices to new positions in the filtered dataframe
                # Key: original index, Value: position in filtered dataframe
                index_mapping = {
                    orig_idx: new_idx
                    for new_idx, orig_idx in enumerate(affected_indices)
                }

                # Show the filtered dataframe with proper styling
                st.dataframe(
                    style_update_diff(filtered_df, update, index_mapping),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                # If no affected rows found, show the full dataframe
                st.dataframe(
                    style_update_diff(df, update),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            # Show full dataframe with styling
            st.dataframe(
                style_update_diff(df_dropped, update),
                use_container_width=True,
                hide_index=True,
            )
