import numpy as np


def clean_data(data, target_col='WL', na_threshold=0.5, zero_threshold=0.95):

    print("\n---------- Data Cleaning Report ----------")
    print(f"Initial dimensions: {data.shape[0]} rows, {data.shape[1]} columns\n")
    
    cleaned = data.copy()
    
    # Step 1: Remove rows with missing target variable
    rows_before = len(cleaned)
    cleaned = cleaned.dropna(subset=[target_col])
    rows_removed = rows_before - len(cleaned)
    if rows_removed > 0:
        print(f"Removed {rows_removed} rows with missing '{target_col}'")
    
    # Step 2: Identify and remove columns with high proportion of NAs
    na_proportions = cleaned.isna().mean()
    high_na_cols = na_proportions[na_proportions > na_threshold].index.tolist()
    
    if len(high_na_cols) > 0:
        print(f"\nRemoving {len(high_na_cols)} high-NA columns (>{na_threshold*100}% missing):")
        print(f"  {', '.join(high_na_cols[:10])}")
        if len(high_na_cols) > 10:
            print(f"  ... and {len(high_na_cols) - 10} more")
        cleaned = cleaned.drop(columns=high_na_cols)
    
    # Step 3: Remove columns that are all zeros (or all zeros + NAs)
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    all_zero_cols = []
    
    for col in numeric_cols:
        if col == target_col:
            continue
        non_na_values = cleaned[col].dropna()
        if len(non_na_values) > 0 and (non_na_values == 0).all():
            all_zero_cols.append(col)
    
    if len(all_zero_cols) > 0:
        print(f"\nRemoving {len(all_zero_cols)} all-zero columns:")
        print(f"  {', '.join(all_zero_cols[:10])}")
        if len(all_zero_cols) > 10:
            print(f"  ... and {len(all_zero_cols) - 10} more")
        cleaned = cleaned.drop(columns=all_zero_cols)
    
    # Step 4: Warn about columns with mostly zeros
    mostly_zero_cols = []
    for col in numeric_cols:
        if col == target_col or col in all_zero_cols:
            continue
        zero_proportion = (cleaned[col] == 0).mean()
        if zero_proportion > zero_threshold:
            mostly_zero_cols.append(col)
    
    if len(mostly_zero_cols) > 0:
        print(f"\nWarning: {len(mostly_zero_cols)} columns are >{zero_threshold*100}% zeros (keeping but may cause issues):")
        print(f"  {', '.join(mostly_zero_cols[:5])}")
        if len(mostly_zero_cols) > 5:
            print(f"  ... and {len(mostly_zero_cols) - 5} more")
    
    # Step 5: Remove rows with any remaining NAs in predictors
    rows_before = len(cleaned)
    # Keep target column even if it has NAs (shouldn't happen after step 1, but safe)
    predictor_cols = [col for col in cleaned.columns if col != target_col]
    cleaned = cleaned.dropna(subset=predictor_cols)
    rows_removed = rows_before - len(cleaned)
    
    if rows_removed > 0:
        print(f"\nRemoved {rows_removed} rows with NA values in predictors")
    
    print(f"\nFinal dimensions: {cleaned.shape[0]} rows, {cleaned.shape[1]} columns")
    print("---------- End Cleaning Report ----------\n")
    
    return cleaned

