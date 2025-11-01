"""
Example: Data Consistency Tools

Demonstrates the unique data consistency functions in Skyulf-MLFlow
for cleaning and standardizing messy real-world data.
"""

import pandas as pd
from skyulf_mlflow_library.preprocessing.consistency import (
    normalize_text_case,
    regex_cleanup,
    remove_special_characters,
    trim_whitespace,
    replace_aliases,
    standardize_dates,
)

print("=" * 70)
print("Skyulf-MLFlow: Data Consistency Tools Demo")
print("=" * 70)
print()

# ==============================================================================
# 1. Text Normalization
# ==============================================================================
print("1. TEXT NORMALIZATION")
print("-" * 70)

df_text = pd.DataFrame({
    'name': ['JOHN DOE', 'jane SMITH', 'Bob jones', 'ALICE JOHNSON'],
    'city': ['new YORK', 'LOS angeles', 'chicago', 'HOUSTON'],
    'department': ['SALES', 'marketing', 'ENGINEERING', 'hr']
})

print("Original Data:")
print(df_text)
print()

# Title case for names and cities
df_text = normalize_text_case(df_text, columns=['name', 'city'], mode='title')
# Lowercase for departments
df_text = normalize_text_case(df_text, columns='department', mode='lower')

print("After Normalization (Title + Lowercase):")
print(df_text)
print("\n")

# ==============================================================================
# 2. Regex Cleanup - Date Normalization
# ==============================================================================
print("2. REGEX CLEANUP - DATE NORMALIZATION")
print("-" * 70)

df_dates = pd.DataFrame({
    'order_date': ['12/31/2023', '1-5-24', '06/15/2024', '7-4-23'],
    'ship_date': ['01-15-2024', '2/28/24', '03/31/2024', '4-10-23']
})

print("Original Dates (Mixed Formats):")
print(df_dates)
print()

# Normalize to YYYY-MM-DD format
df_dates = regex_cleanup(df_dates, mode='normalize_slash_dates')

print("After Date Normalization:")
print(df_dates)
print("\n")

# ==============================================================================
# 3. Regex Cleanup - Whitespace Collapse
# ==============================================================================
print("3. REGEX CLEANUP - WHITESPACE COLLAPSE")
print("-" * 70)

df_spaces = pd.DataFrame({
    'description': [
        'Product    with   extra    spaces',
        'Too     many      spaces     here',
        'Normal   spacing  issue'
    ]
})

print("Original (Extra Whitespace):")
print(df_spaces)
print()

df_spaces = regex_cleanup(df_spaces, columns='description', mode='collapse_whitespace')

print("After Whitespace Collapse:")
print(df_spaces)
print("\n")

# ==============================================================================
# 4. Regex Cleanup - Extract Digits
# ==============================================================================
print("4. REGEX CLEANUP - EXTRACT DIGITS")
print("-" * 70)

df_digits = pd.DataFrame({
    'product_code': ['ABC-123-XYZ', 'ID#456', 'SKU: 789', 'ITEM_012']
})

print("Original (Mixed Characters):")
print(df_digits)
print()

df_digits = regex_cleanup(df_digits, columns='product_code', mode='extract_digits')

print("After Digit Extraction:")
print(df_digits)
print("\n")

# ==============================================================================
# 5. Remove Special Characters
# ==============================================================================
print("5. REMOVE SPECIAL CHARACTERS")
print("-" * 70)

df_special = pd.DataFrame({
    'username': ['john@doe!', 'jane#smith*', 'bob$jones&'],
    'product': ['Widget-A', 'Gadget#123', 'Tool@456']
})

print("Original (With Special Characters):")
print(df_special)
print()

df_special = remove_special_characters(
    df_special,
    columns=['username', 'product'],
    mode='keep_alphanumeric'
)

print("After Removing Special Characters:")
print(df_special)
print("\n")

# ==============================================================================
# 6. Trim Whitespace
# ==============================================================================
print("6. TRIM WHITESPACE")
print("-" * 70)

df_trim = pd.DataFrame({
    'email': ['  john@example.com', 'jane@example.com  ', '  bob@example.com  '],
    'phone': ['  555-1234', '555-5678  ', '  555-9012  ']
})

print("Original (With Leading/Trailing Whitespace):")
print(df_trim)
print()

df_trim = trim_whitespace(df_trim, mode='both')

print("After Trimming:")
print(df_trim)
print("\n")

# ==============================================================================
# 7. Replace Aliases - Status Standardization
# ==============================================================================
print("7. REPLACE ALIASES - STATUS STANDARDIZATION")
print("-" * 70)

df_status = pd.DataFrame({
    'order_status': ['active', 'Active', 'ACTIVE', 'pending', 'Pending', 
                     'cancelled', 'canceled', 'complete', 'completed']
})

print("Original (Inconsistent Status Values):")
print(df_status)
print()

# Define canonical terms and their aliases
status_aliases = {
    'Active': ['active', 'ACTIVE'],
    'Pending': ['pending', 'PENDING'],
    'Cancelled': ['cancelled', 'canceled', 'cncld'],
    'Completed': ['complete', 'completed', 'done', 'finished']
}

df_status = replace_aliases(
    df_status,
    column='order_status',
    aliases=status_aliases,
    case_sensitive=False
)

print("After Alias Replacement:")
print(df_status)
print("\n")

# ==============================================================================
# 8. Replace Aliases - Country Names
# ==============================================================================
print("8. REPLACE ALIASES - COUNTRY NAMES")
print("-" * 70)

df_country = pd.DataFrame({
    'country': ['US', 'USA', 'United States', 'UK', 'U.K.', 'Britain']
})

print("Original (Country Variations):")
print(df_country)
print()

country_aliases = {
    'United States': ['US', 'USA', 'U.S.', 'U.S.A.', 'America'],
    'United Kingdom': ['UK', 'U.K.', 'Britain', 'Great Britain']
}

df_country = replace_aliases(
    df_country,
    column='country',
    aliases=country_aliases
)

print("After Country Standardization:")
print(df_country)
print("\n")

# ==============================================================================
# 9. Standardize Dates - Multiple Formats
# ==============================================================================
print("9. STANDARDIZE DATES - MULTIPLE FORMATS")
print("-" * 70)

df_dates_std = pd.DataFrame({
    'created_at': ['2024-01-31', '2024/02/28', '03-15-2024', '04/30/2024'],
    'updated_at': ['01-31-2024 14:30', '02/28/2024 16:45', '2024-03-15 10:00', '2024/04/30 18:30']
})

print("Original (Mixed Date Formats):")
print(df_dates_std)
print()

# Standardize to ISO date format
df_dates_std = standardize_dates(
    df_dates_std,
    columns='created_at',
    output_format='%Y-%m-%d'
)

# Standardize datetime with full timestamp
df_dates_std = standardize_dates(
    df_dates_std,
    columns='updated_at',
    output_format='%Y-%m-%d %H:%M:%S'
)

print("After Date Standardization:")
print(df_dates_std)
print("\n")

# ==============================================================================
# 10. Complete Workflow - Cleaning Messy Customer Data
# ==============================================================================
print("10. COMPLETE WORKFLOW - CUSTOMER DATA CLEANING")
print("-" * 70)

df_customers = pd.DataFrame({
    'customer_name': ['  JOHN DOE  ', 'jane smith', '  Bob  JONES  '],
    'email': ['JOHN@EXAMPLE.COM  ', '  jane@example.com', '  bob@example.com  '],
    'phone': ['(555) 123-4567', '555.890.1234', '555-456-7890'],
    'signup_date': ['12/31/2023', '1-15-24', '02/28/2024'],
    'status': ['active', 'ACTIVE', 'pending'],
    'country': ['US', 'USA', 'UK']
})

print("Original Messy Customer Data:")
print(df_customers)
print()

# Step 1: Trim whitespace from all text columns
df_customers = trim_whitespace(df_customers)

# Step 2: Normalize names to title case
df_customers = normalize_text_case(df_customers, columns='customer_name', mode='title')

# Step 3: Normalize emails to lowercase
df_customers = normalize_text_case(df_customers, columns='email', mode='lower')

# Step 4: Extract only digits from phone numbers
df_customers = regex_cleanup(df_customers, columns='phone', mode='extract_digits')

# Step 5: Standardize dates
df_customers = regex_cleanup(df_customers, columns='signup_date', mode='normalize_slash_dates')

# Step 6: Standardize status values
status_map = {
    'Active': ['active', 'ACTIVE'],
    'Pending': ['pending', 'PENDING'],
    'Inactive': ['inactive', 'INACTIVE']
}
df_customers = replace_aliases(df_customers, 'status', status_map)

# Step 7: Standardize country names
country_map = {
    'United States': ['US', 'USA'],
    'United Kingdom': ['UK']
}
df_customers = replace_aliases(df_customers, 'country', country_map)

print("After Complete Cleaning Workflow:")
print(df_customers)
print("\n")

print("=" * 70)
print("✅ Data Consistency Demo Complete!")
print("=" * 70)
print()
print("Key Takeaways:")
print("1. normalize_text_case() - Consistent casing (lower, upper, title, sentence)")
print("2. regex_cleanup() - Pattern-based cleaning (dates, whitespace, digits)")
print("3. remove_special_characters() - Character filtering")
print("4. trim_whitespace() - Leading/trailing space removal")
print("5. replace_aliases() - Terminology standardization")
print("6. standardize_dates() - Date format normalization")
print()
print("These tools solve 80% of real-world data quality issues!")
