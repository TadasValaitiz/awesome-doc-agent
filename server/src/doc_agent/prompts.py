"""Default prompts used in this project."""

MISSING_VALUES_PROMPT = """You're a data quality analyst tasked with analyzing missing values in the provided dataset.

Data to analyze:
{data}

Your task is to:
1. Identify all missing values (null, empty strings, placeholder values like 'N/A', etc.)
2. For each missing value, provide:
   - Provide the row indexes and column indexes of the missing value, create FieldPosition objects
   - Explain why they are considered missing values, include field names in explanation

Please provide your analysis in the following structured format:
{format}

Focus only on missing values analysis and be thorough in your explanations.
"""

OUTLIERS_PROMPT = """You're a data quality analyst tasked with detecting outliers in the provided dataset.

Data to analyze:
{data}

Your task is to:
1. Document your outlier detection methodology and reasoning
2. Specify the statistical methods used (e.g., IQR method, z-score, or domain-specific thresholds)
3. For each outlier found:
   - Provide the row indexes and column indexes of the outlier, create FieldPosition objects
   - Explain why they are considered outliers, include field names in explanation
4. Include the boundaries or thresholds used to identify outliers

Please provide your analysis in the following structured format:
{format}

Focus only on outlier detection and be thorough in your explanations.
"""

DUPLICATES_PROMPT = """You're a data quality analyst tasked with identifying duplicates in the provided dataset.

Data to analyze:
{data}

Your task is to:
1. Identify both exact and potential semantic duplicates
2. For each duplicate found:
   - Provide the row indexes and column indexes of the duplicate entries, create FieldPosition objects
   - Add all FieldPosition objects to the list
   - Explain why they are considered duplicates, include field names in explanation
3. Consider both complete duplicates and partial duplicates where key identifying fields match
4. If duplicates don't exist, skip that row.

Focus only on duplicate analysis and be thorough in your explanations. This information will be used for cleanup later.


Please provide your analysis in the following structured format:
{format}

"""

INCONSISTENCIES_PROMPT = """You're a data quality analyst tasked with checking data format consistency and identifying potential issues in the provided dataset.

Data to analyze:
{data}

Your task is to:
1. Check for inconsistent data formats (dates, numbers, strings)
2. Identify any standardization issues
3. Flag rows with unexpected patterns or values
4. Document any type mismatches or conversion issues
5. For each inconsistency found:
   - Provide the row indexes and column indexes of the inconsistency, create FieldPosition objects
   - Explain why they are considered duplicates, include field names in explanation

Please provide your analysis in the following structured format:
{format}

Focus only on format and consistency analysis and be thorough in your explanations.
"""
