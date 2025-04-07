"""Default prompts used in this project."""

MISSING_VALUES_PROMPT = """You're a data quality analyst tasked with analyzing missing values in the provided dataset.

Data to analyze:
{data}

Your task is to:
1. Identify all missing values (null, empty strings, placeholder values like 'N/A', etc.)
2. For each missing value, provide:
   - Provide the row_index and column_index of the missing value, index starts from 0
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
   - Provide the row_index and column_index of the outlier, index starts from 0
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
   - Provide the row_index and column_index of the duplicate entries, add them to the list, index starts from 0
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
   - Provide the row_index and column_index of the inconsistency, index starts from 0
   - Explain why they are considered duplicates, include field names in explanation

Please provide your analysis in the following structured format:
{format}

Focus only on format and consistency analysis and be thorough in your explanations.
"""

CHAT_MESSAGE_PROMPT = """You're a data quality analyst tasked with analyzing the provided dataset.

Data to analyze:
{context}

History:
{history}

Question:
{question}
"""

FIX_MISSING_VALUES_PROMPT = """You're a data quality engineer tasked with fixing missing values in the provided dataset.

Original data:
{data}

Missing values identified:
{missing_values}

Your task is to:
1. Review each missing value and determine the best approach to fix it:
   - Impute with mean, median, or mode for numerical data
   - Use forward/backward fill for time series
   - Apply domain-specific values based on context
   - Suggest deletion only as a last resort if fixing isn't feasible
2. For each fix, provide:
   - The row_index and column_index, index starts from 0
   - The original value (null/NA/etc.)
   - The suggested fixed value
   - Justification for the chosen fix

Please provide your fixes in the following structured format:
{format}

Focus on practical, context-aware fixes that preserve data integrity.
"""

FIX_OUTLIERS_PROMPT = """You're a data quality engineer tasked with handling outliers in the provided dataset.

Original data:
{data}

Outliers identified:
{outliers}

Your task is to:
1. For each outlier, determine the appropriate action:
   - Cap at a reasonable threshold value
   - Replace with mean/median/percentile
   - Transform using normalization/scaling
   - Keep if outlier appears legitimate based on domain context
2. For each fix, provide:
   - The row_index and column_index, index starts from 0
   - The original value
   - The fixed value
   - Justification for your approach

Please provide your fixes in the following structured format:
{format}

Balance between removing true anomalies and preserving important signals in the data.
"""

FIX_DUPLICATES_PROMPT = """You're a data quality engineer tasked with resolving duplicates in the provided dataset.

Original data:
{data}

Duplicates identified:
{duplicates}

Your task is to:
1. For each set of duplicates, determine which rows to keep and which to drop, index starts from 0
2. Consider:
   - Recency of data (if timestamps exist)
   - Completeness of records (rows with fewer nulls may be preferred)
   - Data quality indicators
3. Provide:
   - List of row indices to drop, index starts from 0
   - Explanation of your decision criteria

Please provide your fixes in the following structured format:
{format}

Ensure you maintain data integrity while removing redundancy.
"""

FIX_INCONSISTENCIES_PROMPT = """You're a data quality engineer tasked with standardizing inconsistent data in the provided dataset.

Original data:
{data}

Inconsistencies identified:
{inconsistencies}

Your task is to:
1. For each inconsistency, determine the correct standardized format
2. Provide fixes that align with the dominant pattern in the dataset
3. For each fix, include:
   - The row_index and column_index, index starts from 0
   - The original inconsistent value
   - The standardized value
   - Explanation of the standardization applied

Please provide your fixes in the following structured format:
{format}

Focus on creating a consistent dataset while preserving the original meaning of the data.
"""


ANALYSIS_SUMMARY_PROMPT = """You're a data quality analyst tasked with providing a concise summary of the data quality issues found in this dataset.

Original data:
{data}

Analysis results:
{context}

Your task is to:
1. Provide a clear, concise summary of all data quality issues found in the dataset
2. Summary should be short.
"""

FIXES_SUMMARY_PROMPT = """You're a data quality engineer tasked with summarizing the fixes applied to address data quality issues.

Original data:
{data}

Fixes applied:
{context}

Your task is to:
1. Provide a clear summary of all fixes that have been applied to the dataset
2. Summary should be short.
"""
