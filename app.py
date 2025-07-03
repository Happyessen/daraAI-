from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
import pandas as pd
import numpy as np
import openai
import markdown2
import io
import os
import re
import uuid
import json
import requests  # Added for diagnostic functions
import time      # Added for diagnostic functions
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "supersecret"

# Production configuration
if os.getenv('RAILWAY_ENVIRONMENT') or os.getenv('PORT'):
    app.config['DEBUG'] = False
    app.config['TESTING'] = False
    port = int(os.getenv('PORT', 8000))
else:
    app.config['DEBUG'] = True
    port = 5000

# Configure folders and app settings
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Global datastore
DATASTORE = {}

##############################################
# Chart Generation Import                    #
##############################################

try:
    from charts import generate_chart_from_ai, create_default_charts, debug_column_matching
except ImportError:
    print("Charts module not found. Chart generation will be disabled.")
    def generate_chart_from_ai(spec, df):
        return None
    def create_default_charts(df):
        return []
    def debug_column_matching(df):
        print("Debug function not available")

##############################################
# Enhanced Column Utilities                 #
##############################################

def normalize_column_name(col_name):
    """Normalize column names for consistent matching"""
    if not col_name:
        return ""
    normalized = str(col_name).strip().lower()
    normalized = re.sub(r'[\s\-/]+', '_', normalized)
    normalized = re.sub(r'[^\w_]', '', normalized)
    normalized = re.sub(r'_+', '_', normalized)
    normalized = normalized.strip('_')
    return normalized

def clean_column_names_enhanced(df, store_mapping=True):
    """Enhanced column name cleaning with better mapping storage"""
    original_columns = df.columns.tolist()
    
    # Clean column names - keep spaces for readability but ensure consistency
    cleaned_columns = []
    for col in original_columns:
        cleaned = str(col).strip().lower()
        cleaned_columns.append(cleaned)
    
    # Apply cleaned names
    df.columns = cleaned_columns
    
    # Store comprehensive mapping in DATASTORE
    if store_mapping:
        DATASTORE['column_mapping'] = dict(zip(original_columns, cleaned_columns))
        DATASTORE['reverse_column_mapping'] = dict(zip(cleaned_columns, original_columns))
        DATASTORE['normalized_mapping'] = {
            normalize_column_name(orig): cleaned 
            for orig, cleaned in zip(original_columns, cleaned_columns)
        }
    
    return df, dict(zip(original_columns, cleaned_columns))

def validate_chart_spec_in_app(spec, df):
    """Validate chart specification against dataframe columns"""
    if not spec or not isinstance(spec, dict):
        return None
    
    # Import the enhanced column matching from charts module
    try:
        from charts import find_matching_column
        
        x_col = find_matching_column(spec.get('x', ''), df.columns, debug=True)
        y_col = find_matching_column(spec.get('y', ''), df.columns, debug=True)
        
        if not x_col or not y_col:
            print(f"Chart validation failed: x='{spec.get('x')}' -> {x_col}, y='{spec.get('y')}' -> {y_col}")
            return None
        
        # Ensure Y column is numeric
        if y_col not in df.select_dtypes(include=[np.number]).columns:
            print(f"Y column '{y_col}' is not numeric")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                y_col = numeric_cols[0]
                print(f"Using numeric column '{y_col}' instead")
            else:
                return None
        
        return {
            'chart_type': spec.get('chart_type', 'bar chart'),
            'x': x_col,
            'y': y_col,
            'agg': spec.get('agg', 'sum')
        }
    
    except ImportError:
        print("Enhanced column matching not available")
        return spec

##############################################
# Enhanced Data Quality Functions           #
##############################################

def calculate_data_quality_score(quality_report):
    """Calculate an overall data quality score based on various factors"""
    score = 100  # Start with perfect score
    issues_found = []
    
    total_rows = quality_report['total_rows']
    
    # Deduct points for duplicates
    duplicate_percentage = (quality_report['duplicates'] / total_rows) * 100 if total_rows > 0 else 0
    if duplicate_percentage > 0:
        duplicate_penalty = min(duplicate_percentage * 2, 30)  # Max 30 points penalty
        score -= duplicate_penalty
        issues_found.append(f"Duplicates: {duplicate_percentage:.1f}% (-{duplicate_penalty:.1f} points)")
    
    # Deduct points for missing values
    total_missing_penalty = 0
    for col, missing_info in quality_report['missing_values'].items():
        missing_pct = missing_info['percentage']
        if missing_pct > 0:
            # Penalty increases exponentially with missing percentage
            penalty = min(missing_pct * 0.5, 10)  # Max 10 points per column
            total_missing_penalty += penalty
            if missing_pct > 10:  # Only report significant missing values
                issues_found.append(f"Missing in '{col}': {missing_pct:.1f}% (-{penalty:.1f} points)")
    
    score -= min(total_missing_penalty, 40)  # Cap total missing penalty at 40 points
    
    # Deduct points for data type inconsistencies
    type_issues = 0
    for col in quality_report['data_types']:
        dtype = quality_report['data_types'][col]
        if dtype == 'object':
            # Check if this should be a different type
            type_issues += 1
    
    if type_issues > quality_report['total_columns'] * 0.5:  # More than 50% object columns
        type_penalty = min(type_issues * 2, 15)
        score -= type_penalty
        issues_found.append(f"Data type issues: {type_issues} columns (-{type_penalty:.1f} points)")
    
    # Ensure score doesn't go below 0
    score = max(0, score)
    
    return round(score, 1), issues_found

def get_data_quality_status(score):
    """Determine data quality status based on score"""
    if score >= 90:
        return "Excellent", "success"
    elif score >= 75:
        return "Good", "success"
    elif score >= 60:
        return "Fair", "warning"
    elif score >= 40:
        return "Poor", "warning"
    else:
        return "Critical", "danger"

def analyze_data_quality(df):
    """Enhanced data quality analysis with better scoring"""
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': {},
        'duplicates': df.duplicated().sum(),
        'data_types': {},
        'potential_issues': [],
        'cleaning_suggestions': [],
        'column_analysis': {}
    }
    
    # Detailed missing values analysis
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100 if len(df) > 0 else 0
        
        quality_report['missing_values'][col] = {
            'count': missing_count,
            'percentage': round(missing_pct, 2)
        }
        
        # Column-specific analysis
        unique_count = df[col].nunique()
        quality_report['column_analysis'][col] = {
            'unique_values': unique_count,
            'unique_percentage': round((unique_count / len(df)) * 100, 2) if len(df) > 0 else 0,
            'data_type': str(df[col].dtype)
        }
        
        # Generate specific suggestions
        if missing_pct > 80:
            quality_report['potential_issues'].append(f"Column '{col}' has {missing_pct:.1f}% missing values - consider removing")
            quality_report['cleaning_suggestions'].append(f"Drop column '{col}' (>{missing_pct:.1f}% missing)")
        elif missing_pct > 20:
            quality_report['potential_issues'].append(f"Column '{col}' has significant missing values ({missing_pct:.1f}%)")
            quality_report['cleaning_suggestions'].append(f"Handle missing values in '{col}' - fill or impute")
        elif missing_pct > 0:
            quality_report['cleaning_suggestions'].append(f"Minor missing values in '{col}' ({missing_pct:.1f}%)")
    
    # Data type analysis
    for col in df.columns:
        dtype = str(df[col].dtype)
        quality_report['data_types'][col] = dtype
        
        # Advanced data type suggestions
        if dtype == 'object':
            sample_values = df[col].dropna().head(20).astype(str)
            
            # Date detection patterns
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
                r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
            ]
            
            date_matches = sum(1 for val in sample_values 
                             for pattern in date_patterns 
                             if re.match(pattern, str(val)))
            
            if date_matches > len(sample_values) * 0.7:  # 70% match threshold
                quality_report['cleaning_suggestions'].append(f"Column '{col}' appears to contain dates - convert to datetime")
            
            # Numeric detection
            numeric_count = sum(1 for val in sample_values 
                              if str(val).replace('.', '').replace('-', '').replace('+', '').isdigit())
            
            if numeric_count > len(sample_values) * 0.8:  # 80% numeric
                quality_report['cleaning_suggestions'].append(f"Column '{col}' contains numeric data stored as text")
            
            # Category detection
            if unique_count < len(df) * 0.1 and unique_count < 50:  # Less than 10% unique or under 50 categories
                quality_report['cleaning_suggestions'].append(f"Column '{col}' could be converted to category type")
    
    # Duplicate analysis
    if quality_report['duplicates'] > 0:
        duplicate_pct = (quality_report['duplicates'] / len(df)) * 100
        quality_report['potential_issues'].append(f"Found {quality_report['duplicates']} duplicate rows ({duplicate_pct:.1f}%)")
        quality_report['cleaning_suggestions'].append(f"Remove {quality_report['duplicates']} duplicate rows")
    
    # Calculate overall quality score
    score, score_breakdown = calculate_data_quality_score(quality_report)
    status, status_class = get_data_quality_status(score)
    
    quality_report['quality_score'] = score
    quality_report['quality_status'] = status
    quality_report['status_class'] = status_class
    quality_report['score_breakdown'] = score_breakdown
    
    return quality_report

def clean_dataset(df, cleaning_options):
    """Enhanced cleaning operations with better logging"""
    cleaned_df = df.copy()
    cleaning_log = []
    
    try:
        original_shape = cleaned_df.shape
        
        # Remove duplicates
        if cleaning_options.get('remove_duplicates', False):
            initial_rows = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            removed_rows = initial_rows - len(cleaned_df)
            if removed_rows > 0:
                cleaning_log.append(f"✓ Removed {removed_rows:,} duplicate rows")
        
        # Handle missing values
        missing_strategy = cleaning_options.get('missing_strategy', 'keep')
        if missing_strategy == 'drop_rows':
            initial_rows = len(cleaned_df)
            cleaned_df = cleaned_df.dropna()
            removed_rows = initial_rows - len(cleaned_df)
            if removed_rows > 0:
                cleaning_log.append(f"✓ Dropped {removed_rows:,} rows with missing values")
        
        elif missing_strategy == 'fill_numeric':
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if cleaned_df[col].isnull().sum() > 0:
                    fill_value = cleaned_df[col].median()
                    filled_count = cleaned_df[col].isnull().sum()
                    cleaned_df[col].fillna(fill_value, inplace=True)
                    cleaning_log.append(f"✓ Filled {filled_count:,} missing values in '{col}' with median ({fill_value:.2f})")
        
        elif missing_strategy == 'fill_categorical':
            categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if cleaned_df[col].isnull().sum() > 0:
                    mode_value = cleaned_df[col].mode().iloc[0] if not cleaned_df[col].mode().empty else 'Unknown'
                    filled_count = cleaned_df[col].isnull().sum()
                    cleaned_df[col].fillna(mode_value, inplace=True)
                    cleaning_log.append(f"✓ Filled {filled_count:,} missing values in '{col}' with mode ('{mode_value}')")
        
        # Remove columns with high missing values
        high_missing_threshold = cleaning_options.get('high_missing_threshold', 80)
        cols_to_drop = []
        for col in cleaned_df.columns:
            missing_pct = (cleaned_df[col].isnull().sum() / len(cleaned_df)) * 100 if len(cleaned_df) > 0 else 0
            if missing_pct > high_missing_threshold:
                cols_to_drop.append(col)
        
        if cols_to_drop and cleaning_options.get('drop_high_missing', False):
            cleaned_df = cleaned_df.drop(columns=cols_to_drop)
            cleaning_log.append(f"✓ Dropped {len(cols_to_drop)} columns with >{high_missing_threshold}% missing values: {', '.join(cols_to_drop)}")
        
        # Convert date columns
        date_columns = cleaning_options.get('date_columns', [])
        for col in date_columns:
            if col in cleaned_df.columns:
                try:
                    cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
                    cleaning_log.append(f"✓ Converted '{col}' to datetime")
                except Exception as e:
                    cleaning_log.append(f"✗ Failed to convert '{col}' to datetime: {str(e)}")
        
        # Standardize column names
        if cleaning_options.get('standardize_columns', False):
            old_columns = cleaned_df.columns.tolist()
            cleaned_df.columns = [col.lower().strip().replace(' ', '_') for col in cleaned_df.columns]
            cleaning_log.append("✓ Standardized column names (lowercase, underscores)")
        
        # Final summary
        final_shape = cleaned_df.shape
        rows_removed = original_shape[0] - final_shape[0]
        cols_removed = original_shape[1] - final_shape[1]
        
        if rows_removed > 0 or cols_removed > 0:
            cleaning_log.append(f"📊 Summary: {rows_removed:,} rows and {cols_removed} columns removed")
            cleaning_log.append(f"📊 Final dataset: {final_shape[0]:,} rows × {final_shape[1]} columns")
        
        return cleaned_df, cleaning_log
    
    except Exception as e:
        cleaning_log.append(f"✗ Error during cleaning: {str(e)}")
        return df, cleaning_log

##############################################
# Auto-Clean Suggestions                    #
##############################################

def get_auto_clean_suggestions(df):
    """Generate automatic cleaning suggestions based on data analysis"""
    suggestions = {
        'remove_duplicates': False,
        'missing_strategy': 'keep',
        'drop_high_missing': False,
        'high_missing_threshold': 80,
        'date_columns': [],
        'categorical_columns': [],
        'standardize_columns': False,
        'reasoning': []
    }
    
    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        duplicate_pct = (duplicate_count / len(df)) * 100
        suggestions['remove_duplicates'] = True
        suggestions['reasoning'].append(f"Remove {duplicate_count:,} duplicates ({duplicate_pct:.1f}% of data)")
    
    # Analyze missing values
    high_missing_cols = []
    moderate_missing_cols = []
    
    for col in df.columns:
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        if missing_pct > 70:
            high_missing_cols.append(col)
        elif missing_pct > 5:
            moderate_missing_cols.append(col)
    
    if high_missing_cols:
        suggestions['drop_high_missing'] = True
        suggestions['reasoning'].append(f"Drop {len(high_missing_cols)} columns with >70% missing data")
    
    if moderate_missing_cols:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if any(col in numeric_cols for col in moderate_missing_cols):
            suggestions['missing_strategy'] = 'fill_numeric'
            suggestions['reasoning'].append("Fill missing numeric values with median")
        elif any(col in categorical_cols for col in moderate_missing_cols):
            suggestions['missing_strategy'] = 'fill_categorical'
            suggestions['reasoning'].append("Fill missing categorical values with mode")
    
    # Detect date columns
    date_candidates = []
    for col in df.select_dtypes(include=['object']).columns:
        sample_values = df[col].dropna().head(10).astype(str)
        date_patterns = [r'\d{4}-\d{2}-\d{2}', r'\d{2}/\d{2}/\d{4}', r'\d{2}-\d{2}-\d{4}']
        
        date_matches = sum(1 for val in sample_values 
                          for pattern in date_patterns 
                          if re.match(pattern, str(val)))
        
        if date_matches > len(sample_values) * 0.7:
            date_candidates.append(col)
    
    if date_candidates:
        suggestions['date_columns'] = date_candidates
        suggestions['reasoning'].append(f"Convert {len(date_candidates)} columns to datetime format")
    
    return suggestions

##############################################
# Q&A Functions                             #
##############################################

def generate_data_context(df):
    """Generate enhanced context about the dataset for the AI"""
    context = f"""
Dataset Overview:
- Total rows: {len(df):,}
- Total columns: {len(df.columns)}
- Columns: {', '.join(df.columns)}

Column Details:
"""
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        unique_count = df[col].nunique()
        
        context += f"- {col}: {dtype}, {null_count} nulls, {unique_count} unique values"
        
        if dtype in ['int64', 'float64']:
            try:
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                total_val = df[col].sum()
                context += f", range: {min_val:.2f} to {max_val:.2f}, mean: {mean_val:.2f}, total: {total_val:,.2f}"
            except:
                context += f", numeric column"
        elif dtype == 'object' and unique_count < 20:  # Show more details for key categorical columns
            value_counts = df[col].value_counts().head(10)
            context += f", top values: {dict(value_counts)}"
        
        context += "\n"
    
    # Add key statistics
    if 'amount' in df.columns:
        total_amount = df['amount'].sum()
        avg_amount = df['amount'].mean()
        context += f"\nKey Statistics:\n"
        context += f"- Total Amount: {total_amount:,.2f}\n"
        context += f"- Average Amount: {avg_amount:,.2f}\n"
        context += f"- Number of Transactions: {len(df):,}\n"
        
        # Add breakdown by main categories
        if 'productcategory' in df.columns:
            category_breakdown = df.groupby('productcategory')['amount'].sum().sort_values(ascending=False)
            context += f"- Product Category Breakdown:\n"
            for cat, amt in category_breakdown.head().items():
                context += f"  * {cat}: {amt:,.2f}\n"
    
    return context

def perform_actual_calculation(question, df):
    """Perform actual calculations based on the question and provide real results"""
    question_lower = question.lower()
    results = {}
    
    try:
        # Detect what kind of calculation is needed based on actual column names
        print(f"Available columns: {list(df.columns)}")
        print(f"Question: {question}")
        
        # Check for highest/maximum transaction amount
        if ('highest' in question_lower or 'maximum' in question_lower) and 'transaction' in question_lower and 'amount' in question_lower:
            if 'amount' in df.columns:
                max_amount_idx = df['amount'].idxmax()
                max_amount = df['amount'].max()
                max_transaction = df.loc[max_amount_idx]
                
                results['highest_transaction_amount'] = max_amount
                results['highest_transaction_details'] = {}
                
                # Include all relevant transaction details
                for col in df.columns:
                    if col in ['transactionid', 'customerid', 'transactiondate', 'transactiontype', 
                              'productcategory', 'productsubcategory', 'branchcity', 'amount']:
                        results['highest_transaction_details'][col] = max_transaction[col]
                
                print(f"Highest transaction: {max_amount} with details: {results['highest_transaction_details']}")
        
        # Check for customer with highest transaction
        elif ('customer' in question_lower and 'highest' in question_lower and 
              ('transaction' in question_lower or 'amount' in question_lower)):
            if 'customerid' in df.columns and 'amount' in df.columns:
                # Find customer with highest single transaction
                max_amount_idx = df['amount'].idxmax()
                max_transaction = df.loc[max_amount_idx]
                customer_id = max_transaction['customerid']
                max_amount = max_transaction['amount']
                
                results['customer_highest_transaction'] = customer_id
                results['customer_highest_amount'] = max_amount
                results['customer_transaction_details'] = {}
                
                # Get all details for this transaction
                for col in df.columns:
                    if col in ['transactionid', 'customerid', 'transactiondate', 'transactiontype', 
                              'productcategory', 'amount', 'customersegment', 'monthlyincome']:
                        results['customer_transaction_details'][col] = max_transaction[col]
                
                # Also get total transactions for this customer
                customer_transactions = df[df['customerid'] == customer_id]
                results['customer_total_transactions'] = len(customer_transactions)
                results['customer_total_amount'] = customer_transactions['amount'].sum()
                
                print(f"Customer {customer_id} has highest transaction: {max_amount}")
        
        # Check for product category analysis
        elif 'product' in question_lower and 'category' in question_lower and 'amount' in question_lower:
            if 'productcategory' in df.columns and 'amount' in df.columns:
                category_totals = df.groupby('productcategory')['amount'].sum().sort_values(ascending=False)
                results['product_category_amounts'] = category_totals.to_dict()
                results['highest_category'] = category_totals.index[0]
                results['highest_amount'] = category_totals.iloc[0]
                print(f"Product category totals: {category_totals.to_dict()}")
        
        # Check for transaction type analysis
        elif 'transaction' in question_lower and 'type' in question_lower:
            if 'transactiontype' in df.columns and 'amount' in df.columns:
                type_totals = df.groupby('transactiontype')['amount'].sum().sort_values(ascending=False)
                results['transaction_type_amounts'] = type_totals.to_dict()
        
        # Check for branch analysis
        elif 'branch' in question_lower and 'city' in question_lower:
            if 'branchcity' in df.columns and 'amount' in df.columns:
                branch_totals = df.groupby('branchcity')['amount'].sum().sort_values(ascending=False)
                results['branch_amounts'] = branch_totals.to_dict()
        
        # Check for customer segment analysis
        elif 'customer' in question_lower and 'segment' in question_lower:
            if 'customersegment' in df.columns and 'amount' in df.columns:
                segment_totals = df.groupby('customersegment')['amount'].sum().sort_values(ascending=False)
                results['segment_amounts'] = segment_totals.to_dict()
        
        # Generic amount analysis or insights request
        elif 'amount' in question_lower or 'total' in question_lower or 'insights' in question_lower:
            if 'amount' in df.columns:
                results['total_amount'] = df['amount'].sum()
                results['average_amount'] = df['amount'].mean()
                results['transaction_count'] = len(df)
                results['max_transaction_amount'] = df['amount'].max()
                results['min_transaction_amount'] = df['amount'].min()
                
                # Get highest transaction details
                max_amount_idx = df['amount'].idxmax()
                max_transaction = df.loc[max_amount_idx]
                results['highest_transaction_details'] = {
                    'amount': max_transaction['amount'],
                    'customer_id': max_transaction.get('customerid', 'N/A'),
                    'transaction_type': max_transaction.get('transactiontype', 'N/A'),
                    'product_category': max_transaction.get('productcategory', 'N/A'),
                    'branch_city': max_transaction.get('branchcity', 'N/A'),
                    'date': max_transaction.get('transactiondate', 'N/A')
                }
                
                # Find the main categorical column for grouping
                categorical_cols = df.select_dtypes(include=['object']).columns
                main_cat_col = None
                
                if 'productcategory' in categorical_cols:
                    main_cat_col = 'productcategory'
                elif 'transactiontype' in categorical_cols:
                    main_cat_col = 'transactiontype'
                elif 'branchcity' in categorical_cols:
                    main_cat_col = 'branchcity'
                
                if main_cat_col:
                    group_totals = df.groupby(main_cat_col)['amount'].sum().sort_values(ascending=False)
                    results[f'{main_cat_col}_breakdown'] = group_totals.to_dict()
        
        return results
    except Exception as e:
        print(f"Calculation error: {e}")
        import traceback
        traceback.print_exc()
        return {}

def answer_data_question(question, df):
    """Use AI to answer questions about the data with actual calculations"""
    data_context = generate_data_context(df)
    
    # Perform actual calculations
    calculations = perform_actual_calculation(question, df)
    calculation_text = ""
    if calculations:
        calculation_text = f"\nACTUAL CALCULATED RESULTS:\n"
        for key, value in calculations.items():
            if isinstance(value, dict):
                calculation_text += f"{key}:\n"
                for k, v in value.items():
                    if isinstance(v, (int, float)):
                        calculation_text += f"  - {k}: {v:,.2f}\n"
                    else:
                        calculation_text += f"  - {k}: {v}\n"
            elif isinstance(value, (int, float)):
                calculation_text += f"{key}: {value:,.2f}\n"
            else:
                calculation_text += f"{key}: {value}\n"
        calculation_text += "\n"
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a data analyst assistant. Answer questions about the provided dataset by providing actual numerical results and insights. "
                "CRITICAL: ALWAYS use the EXACT numbers and details from the 'ACTUAL CALCULATED RESULTS' section. "
                "DO NOT calculate anything yourself - only use the pre-calculated values provided. "
                "When you see transaction details, include all the specific information provided. "
                "When you see pre-calculated results, use those exact numbers in your response. "
                "Provide clear, actionable insights with real data from the dataset. "
                "Format your response professionally with clear sections and bullet points. "
                "Focus on answering the question directly with specific numerical results and details from the calculations."
            )
        },
        {
            "role": "user",
            "content": f"Dataset Context:\n{data_context}\n{calculation_text}\nQuestion: {question}\n\nProvide the actual calculated answer using ONLY the numbers and details from the ACTUAL CALCULATED RESULTS section above. Include all specific transaction details when available. Do not perform any calculations yourself."
        }
    ]
    
    try:
        client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2024-02-15-preview"
        )
        
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=messages,
            temperature=0.1,  # Lower temperature for more factual responses
            max_tokens=2000
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error generating answer: {str(e)}"

##############################################
# KPI and Chart Parsing Functions          #
##############################################

def extract_kpis_from_answer(answer, df=None):
    """Extract KPIs using multiple patterns for any LLM output style."""
    if not answer:
        return []
        
    kpis = []
    seen = set()
    patterns = [
        r"\*\*([^\*]+)\*\*:? [`']?(sum|avg|count|min|max)\(([a-zA-Z0-9_\s\/]+)\)[`']?",
        r"([A-Za-z ]+)\s*\(\s*([a-zA-Z0-9_\s]+)\s*,\s*aggregation:\s*([a-z]+)\s*\)",
        r"([A-Za-z ]+): (sum|avg|count|min|max)\(([a-zA-Z0-9_\s\/]+)\)",
        r"(sum|avg|count|min|max)\(([a-zA-Z0-9_\s\/]+)\)"
    ]
    
    if df is not None:
        df_work = df.copy()
        # Create a mapping for column lookup
        col_lookup = {col.lower().strip(): col for col in df_work.columns}
    
    try:
        for pattern in patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            for m in matches:
                try:
                    if len(m) == 3:
                        label, agg, col = m
                    elif len(m) == 2:
                        agg, col = m
                        label = f"{agg.capitalize()} {col.capitalize()}"
                    else:
                        continue
                    
                    # Create unique key using original label to avoid duplicates
                    key = (label.strip().lower(), agg.strip().lower(), col.strip().lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    
                    col = col.lower().strip()
                    agg = agg.lower().strip()
                    label = label.strip()
                    value, isnum = None, False
                    
                    if df is not None:
                        # Find matching column
                        actual_col = None
                        if col in col_lookup:
                            actual_col = col_lookup[col]
                        else:
                            # Try fuzzy matching
                            for df_col in df_work.columns:
                                if col in df_col.lower() or df_col.lower() in col:
                                    actual_col = df_col
                                    break
                        
                        if actual_col and actual_col in df_work.columns:
                            try:
                                if agg == 'sum':
                                    value = round(df_work[actual_col].sum(), 2)
                                    isnum = True
                                elif agg == 'avg':
                                    value = round(df_work[actual_col].mean(), 2)
                                    isnum = True
                                elif agg == 'count':
                                    value = int(df_work[actual_col].nunique()) if df_work[actual_col].dtype == 'object' else len(df_work)
                                    isnum = True
                                elif agg == 'min':
                                    value = round(df_work[actual_col].min(), 2)
                                    isnum = True
                                elif agg == 'max':
                                    value = round(df_work[actual_col].max(), 2)
                                    isnum = True
                            except Exception:
                                value = f"{agg}({col})"
                        else:
                            value = f"{agg}({col})"
                    else:
                        value = f"{agg}({col})"
                    
                    # Skip axis-related labels and ensure unique KPIs
                    if label.lower() not in ['x-axis', 'y-axis', 'values', 'labels']:
                        # Additional check to avoid duplicate KPIs with same calculation
                        is_duplicate = False
                        for existing_kpi in kpis:
                            if (existing_kpi['label'].lower() == label.lower() or 
                                (existing_kpi['value'] == value and existing_kpi['isnum'] == isnum)):
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            kpis.append({'label': label, 'value': value, 'isnum': isnum})
                
                except Exception as e:
                    print(f"Error processing KPI match {m}: {e}")
                    continue
    
    except Exception as e:
        print(f"Error extracting KPIs: {e}")
    
    return kpis

def parse_ai_answer_for_charts(answer):
    """Parse AI answer for chart specifications with improved pattern matching"""
    if not answer:
        return []
    
    charts = []
    patterns = [
        # Pattern 1: **Chart Type**: ... **X-Axis**: ... **Y-Axis**: ...
        r'\*\*Chart Type\*\*:\s*([^*\n]+).*?\*\*X[- ]?Axis\*\*:\s*([a-zA-Z0-9_\s]+).*?\*\*Y[- ]?Axis\*\*:\s*(sum|avg|count|min|max)\(([a-zA-Z0-9_\s]+)\)',
        # Pattern 2: Chart Type: ... X-Axis: ... Y-Axis: ...
        r'(Bar Chart|Line Chart|Pie Chart|Horizontal Bar Chart|Scatter Plot|Histogram).*?X[- ]?Axis:\s*([a-zA-Z0-9_\s]+).*?Y[- ]?Axis:\s*(sum|avg|count|min|max)\(([a-zA-Z0-9_\s]+)\)',
        # Pattern 3: X-Axis: ... Y-Axis: ... (default to bar chart)
        r'X[- ]?Axis:\s*([a-zA-Z0-9_\s]+).*?Y[- ]?Axis:\s*(sum|avg|count|min|max)\(([a-zA-Z0-9_\s]+)\)',
        # Pattern 4: Simple aggregation pattern
        r'(sum|avg|count|min|max)\(([a-zA-Z0-9_\s]+)\)\s+by\s+([a-zA-Z0-9_\s]+)'
    ]
    
    try:
        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, answer, re.IGNORECASE | re.DOTALL)
            
            for match in matches:
                try:
                    if i == 0 and len(match) == 4:  # Pattern 1
                        chart_type, x, agg, y = match
                    elif i == 1 and len(match) == 4:  # Pattern 2
                        chart_type, x, agg, y = match
                    elif i == 2 and len(match) == 3:  # Pattern 3
                        x, agg, y = match
                        chart_type = "bar chart"
                    elif i == 3 and len(match) == 3:  # Pattern 4
                        agg, y, x = match
                        chart_type = "bar chart"
                    else:
                        continue
                    
                    # Clean and validate the extracted values
                    chart_type = chart_type.strip().lower()
                    x = x.strip().lower()
                    y = y.strip().lower()
                    agg = agg.strip().lower()
                    
                    # Skip invalid combinations
                    if x == 'avg' or y == 'avg' or x == agg or y == agg:
                        continue
                    
                    # Skip if column names are too generic or invalid
                    if x in ['column', 'field', 'data'] or y in ['column', 'field', 'data']:
                        continue
                    
                    charts.append({
                        'chart_type': chart_type,
                        'x': x,
                        'agg': agg,
                        'y': y,
                    })
                    
                except Exception as e:
                    print(f"Error processing match {match}: {e}")
                    continue
        
        # Remove duplicates while preserving order
        unique_charts = []
        seen = set()
        for chart in charts:
            chart_key = (chart['x'], chart['y'], chart['agg'])
            if chart_key not in seen:
                seen.add(chart_key)
                unique_charts.append(chart)
        
        return unique_charts
    
    except Exception as e:
        print(f"Error parsing chart specifications: {e}")
        return []

def parse_markdown_tables(answer):
    """Parse markdown tables from AI answer"""
    if not answer:
        return []
        
    tables = []
    try:
        table_lines = []
        in_table = False
        
        for line in answer.splitlines():
            if "|" in line:
                table_lines.append(line)
                in_table = True
            elif in_table:
                tables.append("\n".join(table_lines))
                table_lines = []
                in_table = False
        
        if table_lines:
            tables.append("\n".join(table_lines))
        
        html_tables = []
        for t in tables:
            try:
                rows = [r.strip() for r in t.splitlines() if "|" in r]
                if not rows:
                    continue
                
                html = "<table class='table table-striped table-bordered'>"
                for i, row in enumerate(rows):
                    cols = [c.strip() for c in row.split("|") if c.strip()]
                    if not cols:
                        continue
                    
                    if i == 0:
                        html += "<thead><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr></thead><tbody>"
                    else:
                        html += "<tr>" + "".join(f"<td>{c}</td>" for c in cols) + "</tr>"
                
                html += "</tbody></table>"
                html_tables.append(html)
            except Exception as e:
                print(f"Error processing table: {e}")
                continue
        
        return html_tables
    
    except Exception as e:
        print(f"Error parsing markdown tables: {e}")
        return []

#############################
#         FLASK ROUTES      #
#############################

# Replace your current index route with this one:

@app.route('/', methods=['GET', 'POST'])
def index():
    table, error, filename = None, None, None
    
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith(('.csv', '.xlsx')):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            try:
                df = pd.read_csv(filepath) if file.filename.endswith('.csv') else pd.read_excel(filepath)
                
                # Store original data first
                DATASTORE['original_df'] = df.copy()
                DATASTORE['original_columns'] = df.columns.tolist()
                
                # Enhanced column cleaning with better mapping
                df, column_mapping = clean_column_names_enhanced(df)
                
                # Store cleaned data
                DATASTORE['df'] = df
                
                filename = file.filename
                table = df.head(20).to_html(classes='table table-striped', index=False, border=0)
                
                # Log column cleaning with more detail
                print(f"Column names cleaned: {column_mapping}")
                print(f"DataFrame shape: {df.shape}")
                print(f"Column data types: {df.dtypes.to_dict()}")
                
            except Exception as e:
                error = f"Failed to read file: {str(e)}"
        else:
            error = "Only CSV or Excel files allowed."
    
    # 🔥 FRESH START CHANGE: Clear data on GET requests (fresh visits)
    else:
        DATASTORE.clear()  # Clear all previous data for fresh start
        table = None
    
    return render_template('index.html', table=table, error=error, filename=filename)

@app.route('/data-quality')
def data_quality():
    """Enhanced data quality analysis page"""
    df = DATASTORE.get('df')
    if df is None:
        return render_template('data_quality.html', error="No data loaded. Please upload a file first.")
    
    # Generate comprehensive quality report
    quality_report = analyze_data_quality(df)
    
    # Get auto-cleaning suggestions
    auto_suggestions = get_auto_clean_suggestions(df)
    
    return render_template('data_quality.html', 
                         quality_report=quality_report, 
                         auto_suggestions=auto_suggestions,
                         error=None)

@app.route('/clean-data', methods=['POST'])
def clean_data():
    """Enhanced data cleaning with better feedback"""
    df = DATASTORE.get('df')
    if df is None:
        return jsonify({'error': 'No data loaded'})
    
    cleaning_options = request.json
    cleaned_df, cleaning_log = clean_dataset(df, cleaning_options)
    
    # Update datastore with cleaned data
    DATASTORE['df'] = cleaned_df
    DATASTORE['cleaning_log'] = cleaning_log
    
    # Generate new quality report
    new_quality_report = analyze_data_quality(cleaned_df)
    
    return jsonify({
        'success': True,
        'cleaning_log': cleaning_log,
        'old_shape': df.shape,
        'new_shape': cleaned_df.shape,
        'new_quality_score': new_quality_report['quality_score'],
        'new_quality_status': new_quality_report['quality_status']
    })

@app.route('/auto-clean', methods=['POST'])
def auto_clean():
    """Apply automatic cleaning suggestions"""
    df = DATASTORE.get('df')
    if df is None:
        return jsonify({'error': 'No data loaded'})
    
    # Get auto suggestions and apply them
    auto_suggestions = get_auto_clean_suggestions(df)
    cleaned_df, cleaning_log = clean_dataset(df, auto_suggestions)
    
    # Update datastore
    DATASTORE['df'] = cleaned_df
    DATASTORE['cleaning_log'] = cleaning_log
    
    # Generate new quality report
    new_quality_report = analyze_data_quality(cleaned_df)
    
    return jsonify({
        'success': True,
        'cleaning_log': cleaning_log,
        'applied_suggestions': auto_suggestions['reasoning'],
        'old_shape': df.shape,
        'new_shape': cleaned_df.shape,
        'new_quality_score': new_quality_report['quality_score'],
        'new_quality_status': new_quality_report['quality_status']
    })

@app.route('/ask-question', methods=['POST'])
def ask_question():
    """Answer questions about the data using AI"""
    df = DATASTORE.get('df')
    if df is None:
        return jsonify({'error': 'No data loaded'})
    
    question = request.json.get('question', '')
    if not question:
        return jsonify({'error': 'No question provided'})
    
    answer = answer_data_question(question, df)
    
    # Extract any KPIs and charts from the answer
    kpis = extract_kpis_from_answer(answer, df)
    chart_specs = parse_ai_answer_for_charts(answer)
    
    chart_urls = []
    if chart_specs:
        for spec in chart_specs:
            # Validate spec before generating chart
            validated_spec = validate_chart_spec_in_app(spec, df)
            if validated_spec:
                url = generate_chart_from_ai(validated_spec, df)
                if url:
                    chart_urls.append(url)
    
    return jsonify({
        'answer': answer,
        'answer_html': markdown2.markdown(answer),
        'kpis': kpis,
        'chart_urls': chart_urls
    })

@app.route('/dashboard')
def dashboard():
    """Generate and display dashboard with improved chart generation"""
    df = DATASTORE.get('df')
    if df is None:
        return render_template('dashboard.html', 
                             error="No data loaded. Please upload a file first.",
                             dashboard_plan_html="", kpis=[], chart_urls=[], tables=[])
    
    # Enhanced debugging information
    print(f"=== DASHBOARD GENERATION DEBUG ===")
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types: {df.dtypes.to_dict()}")
    
    # Run column matching debug
    debug_column_matching(df)
    
    # Create sample data for AI analysis (first 20 rows)
    sample = df.head(20).to_csv(index=False)
    
    # Enhanced prompt with explicit column information
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert business intelligence analyst. Create a comprehensive executive dashboard plan.\n\n"
                "IMPORTANT FORMATTING REQUIREMENTS:\n"
                "- For KPIs, use this EXACT format: **KPI Name**: aggregation(column_name)\n"
                "- For charts, use this EXACT format:\n"
                "  **Chart Type**: Bar Chart (or Pie Chart, Line Chart, Horizontal Bar Chart)\n"
                "  **X-Axis**: column_name\n"
                "  **Y-Axis**: aggregation(column_name)\n\n"
                "Example:\n"
                "**Total Revenue**: sum(amount)\n"
                "**Chart Type**: Bar Chart\n"
                "**X-Axis**: branch\n"
                "**Y-Axis**: sum(amount)\n\n"
                f"AVAILABLE COLUMNS: {', '.join(df.columns)}\n"
                f"NUMERIC COLUMNS: {', '.join(df.select_dtypes(include=[np.number]).columns)}\n"
                f"CATEGORICAL COLUMNS: {', '.join(df.select_dtypes(include=['object', 'category']).columns)}\n\n"
                "RULES:\n"
                "- ONLY use exact column names shown above\n"
                "- Y-Axis must use numeric columns (quantity, time, amount)\n"
                "- Use aggregations: sum, avg, count, min, max\n"
                "- Create 2-3 charts maximum\n"
                "- Focus on business insights\n"
            )
        },
        {
            "role": "user",
            "content": f"Dataset sample:\n{sample}\n\nCreate a dashboard plan using the EXACT formatting requirements above."
        }
    ]
    
    try:
        client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2024-02-15-preview"
        )
        
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=messages,
            temperature=0.3,
            max_tokens=1500
        )
        
        dashboard_plan = response.choices[0].message.content.strip()
        print(f"Dashboard plan generated: {dashboard_plan[:200]}...")
        
    except Exception as e:
        error_msg = f"Error contacting Azure OpenAI: {str(e)}"
        print(error_msg)
        return render_template('dashboard.html', 
                             error=error_msg,
                             dashboard_plan_html="", kpis=[], chart_urls=[], tables=[])
    
    # Extract KPIs with improved handling
    kpis = extract_kpis_from_answer(dashboard_plan, df)
    print(f"Extracted KPIs: {len(kpis)} - {[kpi['label'] for kpi in kpis]}")
    
    # Extract and validate chart specifications
    chart_specs = parse_ai_answer_for_charts(dashboard_plan)
    print(f"Raw chart specifications: {chart_specs}")
    
    # Generate charts with enhanced validation
    chart_urls = []
    successful_charts = 0
    
    for i, spec in enumerate(chart_specs):
        try:
            print(f"\n--- Generating Chart {i+1} ---")
            print(f"Original spec: {spec}")
            
            # Validate and fix specification
            validated_spec = validate_chart_spec_in_app(spec, df)
            if validated_spec:
                print(f"Validated spec: {validated_spec}")
                url = generate_chart_from_ai(validated_spec, df)
                if url:
                    chart_urls.append(url)
                    successful_charts += 1
                    print(f"Chart {i+1} generated successfully: {url}")
                else:
                    print(f"Chart {i+1} generation failed")
            else:
                print(f"Chart {i+1} validation failed")
        except Exception as e:
            print(f"Error generating chart {i+1}: {e}")
            continue
    
    print(f"Generated {successful_charts} out of {len(chart_specs)} charts from AI specs")
    
    # If no charts were generated, create default charts
    if not chart_urls:
        try:
            print("No charts generated from AI specs, creating default charts...")
            default_charts = create_default_charts(df)
            if default_charts:
                chart_urls.extend(default_charts)
                print(f"Created {len(default_charts)} default charts")
            else:
                print("Failed to create default charts")
        except Exception as e:
            print(f"Error creating default charts: {e}")
    
    # Extract markdown tables
    html_tables = parse_markdown_tables(dashboard_plan)
    
    # Convert dashboard plan to HTML
    dashboard_plan_html = markdown2.markdown(dashboard_plan)
    
    print(f"Final dashboard: {len(kpis)} KPIs, {len(chart_urls)} charts, {len(html_tables)} tables")
    print("=== END DASHBOARD DEBUG ===")
    
    return render_template('dashboard.html',
                         dashboard_plan_html=dashboard_plan_html,
                         kpis=kpis,
                         chart_urls=chart_urls,
                         tables=html_tables,
                         error=None)

##############################################
# DIAGNOSTIC ROUTES                          #
##############################################

@app.route('/check-env')
def check_env():
    """Check if environment variables are loaded correctly"""
    return jsonify({
        'AZURE_OPENAI_ENDPOINT': os.getenv("AZURE_OPENAI_ENDPOINT"),
        'AZURE_OPENAI_DEPLOYMENT': os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        'AZURE_OPENAI_API_KEY_PREVIEW': os.getenv("AZURE_OPENAI_API_KEY")[:10] + "..." if os.getenv("AZURE_OPENAI_API_KEY") else None,
        'environment_file_exists': os.path.exists('.env'),
        'current_directory': os.getcwd(),
        'env_file_path': os.path.abspath('.env') if os.path.exists('.env') else 'Not found'
    })

@app.route('/azure-debug')
def azure_debug():
    """Complete Azure OpenAI debugging"""
    
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    debug_info = {
        'configuration': {
            'endpoint': endpoint,
            'deployment': deployment,
            'has_api_key': bool(api_key),
            'api_key_length': len(api_key) if api_key else 0
        },
        'issues': [],
        'fixes': [],
        'test_results': {}
    }
    
    # Check each component
    if not endpoint:
        debug_info['issues'].append("AZURE_OPENAI_ENDPOINT not set")
        debug_info['fixes'].append("Add AZURE_OPENAI_ENDPOINT to your .env file")
    else:
        # Validate endpoint format
        if not endpoint.startswith('https://'):
            debug_info['issues'].append("Endpoint should start with https://")
        
        if '.openai.azure.com' not in endpoint:
            debug_info['issues'].append("Endpoint should contain '.openai.azure.com'")
            debug_info['fixes'].append(f"Change endpoint to: https://[your-resource-name].openai.azure.com/")
        
        if not endpoint.endswith('/'):
            debug_info['issues'].append("Endpoint should end with /")
            debug_info['fixes'].append(f"Add trailing slash: {endpoint}/")
    
    if not deployment:
        debug_info['issues'].append("AZURE_OPENAI_DEPLOYMENT not set")
        debug_info['fixes'].append("Add AZURE_OPENAI_DEPLOYMENT to your .env file")
    
    if not api_key:
        debug_info['issues'].append("AZURE_OPENAI_API_KEY not set")
        debug_info['fixes'].append("Add AZURE_OPENAI_API_KEY to your .env file")
    
    # Test different API versions if basic config is present
    if endpoint and deployment and api_key:
        api_versions = ["2024-02-15-preview", "2023-12-01-preview", "2023-05-15"]
        
        for api_version in api_versions:
            test_url = f"{endpoint}openai/deployments/{deployment}/chat/completions?api-version={api_version}"
            
            try:
                headers = {
                    "Content-Type": "application/json",
                    "api-key": api_key
                }
                
                data = {
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 1
                }
                
                response = requests.post(test_url, headers=headers, json=data, timeout=5)
                
                debug_info['test_results'][api_version] = {
                    'status_code': response.status_code,
                    'url': test_url,
                    'success': response.status_code == 200,
                    'error': response.text if response.status_code != 200 else None
                }
                
                if response.status_code == 200:
                    debug_info['fixes'].append(f"✅ Working configuration found with API version {api_version}")
                    break
                    
            except Exception as e:
                debug_info['test_results'][api_version] = {
                    'status_code': 'ERROR',
                    'error': str(e),
                    'success': False
                }
    
    return jsonify(debug_info)

@app.route('/test-openai-simple')
def test_openai_simple():
    """Simple test of OpenAI configuration"""
    try:
        client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2024-02-15-preview"
        )
        
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        
        return jsonify({
            'status': 'SUCCESS',
            'response': response.choices[0].message.content,
            'model_used': os.getenv("AZURE_OPENAI_DEPLOYMENT")
        })
        
    except Exception as e:
        return jsonify({
            'status': 'ERROR',
            'error': str(e),
            'error_type': type(e).__name__,
            'endpoint': os.getenv("AZURE_OPENAI_ENDPOINT"),
            'deployment': os.getenv("AZURE_OPENAI_DEPLOYMENT")
        })

@app.route('/create-env-file')
def create_env_file():
    """Create a template .env file"""
    env_content = """# Azure OpenAI Configuration - Replace with your actual values
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your-deployment-name

# How to get these values:
# 1. Go to https://portal.azure.com
# 2. Find your OpenAI resource
# 3. Go to "Keys and Endpoint" - copy the endpoint and key
# 4. Go to "Model deployments" - copy the deployment name exactly
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        
        return jsonify({
            'status': 'success',
            'message': '.env file created successfully',
            'path': os.path.abspath('.env'),
            'next_steps': [
                'Edit the .env file with your actual Azure values',
                'Restart your Flask application',
                'Test with /check-env'
            ]
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        })

@app.route('/dashboard-working')
def dashboard_working():
    """Working dashboard without AI dependency"""
    df = DATASTORE.get('df')
    if df is None:
        return render_template('dashboard.html', 
                             error="No data loaded. Please upload a file first.",
                             dashboard_plan_html="", kpis=[], chart_urls=[], tables=[])
    
    try:
        # Create KPIs from actual data
        kpis = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Basic statistics
        kpis.append({'label': 'Total Records', 'value': f'{len(df):,}', 'isnum': True})
        kpis.append({'label': 'Total Columns', 'value': f'{len(df.columns)}', 'isnum': True})
        
        # Numeric column statistics
        for col in numeric_cols[:3]:  # Top 3 numeric columns
            total = df[col].sum()
            avg = df[col].mean()
            kpis.append({'label': f'Total {col.title()}', 'value': f'{total:,.2f}', 'isnum': True})
            kpis.append({'label': f'Average {col.title()}', 'value': f'{avg:.2f}', 'isnum': True})
        
        # Generate charts
        chart_urls = []
        try:
            default_charts = create_default_charts(df)
            chart_urls = default_charts[:3]  # Limit to 3 charts
        except Exception as e:
            print(f"Chart generation error: {e}")
        
        # Simple dashboard plan
        categorical_cols = df.select_dtypes(include=['object']).columns
        dashboard_plan_html = f"""
        <h2>Dataset Overview</h2>
        <div class="row">
            <div class="col-md-6">
                <h4>Basic Information</h4>
                <ul>
                    <li>Total Rows: {len(df):,}</li>
                    <li>Total Columns: {len(df.columns)}</li>
                    <li>Numeric Columns: {len(numeric_cols)}</li>
                    <li>Text Columns: {len(categorical_cols)}</li>
                </ul>
            </div>
            <div class="col-md-6">
                <h4>Column Summary</h4>
                <p><strong>Numeric:</strong> {', '.join(numeric_cols[:5])}</p>
                <p><strong>Categorical:</strong> {', '.join(categorical_cols[:5])}</p>
            </div>
        </div>
        """
        
        return render_template('dashboard.html',
                             dashboard_plan_html=dashboard_plan_html,
                             kpis=kpis[:8],
                             chart_urls=chart_urls,
                             tables=[],
                             error=None)
    
    except Exception as e:
        return render_template('dashboard.html',
                             error=f"Dashboard generation failed: {str(e)}",
                             dashboard_plan_html="", kpis=[], chart_urls=[], tables=[])

##############################################
# ADDITIONAL ROUTES                          #
##############################################

@app.route('/debug-data', methods=['GET'])
def debug_data():
    """Debug route to check data consistency"""
    df = DATASTORE.get('df')
    if df is None:
        return jsonify({'error': 'No cleaned data available'})
    
    try:
        debug_info = {
            'dataframe_shape': df.shape,
            'columns': list(df.columns),
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'sample_data': df.head().to_dict(),
            'column_mappings': DATASTORE.get('column_mapping', {}),
        }
        
        # Add data summary
        debug_info['data_summary'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'text_columns': len(df.select_dtypes(include=['object']).columns),
            'missing_values_total': int(df.isnull().sum().sum()),
            'duplicates': int(df.duplicated().sum())
        }
        
        return jsonify(debug_info)
    
    except Exception as e:
        import traceback
        return jsonify({
            'error': f'Debug failed: {str(e)}',
            'traceback': traceback.format_exc()
        })

@app.route('/suggest-data-types', methods=['POST'])
def suggest_data_types():
    """AI-powered data type suggestions with enhanced error handling"""
    try:
        df = DATASTORE.get('df')
        if df is None:
            return jsonify({'success': False, 'error': 'No data loaded'})
        
        suggestions = {}
        
        print(f"Analyzing data types for {len(df.columns)} columns...")  # Debug log
        
        for column in df.columns:
            current_type = str(df[column].dtype)
            sample_values = df[column].dropna().head(10).astype(str).tolist()
            
            print(f"Column '{column}': {current_type}, sample: {sample_values[:3]}")  # Debug log
            
            # Auto-detect potential data types
            if current_type == 'object':
                # Check for dates
                date_patterns = [
                    r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                    r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                    r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
                ]
                
                date_matches = 0
                for val in sample_values:
                    for pattern in date_patterns:
                        if re.match(pattern, str(val)):
                            date_matches += 1
                            break
                
                if len(sample_values) > 0 and date_matches > len(sample_values) * 0.7:  # 70% match threshold
                    suggestions[column] = 'datetime'
                    print(f"Suggested datetime for {column}")
                    continue
                
                # Check for booleans
                bool_values = ['true', 'false', '1', '0', 'yes', 'no', 'y', 'n']
                bool_matches = sum(1 for val in sample_values 
                                 if str(val).lower() in bool_values)
                
                if len(sample_values) > 0 and bool_matches == len(sample_values):
                    suggestions[column] = 'boolean'
                    print(f"Suggested boolean for {column}")
                    continue
                
                # Check for categories (if unique values are less than 50% of total)
                unique_count = df[column].nunique()
                if unique_count < len(df) * 0.1 and unique_count < 50:
                    suggestions[column] = 'category'
                    print(f"Suggested category for {column}")
                    continue
                
                # Check for numeric values stored as strings
                numeric_matches = 0
                for val in sample_values:
                    clean_val = str(val).replace('.', '').replace('-', '').replace('+', '')
                    if clean_val.isdigit():
                        numeric_matches += 1
                
                if len(sample_values) > 0 and numeric_matches > len(sample_values) * 0.8:  # 80% numeric
                    if any('.' in str(val) for val in sample_values):
                        suggestions[column] = 'float'
                        print(f"Suggested float for {column}")
                    else:
                        suggestions[column] = 'int'
                        print(f"Suggested int for {column}")
                    continue
            
            # Check for potential date columns that are already numeric (timestamps)
            elif current_type in ['int64', 'float64']:
                try:
                    min_val = df[column].min()
                    max_val = df[column].max()
                    # Check if values look like timestamps
                    if min_val > 1000000000 and max_val < 9999999999:  # Unix timestamp range
                        suggestions[column] = 'datetime'
                        print(f"Suggested datetime for timestamp column {column}")
                except:
                    pass
        
        print(f"Final suggestions: {suggestions}")  # Debug log
        
        return jsonify({
            'success': True,
            'suggestions': suggestions
        })
    
    except Exception as e:
        print(f"Error in suggest_data_types: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Data type suggestion failed: {str(e)}'})

@app.route('/convert-data-types', methods=['POST'])
def convert_data_types():
    """Convert data types for specified columns"""
    try:
        df = DATASTORE.get('df')
        if df is None:
            return jsonify({'error': 'No data loaded'})
        
        data_type_changes = request.get_json().get('data_type_changes', {})
        if not data_type_changes:
            return jsonify({'error': 'No data type changes provided'})
        
        conversion_log = []
        converted_df = df.copy()
        
        for column, new_type in data_type_changes.items():
            if column not in converted_df.columns:
                conversion_log.append(f"Column '{column}' not found - skipped")
                continue
            
            try:
                original_type = str(converted_df[column].dtype)
                
                if new_type == 'datetime':
                    converted_df[column] = pd.to_datetime(converted_df[column], errors='coerce')
                    conversion_log.append(f"'{column}': {original_type} → datetime")
                
                elif new_type == 'category':
                    converted_df[column] = converted_df[column].astype('category')
                    conversion_log.append(f"'{column}': {original_type} → category")
                
                elif new_type == 'int':
                    # Handle NaN values before converting to int
                    if converted_df[column].isnull().any():
                        converted_df[column] = converted_df[column].fillna(0)
                    converted_df[column] = pd.to_numeric(converted_df[column], errors='coerce').astype('Int64')
                    conversion_log.append(f"'{column}': {original_type} → integer")
                
                elif new_type == 'float':
                    converted_df[column] = pd.to_numeric(converted_df[column], errors='coerce')
                    conversion_log.append(f"'{column}': {original_type} → float")
                
                elif new_type == 'string':
                    converted_df[column] = converted_df[column].astype('string')
                    conversion_log.append(f"'{column}': {original_type} → string")
                
                elif new_type == 'boolean':
                    # Convert to boolean with common mappings
                    bool_map = {'true': True, 'false': False, '1': True, '0': False, 
                               'yes': True, 'no': False, 'y': True, 'n': False}
                    converted_df[column] = converted_df[column].astype(str).str.lower().map(bool_map)
                    converted_df[column] = converted_df[column].astype('boolean')
                    conversion_log.append(f"'{column}': {original_type} → boolean")
                
            except Exception as e:
                conversion_log.append(f"Failed to convert '{column}' to {new_type}: {str(e)}")
        
        # Update the datastore
        DATASTORE['df'] = converted_df
        
        return jsonify({
            'success': True,
            'conversion_log': conversion_log
        })
    
    except Exception as e:
        print(f"Error in convert_data_types: {e}")
        return jsonify({'error': f'Data type conversion failed: {str(e)}'})

@app.route('/download-clean-data')
def download_clean_data():
    """Download the cleaned dataset as CSV"""
    df = DATASTORE.get('df')
    if df is None:
        return jsonify({'error': 'No cleaned data available'})
    
    try:
        # Create a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cleaned_data_{timestamp}.csv"
        
        # Create a temporary file in memory
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        # Convert to bytes for download
        mem = io.BytesIO()
        mem.write(output.getvalue().encode('utf-8'))
        mem.seek(0)
        
        return send_file(
            mem,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        return jsonify({'error': f'Failed to download data: {str(e)}'})

@app.route('/download-original-data')
def download_original_data():
    """Download the original dataset as CSV"""
    df = DATASTORE.get('original_df')
    if df is None:
        return jsonify({'error': 'No original data available'})
    
    try:
        # Create a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"original_data_{timestamp}.csv"
        
        # Create a temporary file in memory
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        # Convert to bytes for download
        mem = io.BytesIO()
        mem.write(output.getvalue().encode('utf-8'))
        mem.seek(0)
        
        return send_file(
            mem,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        return jsonify({'error': f'Failed to download data: {str(e)}'})

@app.route('/download-data-summary')
def download_data_summary():
    """Download a summary report of data cleaning operations"""
    df = DATASTORE.get('df')
    original_df = DATASTORE.get('original_df')
    cleaning_log = DATASTORE.get('cleaning_log', [])
    
    if df is None:
        return jsonify({'error': 'No data available'})
    
    try:
        # Create summary report
        summary_lines = [
            "DATA CLEANING SUMMARY REPORT",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "DATASET OVERVIEW:",
            f"- Current dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns"
        ]
        
        if original_df is not None:
            summary_lines.extend([
                f"- Original dataset shape: {original_df.shape[0]:,} rows × {original_df.shape[1]} columns",
                f"- Rows removed: {original_df.shape[0] - df.shape[0]:,}",
                f"- Columns removed: {original_df.shape[1] - df.shape[1]}"
            ])
        
        summary_lines.extend([
            "",
            "COLUMN INFORMATION:",
            f"- Total columns: {len(df.columns)}",
            f"- Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}",
            f"- Text columns: {len(df.select_dtypes(include=['object']).columns)}",
            f"- Date columns: {len(df.select_dtypes(include=['datetime']).columns)}",
            ""
        ])
        
        # Add data quality info
        try:
            quality_report = analyze_data_quality(df)
            summary_lines.extend([
                "DATA QUALITY METRICS:",
                f"- Quality Score: {quality_report['quality_score']}% ({quality_report['quality_status']})",
                f"- Duplicate rows: {quality_report['duplicates']:,}",
                f"- Columns with missing values: {sum(1 for col, info in quality_report['missing_values'].items() if info['count'] > 0)}",
                ""
            ])
        except:
            summary_lines.extend([
                "DATA QUALITY METRICS:",
                "- Unable to calculate quality metrics",
                ""
            ])
        
        # Add cleaning operations log
        if cleaning_log:
            summary_lines.extend([
                "CLEANING OPERATIONS PERFORMED:",
                *[f"- {log}" for log in cleaning_log],
                ""
            ])
        
        # Add column mappings
        column_mapping = DATASTORE.get('column_mapping', {})
        if column_mapping:
            summary_lines.extend([
                "COLUMN NAME CHANGES:",
                *[f"- '{old}' → '{new}'" for old, new in column_mapping.items()],
                ""
            ])
        
        # Add column details
        summary_lines.extend(["COLUMN DETAILS:"])
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            summary_lines.append(f"- {col}: {dtype}, {missing_count} missing, {unique_count} unique values")
        
        # Create the summary text
        summary_text = "\n".join(summary_lines)
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data_cleaning_summary_{timestamp}.txt"
        
        # Create file in memory
        mem = io.BytesIO()
        mem.write(summary_text.encode('utf-8'))
        mem.seek(0)
        
        return send_file(
            mem,
            mimetype='text/plain',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        return jsonify({'error': f'Failed to generate summary: {str(e)}'})

@app.route('/debug-charts')
def debug_charts():
    """Debug route to test chart generation"""
    df = DATASTORE.get('df')
    if df is None:
        return jsonify({'error': 'No data loaded'})
    
    # Test chart specs
    test_specs = [
        {
            'chart_type': 'bar chart',
            'x': 'branch',
            'y': 'amount',
            'agg': 'sum'
        },
        {
            'chart_type': 'pie chart',
            'x': 'category',
            'y': 'quantity',
            'agg': 'sum'
        }
    ]
    
    results = []
    for i, spec in enumerate(test_specs):
        try:
            validated_spec = validate_chart_spec_in_app(spec, df)
            
            if validated_spec:
                url = generate_chart_from_ai(validated_spec, df)
                results.append({
                    'spec': spec,
                    'validated_spec': validated_spec,
                    'url': url,
                    'success': url is not None
                })
            else:
                results.append({
                    'spec': spec,
                    'validated_spec': None,
                    'url': None,
                    'success': False,
                    'error': 'Validation failed'
                })
        except Exception as e:
            results.append({
                'spec': spec,
                'validated_spec': None,
                'url': None,
                'success': False,
                'error': str(e)
            })
    
    return jsonify({
        'data_info': {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist()
        },
        'column_mappings': DATASTORE.get('column_mapping', {}),
        'test_results': results
    })

@app.route('/reset-data')
def reset_data():
    """Reset to original data"""
    if 'original_df' in DATASTORE:
        # Reset to original and re-clean column names
        original_df = DATASTORE['original_df'].copy()
        df, column_mapping = clean_column_names_enhanced(original_df)
        DATASTORE['df'] = df
        return jsonify({'success': True, 'message': 'Data reset to original state'})
    return jsonify({'error': 'No original data found'})

##############################################
# MAIN APPLICATION ENTRY POINT              #
##############################################

if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'], host='0.0.0.0', port=port)