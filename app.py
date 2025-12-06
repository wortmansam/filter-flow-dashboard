import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np

# --- Configuration ---
DATA_FILE = 'your_data.csv' 
FILTER_PAIRS_FILE = 'filter_pairs.csv' # MUST be created with headers: Store, Product, Pump, Filter, Batch_Rule

# Define column names for clarity
COL_STORE = 'Store'
COL_FUEL_GRADE = 'Fuel Grade'
COL_FUEL_POSIT = 'Fuel Position'
COL_QUANTITY = 'Quantity'
COL_START_DATE = 'Disp Start Date'
COL_START_TIME = 'Disp Start Time'
COL_END_DATE = 'Disp End Date'
COL_END_TIME = 'Disp End Time'
COL_FILTER_ID = 'Filter' # This comes from the 'Filter' column in the lookup CSV

# --- Fixed Visual Range and Height for Flow Rate Graphs ---
FIXED_FLOW_MIN = 5.0
FIXED_FLOW_MAX = 10.0
CHART_HEIGHT = 700 # Increased height in pixels for better separation

# Function to format military time string (e.g., 227 or 231924) to HH:MM:SS
def format_military_time(time_series):
    # Convert to string, ensure 6 digits with leading zeros (e.g., 227 -> '000227')
    time_str = time_series.astype(str).str.zfill(6)
    # Format as HH:MM:SS
    return time_str.str.slice(0, 2) + ':' + time_str.str.slice(2, 4) + ':' + time_str.str.slice(4, 6)


@st.cache_data
def load_data(main_file_path, filter_pairs_file_path):
    st.info("Loading and processing large dataset... This may take a moment on the first run.")
    
    # --- 1. Load Main Data ---
    df = pd.read_csv(main_file_path)

   # --- 2. Load and Apply Lookup Logic (SUMIFS Equivalent) ---
    try:
        filter_pairs = pd.read_csv(filter_pairs_file_path)
        
        # Rename lookup headers to match main data headers for the merge key
        filter_pairs = filter_pairs.rename(columns={
            'Product': COL_FUEL_GRADE,
            'Pump': COL_FUEL_POSIT
        })
        
        # Perform the MERGE (VLOOKUP/SUMIFS equivalent) on all three key columns
        # Include the 'Batch_Rule' column in the merge
        df = pd.merge(
            df,
            filter_pairs[[COL_STORE, COL_FUEL_GRADE, COL_FUEL_POSIT, COL_FILTER_ID, 'Batch_Rule']],
            on=[COL_STORE, COL_FUEL_GRADE, COL_FUEL_POSIT],
            how='left' 
        )
        
        df[COL_FILTER_ID] = df[COL_FILTER_ID].fillna('Unknown Filter')
        df['Batch_Rule'] = df['Batch_Rule'].fillna('200-10') # Default to 200-10 if rule is missing

    except FileNotFoundError:
        st.error(f"Lookup file '{filter_pairs_file_path}' not found! Cannot assign Filter IDs.")
        df[COL_FILTER_ID] = 'Lookup Missing'
        st.stop()
    except KeyError as e:
        # This will catch the missing 'Batch_Rule' column if the user hasn't updated their CSV
        st.error(f"Error in lookup headers. Missing column: {e}. Check 'filter_pairs.csv' headers!")
        df[COL_FILTER_ID] = 'Lookup Error'
        st.stop()


    # --- 3. Calculated Columns (Date, Flow Rate, Batches, and Formatting) ---
    
    # Date/Time combination for accurate chronological sorting
    df['Start_Time_Combined'] = pd.to_datetime(
        df[COL_START_DATE].astype(str) + df[COL_START_TIME].astype(str).str.zfill(6),
        format='%Y%m%d%H%M%S', errors='coerce'
    ).sort_values() 

    df['End_Time_Combined'] = pd.to_datetime(
        df[COL_END_DATE].astype(str) + df[COL_END_TIME].astype(str).str.zfill(6),
        format='%Y%m%d%H%M%S', errors='coerce'
    )

    # Formatting columns for display
    df['Start Date Formatted'] = pd.to_datetime(df[COL_START_DATE], format='%Y%m%d', errors='coerce').dt.strftime('%m/%d/%Y')
    df['End Date Formatted'] = pd.to_datetime(df[COL_END_DATE], format='%Y%m%d', errors='coerce').dt.strftime('%m/%d/%Y')
    df['Start Time Formatted'] = format_military_time(df[COL_START_TIME])
    df['End Time Formatted'] = format_military_time(df[COL_END_TIME])
    
    # Flow Rate calculation
    df['Duration_Sec'] = (df['End_Time_Combined'] - df['Start_Time_Combined']).dt.total_seconds()
    
    df['Flow Rate'] = np.where(
        (df['Duration_Sec'] > 0),
        df[COL_QUANTITY] / (df['Duration_Sec'] / 60),
        0
    )

    # Batches (now grouped by Store AND the correct Filter ID)
    df = df.sort_values(by='Start_Time_Combined').reset_index(drop=True)
    
    # CRITICAL BATCH LOGIC: Grouping by Store and the newly looked-up Filter ID
    GROUP_COLS_BATCH = [COL_STORE, COL_FILTER_ID]
    
    df['Group_Count'] = df.groupby(GROUP_COLS_BATCH).cumcount() + 1
    
    df['Batch 20'] = np.ceil(df['Group_Count'] / 20).astype(int)
    df['Batch 200'] = np.ceil(df['Group_Count'] / 200).astype(int)
    
    
    # --- 4. Final Cleaning and Return ---
    df = df.dropna(subset=[COL_QUANTITY, 'Flow Rate', 'Start_Time_Combined'])
    st.success(f"Data Loaded, Lookups Performed, and Formulas Applied: {len(df):,} records remaining.")
    
    return df

# --- Load Data and Set App Title ---
try:
    df = load_data(DATA_FILE, FILTER_PAIRS_FILE)
except Exception as e:
    st.error(f"FATAL ERROR during data loading or processing: {e}")
    st.stop()


st.title("⛽ Filter Flow Rate Monitoring System")
st.markdown(f"### Analyzing {len(df):,} Fuel Dispensing Events")

# --------------------------------------------------------------------------
# --- UI Filters (Sidebar) ---
# --------------------------------------------------------------------------
st.sidebar.header("Filter Options")

# 1. Primary Filter: Select Store (Already sorted)
selected_store = st.sidebar.selectbox(
    '1. Select Store to Monitor:',
    options=['All'] + sorted(df[COL_STORE].unique().astype(str).tolist())
)

# --- DYNAMIC FILTERING LOGIC ---
if selected_store != 'All':
    filtered_options_df = df[df[COL_STORE].astype(str) == selected_store]
else:
    filtered_options_df = df.copy()
    
# 2. Secondary Filter: Select the single Filter ID (FIXED: Numerical Sorting)
filter_options = filtered_options_df[COL_FILTER_ID].unique().tolist()
# Sort numerically by converting to float, putting non-numeric last ('Unknown Filter')
sorted_options = sorted(filter_options, key=lambda x: pd.to_numeric(x, errors='coerce') if x is not None else float('inf'))

selected_filter_id = st.sidebar.selectbox(
    '2. Select Filter ID (1-12):',
    options=['All'] + [str(x) for x in sorted_options] # Ensure they are strings for display
)

# --- Apply Filters ---
filtered_df = df.copy()

if selected_store != 'All':
    filtered_df = filtered_df[filtered_df[COL_STORE].astype(str) == selected_store]

# Filter by the single, looked-up Filter ID
if selected_filter_id != 'All':
    filtered_df = filtered_df[filtered_df[COL_FILTER_ID].astype(str) == selected_filter_id]

# --------------------------------------------------------------------------
# --- Core Analysis Functions ---
# --------------------------------------------------------------------------

# Define the function to find the Nth largest value
def get_nth_largest(group, n):
    """Returns the Nth largest Flow Rate in a given group."""
    if len(group) < n:
        # Return max flow rate if the group is too small
        return group['Flow Rate'].max() if not group.empty else 0
    return group['Flow Rate'].nlargest(n).iloc[-1]

# Function to remove the last batch from an analysis result
def discard_last_batch(group, batch_col_name):
    if group.empty:
        return group
    max_batch = group[batch_col_name].max()
    return group[group[batch_col_name] != max_batch]


# --------------------------------------------------------------------------
# --- Multi-Filter Comparison Graph (FINAL, UNIFIED X-AXIS REVISION) ---
# --------------------------------------------------------------------------

if selected_store != 'All':
    st.header(f"Multi-Filter Comparison: Store {selected_store}")

    comparison_data = df[df[COL_STORE].astype(str) == selected_store].copy()
    
    all_filters = comparison_data[COL_FILTER_ID].unique()
    comparison_results = []
    
    # Iterate through all unique filters to apply their specific N and Batch Size
    for filter_id in all_filters:
        # 1. Filter data for this specific filter ID and get its rule
        filter_data = comparison_data[comparison_data[COL_FILTER_ID] == filter_id].copy()
        
        if filter_data.empty:
            continue
            
        rule_str = filter_data['Batch_Rule'].iloc[0]
        
        # 2. Parse the rule to get the dynamic batch size and Nth value
        try:
            batch_size_val, NTH_VALUE = map(int, rule_str.split('-'))
        except ValueError:
            batch_size_val = 200
            NTH_VALUE = 10
            
        BATCH_SIZE_COL_CURRENT = f'Batch {batch_size_val}'

        # 3. Calculate Nth largest flow rate using the filter's specific N and BATCH SIZE
        filter_analysis_df = filter_data.groupby(BATCH_SIZE_COL_CURRENT).apply(
            lambda x: get_nth_largest(x, NTH_VALUE)
        ).reset_index(name='Nth_Largest_Flow_Rate_Comp')
        
        # 4. Discard the last batch (UNIVERSAL RULE)
        filter_analysis_df = discard_last_batch(filter_analysis_df, BATCH_SIZE_COL_CURRENT)
        
        # 5. Skip if data disappeared after discarding the last batch
        if filter_analysis_df.empty:
            continue
            
        # 6. CRITICAL STEP: Create the unified, sequential X-axis for plotting
        filter_analysis_df = filter_analysis_df.reset_index(drop=True)
        filter_analysis_df['Batch_Index'] = filter_analysis_df.index + 1
        
        filter_analysis_df[COL_FILTER_ID] = filter_id
        filter_analysis_df['N_Value'] = NTH_VALUE
        filter_analysis_df['Batch_Rule_Text'] = f'Batch {batch_size_val}, {NTH_VALUE}th Max'
        
        comparison_results.append(filter_analysis_df)

    if comparison_results:
        comparison_analysis_df = pd.concat(comparison_results)
        
        # NEW FIX: Add numeric sorting column and sort the DataFrame to fix legend order
        comparison_analysis_df['Filter_Numeric_ID'] = pd.to_numeric(comparison_analysis_df[COL_FILTER_ID], errors='coerce').fillna(9999).astype(int)
        comparison_analysis_df = comparison_analysis_df.sort_values(by='Filter_Numeric_ID', kind='stable')
        
        # Get the range of N values used for the title
        n_min = comparison_analysis_df['N_Value'].min()
        n_max = comparison_analysis_df['N_Value'].max()
        n_text = f"{n_min}" if n_min == n_max else f"{n_min} to {n_max}"

        fig_comparison = px.line(
            comparison_analysis_df,
            x='Batch_Index', # Use the unified index for the X-axis
            y='Nth_Largest_Flow_Rate_Comp',
            color=COL_FILTER_ID, 
            title=f'Comparison of All Filters by Sequential Batch Index ({n_text}th Largest)',
            labels={'Nth_Largest_Flow_Rate_Comp': 'Flow Rate (Units/Min)', 'Batch_Index': 'Sequential Batch Index'},
            hover_data={
                'Batch_Rule_Text': True, 
                COL_FILTER_ID: True,
                'Nth_Largest_Flow_Rate_Comp': ':.2f'
            },
            template="plotly_white",
            markers=True,
            height=CHART_HEIGHT
        )

        # Apply the fixed Y-axis range
        fig_comparison.update_yaxes(
            range=[FIXED_FLOW_MIN, FIXED_FLOW_MAX], 
            tick0=FIXED_FLOW_MIN, 
            dtick=0.5
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        st.markdown("---")
    else:
        st.warning(f"No sufficient data for multi-filter comparison at Store {selected_store} after applying dynamic rules and discarding the last batch.")


# --------------------------------------------------------------------------
# --- Single-Filter Detailed Analysis ---
# --------------------------------------------------------------------------

st.header("Batch Flow Rate Analysis")

if filtered_df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()
    
if selected_filter_id == 'All':
    st.warning("Please select a specific Filter ID for the detailed trend analysis.")
    st.stop()
    
# Get the unique Batch_Rule for the selected Filter ID
rule_str = filtered_df['Batch_Rule'].iloc[0] # Get the rule from the first row of the filtered data

# Parse the rule string (e.g., '200-10' or '20-3')
try:
    batch_size_val, NTH_VALUE = map(int, rule_str.split('-'))
except ValueError:
    st.error(f"Invalid Batch Rule found: '{rule_str}'. Please check your filter_pairs.csv.")
    st.stop()

# Set the dynamic batch column name
BATCH_SIZE_COL = f'Batch {batch_size_val}'


# Group by the dynamically determined batch column and calculate the Nth largest flow rate
# Pass the dynamically determined NTH_VALUE to the function
batch_analysis_df = filtered_df.groupby(BATCH_SIZE_COL).apply(
    lambda x: get_nth_largest(x, NTH_VALUE)
).reset_index(name='Nth_Largest_Flow_Rate')

# --- IMPLEMENT: Discard the last batch ---
batch_analysis_df = discard_last_batch(batch_analysis_df, BATCH_SIZE_COL)

if batch_analysis_df.empty:
    st.warning(f"Insufficient data remaining for Filter ID {selected_filter_id} after discarding the last batch.")
    st.stop()


# --- Visualization and Metrics ---

st.subheader(f"{NTH_VALUE}th Largest Flow Rate Trend by {BATCH_SIZE_COL}")

# Create the initial line chart
fig_batch_trend = px.line(
    batch_analysis_df,
    x=BATCH_SIZE_COL,
    y='Nth_Largest_Flow_Rate',
    title=f'Flow Rate Trend (Store: {selected_store}, Filter: {selected_filter_id})',
    labels={'Nth_Largest_Flow_Rate': f'{NTH_VALUE}th Largest Flow Rate (Units/Min)'},
    template="plotly_white",
    markers=True,
    height=CHART_HEIGHT # Set new chart height
)

# Set Fixed Y-Axis Range
fig_batch_trend.update_yaxes(
    range=[FIXED_FLOW_MIN, FIXED_FLOW_MAX], 
    tick0=FIXED_FLOW_MIN, 
    dtick=0.5 # Using 0.5 for better detail in the smaller range
)

st.plotly_chart(fig_batch_trend, use_container_width=True)

st.markdown("---")

# --- Key Performance Indicators (KPIs) ---
st.header("Summary Metrics")

avg_flow_rate = filtered_df['Flow Rate'].mean()
num_events = len(filtered_df)
avg_batch_flow = batch_analysis_df['Nth_Largest_Flow_Rate'].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Total Events Analyzed", f"{num_events:,}")
col2.metric("Average Flow Rate (Overall)", f"{avg_flow_rate:,.2f} /Min")
col3.metric(f"Avg {NTH_VALUE}th Flow Rate ({BATCH_SIZE_COL})", f"{avg_batch_flow:,.2f} /Min")


st.markdown("---")
# --- Raw Transaction Drill-Down ---
st.header("Raw Transaction Drill-Down")

# Get list of batches available in the current filtered analysis (using the dynamic column name)
available_batches = sorted(batch_analysis_df[BATCH_SIZE_COL].unique().tolist()) # Already sorted numerically

if available_batches:
    selected_batch_num = st.selectbox(
        f'Select a {BATCH_SIZE_COL} for Raw Transaction Detail:',
        options=available_batches
    )
    
    # Filter the raw data (filtered_df) using the selected batch number and the dynamic column
    # Sorting by 'Start_Time_Combined' ensures strict chronological order
    raw_batch_data = filtered_df[filtered_df[BATCH_SIZE_COL] == selected_batch_num].sort_values(
        by='Start_Time_Combined', 
        ascending=True
    )
    
    # Display the raw data with formatted columns
    st.dataframe(raw_batch_data[[
        'Start Date Formatted', 'Start Time Formatted', 'End Time Formatted', 
        COL_FUEL_GRADE, COL_FUEL_POSIT, COL_QUANTITY, 'Duration_Sec', 'Flow Rate', BATCH_SIZE_COL
    ]])
    
    # Display metrics for this specific batch
    batch_flow_rate = batch_analysis_df[
        batch_analysis_df[BATCH_SIZE_COL] == selected_batch_num
    ]['Nth_Largest_Flow_Rate'].iloc[0]
    
    st.info(
        f"Selected Batch {selected_batch_num}: "
        f"**{len(raw_batch_data)}** transactions. "
        f"**{NTH_VALUE}th Largest Flow Rate** was **{batch_flow_rate:,.2f} /Min**."
    )
else:
    st.warning(f"No complete batches available for drill-down after discarding the last batch.")

st.markdown("---")
st.subheader("Batch Data Sample (Aggregated)")
st.dataframe(batch_analysis_df)