import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from datetime import date

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

# --- Product Categorization ---
def define_product_category(df):
    """Categorizes Fuel Grades 1000, 1002, and 1004 as 'Gasoline', others as 'Diesel'."""
    # Grouping based on user request: 1000, 1002, 1004 (Gas) vs. 1003 (Diesel)
    gasoline_grades = ['1000', '1002', '1004'] 
    
    # Convert fuel grade to string for comparison
    grade_str = df[COL_FUEL_GRADE].astype(str)
    
    # Create the new Product_Category column
    df['Product_Category'] = grade_str.apply(
        lambda x: 'Gasoline' if x in gasoline_grades else 'Diesel'
    )
    return df


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

    # --- NEW Date Components for Historical Analysis (YOY) ---
    df['Calendar_Date'] = pd.to_datetime(df[COL_START_DATE], format='%Y%m%d', errors='coerce')
    df['Year'] = df['Calendar_Date'].dt.year
    df['WeekOfYear'] = df['Calendar_Date'].dt.isocalendar().week.astype(int) # ISO week number (better for consistency)
    df['DayOfWeek'] = df['Calendar_Date'].dt.day_name()
    df['DayOfYear'] = df['Calendar_Date'].dt.dayofyear # Used for YOY alignment
    
    # Formatting columns for display
    df['Start Date Formatted'] = df['Calendar_Date'].dt.strftime('%m/%d/%Y')
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
    df = df.dropna(subset=[COL_QUANTITY, 'Flow Rate', 'Start_Time_Combined', 'Calendar_Date'])
    st.success(f"Data Loaded, Lookups Performed, and Formulas Applied: {len(df):,} records remaining.")
    
    return df

# --- Core Analysis Functions ---

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


# --- Load Data and Set App Title ---
try:
    df = load_data(DATA_FILE, FILTER_PAIRS_FILE)
except Exception as e:
    st.error(f"FATAL ERROR during data loading or processing: {e}")
    st.stop()

# --- Global Calculations ---
# 1. Apply Product Categorization GLOBALLY
df = define_product_category(df)

# 2. Determine the latest possible transaction time in the file
MAX_GLOBAL_TIME = df['Start_Time_Combined'].max()


st.title("⛽ Filter & Uptime Monitoring Dashboard")
st.markdown(f"### Analyzing {len(df):,} Fuel Dispensing Events")

# --- Initial date range calculation for filter default ---
# Ensure min/max dates are simple date objects for st.date_input
min_date = df['Calendar_Date'].min().date()
max_date = df['Calendar_Date'].max().date()


# --------------------------------------------------------------------------
# --- UI Filters (Sidebar) ---
# --------------------------------------------------------------------------
st.sidebar.header("Filter Options")

# 1. Primary Filter: Select Store
selected_store = st.sidebar.selectbox(
    '1. Select Store to Monitor:',
    options=['All'] + sorted(df[COL_STORE].unique().astype(str).tolist())
)

# --- DYNAMIC FILTERING LOGIC for Filter Status Tab ---
if selected_store != 'All':
    filtered_options_df = df[df[COL_STORE].astype(str) == selected_store]
else:
    filtered_options_df = df.copy()
    
# 2. Secondary Filter: Select the single Filter ID
filter_options = filtered_options_df[COL_FILTER_ID].unique().tolist()
sorted_options = sorted(filter_options, key=lambda x: pd.to_numeric(x, errors='coerce') if x is not None else float('inf'))

# NOTE: This filter only applies to the Filter Status tab (Tab 1)
selected_filter_id = st.sidebar.selectbox(
    '2. Select Filter ID (1-12):',
    options=['All'] + [str(x) for x in sorted_options] # Ensure they are strings for display
)

# --------------------------------------------------------------------------
# --- CRITICAL FILTERING FIX: Create two different dataframes for two tabs ---
# --------------------------------------------------------------------------

# 1. Filter by Store ONLY (Used for Uptime Status - Tab 2, and Historical Sales - Tab 3)
store_filtered_df = df.copy()
if selected_store != 'All':
    store_filtered_df = store_filtered_df[store_filtered_df[COL_STORE].astype(str) == selected_store]

# 2. Filter by Store AND Filter ID (Used for Filter Status - Tab 1)
filter_id_filtered_df = store_filtered_df.copy()
if selected_filter_id != 'All':
    filter_id_filtered_df = filter_id_filtered_df[filter_id_filtered_df[COL_FILTER_ID].astype(str) == selected_filter_id]


# --------------------------------------------------------------------------
# --- TAB LAYOUT ---
# --------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Filter Status", "📈 Uptime Status", "💰 Historical Fuel Sales"])

# --------------------------------------------------------------------------
# --- TAB 1: FILTER STATUS (Uses filter_id_filtered_df) ---
# --------------------------------------------------------------------------
with tab1:
    
    # --------------------------------------------------------------------------
    # --- Multi-Filter Comparison Graph ---
    # --------------------------------------------------------------------------

    if selected_store != 'All':
        st.header(f"Multi-Filter Comparison: Store {selected_store}")

        comparison_data = df[df[COL_STORE].astype(str) == selected_store].copy()
        
        all_filters = comparison_data[COL_FILTER_ID].unique()
        comparison_results = []
        
        for filter_id in all_filters:
            filter_data = comparison_data[comparison_data[COL_FILTER_ID] == filter_id].copy()
            
            if filter_data.empty:
                continue
                
            rule_str = filter_data['Batch_Rule'].iloc[0]
            
            try:
                batch_size_val, NTH_VALUE = map(int, rule_str.split('-'))
            except ValueError:
                batch_size_val = 200
                NTH_VALUE = 10
                
            BATCH_SIZE_COL_CURRENT = f'Batch {batch_size_val}'

            filter_analysis_df = filter_data.groupby(BATCH_SIZE_COL_CURRENT).apply(
                lambda x: get_nth_largest(x, NTH_VALUE)
            ).reset_index(name='Nth_Largest_Flow_Rate_Comp')
            
            filter_analysis_df = discard_last_batch(filter_analysis_df, BATCH_SIZE_COL_CURRENT)
            
            if filter_analysis_df.empty:
                continue
                
            filter_analysis_df = filter_analysis_df.reset_index(drop=True)
            filter_analysis_df['Batch_Index'] = filter_analysis_df.index + 1
            
            filter_analysis_df[COL_FILTER_ID] = filter_id
            filter_analysis_df['N_Value'] = NTH_VALUE
            filter_analysis_df['Batch_Rule_Text'] = f'Batch {batch_size_val}, {NTH_VALUE}th Max'
            
            comparison_results.append(filter_analysis_df)

        if comparison_results:
            comparison_analysis_df = pd.concat(comparison_results)
            
            # FIX: Add numeric sorting column and sort the DataFrame to fix legend order
            comparison_analysis_df['Filter_Numeric_ID'] = pd.to_numeric(comparison_analysis_df[COL_FILTER_ID], errors='coerce').fillna(9999).astype(int)
            comparison_analysis_df = comparison_analysis_df.sort_values(by='Filter_Numeric_ID', kind='stable')
            
            n_min = comparison_analysis_df['N_Value'].min()
            n_max = comparison_analysis_df['N_Value'].max()
            n_text = f"{n_min}" if n_min == n_max else f"{n_min} to {n_max}"

            fig_comparison = px.line(
                comparison_analysis_df,
                x='Batch_Index', 
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
    # --- Single-Filter Detailed Analysis (Uses filter_id_filtered_df) ---
    # --------------------------------------------------------------------------

    if selected_filter_id != 'All':
        st.header("Batch Flow Rate Analysis")
        
        # NOTE: Using filter_id_filtered_df here
        if filter_id_filtered_df.empty:
            st.warning("No data matches the selected filters.")
        else:
            # Get the unique Batch_Rule for the selected Filter ID
            rule_str = filter_id_filtered_df['Batch_Rule'].iloc[0]

            # Parse the rule string (e.g., '200-10' or '20-3')
            try:
                batch_size_val, NTH_VALUE = map(int, rule_str.split('-'))
            except ValueError:
                st.error(f"Invalid Batch Rule found: '{rule_str}'. Please check your filter_pairs.csv.")
                st.stop()

            BATCH_SIZE_COL = f'Batch {batch_size_val}'

            batch_analysis_df = filter_id_filtered_df.groupby(BATCH_SIZE_COL).apply(
                lambda x: get_nth_largest(x, NTH_VALUE)
            ).reset_index(name='Nth_Largest_Flow_Rate')

            batch_analysis_df = discard_last_batch(batch_analysis_df, BATCH_SIZE_COL)

            if batch_analysis_df.empty:
                st.warning(f"Insufficient data remaining for Filter ID {selected_filter_id} after discarding the last batch.")
            else:
                # --- Visualization and Metrics ---
                st.subheader(f"{NTH_VALUE}th Largest Flow Rate Trend by {BATCH_SIZE_COL}")

                fig_batch_trend = px.line(
                    batch_analysis_df,
                    x=BATCH_SIZE_COL,
                    y='Nth_Largest_Flow_Rate',
                    title=f'Flow Rate Trend (Store: {selected_store}, Filter: {selected_filter_id})',
                    labels={'Nth_Largest_Flow_Rate': f'{NTH_VALUE}th Largest Flow Rate (Units/Min)'},
                    template="plotly_white",
                    markers=True,
                    height=CHART_HEIGHT
                )

                fig_batch_trend.update_yaxes(
                    range=[FIXED_FLOW_MIN, FIXED_FLOW_MAX], 
                    tick0=FIXED_FLOW_MIN, 
                    dtick=0.5
                )

                st.plotly_chart(fig_batch_trend, use_container_width=True)
                st.markdown("---")

                # --- Key Performance Indicators (KPIs) ---
                st.header("Summary Metrics")

                avg_flow_rate = filter_id_filtered_df['Flow Rate'].mean()
                num_events = len(filter_id_filtered_df)
                avg_batch_flow = batch_analysis_df['Nth_Largest_Flow_Rate'].mean()

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Events Analyzed", f"{num_events:,}")
                col2.metric("Average Flow Rate (Overall)", f"{avg_flow_rate:,.2f} /Min")
                col3.metric(f"Avg {NTH_VALUE}th Flow Rate ({BATCH_SIZE_COL})", f"{avg_batch_flow:,.2f} /Min")


                st.markdown("---")
                # --- Raw Transaction Drill-Down ---
                st.header("Raw Transaction Drill-Down")

                available_batches = sorted(batch_analysis_df[BATCH_SIZE_COL].unique().tolist())

                if available_batches:
                    selected_batch_num = st.selectbox(
                        f'Select a {BATCH_SIZE_COL} for Raw Transaction Detail:',
                        options=available_batches
                    )

                    raw_batch_data = filter_id_filtered_df[filter_id_filtered_df[BATCH_SIZE_COL] == selected_batch_num].sort_values(
                        by='Start_Time_Combined', 
                        ascending=True
                    )

                    st.dataframe(raw_batch_data[[
                        'Start Date Formatted', 'Start Time Formatted', 'End Time Formatted', 
                        COL_FUEL_GRADE, COL_FUEL_POSIT, COL_QUANTITY, 'Duration_Sec', 'Flow Rate', BATCH_SIZE_COL
                    ]])

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

# --------------------------------------------------------------------------
# --- TAB 2: UPTIME STATUS (Uses store_filtered_df) ---
# --------------------------------------------------------------------------
with tab2:
    st.header("Pump Uptime Status: Gallons Dispensed Per Day")
    
    if store_filtered_df.empty:
        st.warning("No data matches the selected Store for Uptime Analysis.")
    else:
        # --- 1. Product Category Selection (Radio Button) ---
        selected_category = st.radio(
            "Select Product Category:",
            ('Gasoline', 'Diesel'),
            horizontal=True
        )
        
        category_filtered_df = store_filtered_df[store_filtered_df['Product_Category'] == selected_category].copy()
        
        if category_filtered_df.empty:
            st.warning(f"No {selected_category} data found for the selected Store.")
            # Do not stop, but prevent further calculations
        else:
            # --------------------------------------------------------------------------
            # --- CURRENT DOWNTIME IDENTIFIER (NEW SECTION) ---
            # --------------------------------------------------------------------------
            st.markdown("---")
            st.header("🚨 Current Downtime Identifier (By Fuel Type)")
            st.markdown(f"**Latest Transaction in Data:** **{MAX_GLOBAL_TIME.strftime('%Y-%m-%d %H:%M:%S')}**")

            # CRITICAL FIX: Group by Store, Pump Position, AND Product Category
            # This ensures that downtime for 'Gasoline' is calculated independently of 'Diesel' on the same pump.
            downtime_df = category_filtered_df.groupby([COL_STORE, COL_FUEL_POSIT, 'Product_Category']).agg(
                Last_Used_Time=('Start_Time_Combined', 'max')
            ).reset_index()

            # Create the combined key column for display
            downtime_df['Store_Pump_Key'] = (
                downtime_df[COL_STORE].astype(str) + ' - Pump ' + 
                downtime_df[COL_FUEL_POSIT].astype(str) + ' - ' + 
                downtime_df['Product_Category']
            )

            # Calculate Downtime Duration (Time since last use)
            downtime_df['Downtime_Delta'] = MAX_GLOBAL_TIME - downtime_df['Last_Used_Time']
            
            # Convert timedelta to total hours for sorting and display
            downtime_df['Downtime_Hours'] = downtime_df['Downtime_Delta'].dt.total_seconds() / 3600

            # Sort by Downtime_Hours (highest is the longest downtime)
            downtime_df = downtime_df.sort_values(by='Downtime_Hours', ascending=False)
            
            # Format the output for readability
            downtime_df['Last_Used_Formatted'] = downtime_df['Last_Used_Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            # Extract only the days/hours/minutes from the timedelta string
            downtime_df['Downtime_Duration'] = downtime_df['Downtime_Delta'].apply(lambda x: str(x).split('.')[0])
            
            # Select columns for display, using the new combined key
            display_df = downtime_df[[
                'Store_Pump_Key', 
                'Last_Used_Formatted', 
                'Downtime_Duration',
                'Downtime_Hours'
            ]].rename(columns={
                'Store_Pump_Key': 'Store + Pump + Fuel Type',
                'Last_Used_Formatted': 'Last Transaction Time',
                'Downtime_Duration': 'Downtime Since Last Transaction',
                'Downtime_Hours': 'Downtime (Hours)'
            })
            
            st.subheader(f"Pumps Inactive Since Last Global Transaction ({selected_category})")
            
            # Add a threshold slider filter
            max_inactive_hours = display_df['Downtime (Hours)'].max()
            
            # Determine initial slider value (e.g., 24 hours, or the max if it's less than 24)
            initial_value = min(max_inactive_hours, 24.0) if max_inactive_hours > 0 else 0
            
            threshold_hours = st.slider(
                'Show pumps inactive for more than (Hours):', 
                min_value=0.0, 
                max_value=max(168.0, max_inactive_hours + 10), # Max up to 1 week (168h) or 10 hours above observed max
                value=initial_value,
                step=0.5,
                format='%.1f hours'
            )
            
            filtered_display_df = display_df[display_df['Downtime (Hours)'] >= threshold_hours]
            
            if filtered_display_df.empty:
                st.success(f"All **{selected_category}** fuel types at the pumps were used within the last **{threshold_hours}** hours. Uptime is good!")
            else:
                st.dataframe(filtered_display_df, hide_index=True)
                
                # Highlight the most concerning result
                longest_down = filtered_display_df.iloc[0]
                st.warning(
                    f"**Longest Inactive Product:** **{longest_down['Store + Pump + Fuel Type']}** has been inactive for **{longest_down['Downtime Since Last Transaction']}** (or **{longest_down['Downtime (Hours)']:.1f} hours**)."
                )
            
            # --------------------------------------------------------------------------
            # --- CONTINUING WITH EXISTING GPD CHART ---
            # --------------------------------------------------------------------------
            st.markdown("---")
            st.subheader(f"Daily Gallons Dispensed Trend for {selected_category}")


            # --- 2. Pump Filter (MULTI-SELECT FILTER INSIDE TAB 2) ---
            available_pumps = sorted(category_filtered_df[COL_FUEL_POSIT].unique().astype(str).tolist())
            selected_pumps = st.multiselect(
                'Select Pump(s) to View on Chart (Select All to View All):',
                options=available_pumps,
                default=available_pumps
            )
            
            if not selected_pumps:
                st.warning("Please select at least one pump to view the chart.")
            else:
                # --- 3. Filter the DataFrame based on pump selection ---
                pump_filtered_df = category_filtered_df[category_filtered_df[COL_FUEL_POSIT].astype(str).isin(selected_pumps)]
                
                # Check if filtering resulted in empty data
                if pump_filtered_df.empty:
                    st.warning("No data found for the selected pumps and product category.")
                    st.stop()
                    
                # --- 4. Aggregate daily quantity by pump, grouping by Calendar_Date (datetime object) ---
                daily_gpd_df = pump_filtered_df.groupby([
                    'Calendar_Date', COL_STORE, COL_FUEL_POSIT
                ]).agg(
                    Gallons_Dispensed=(COL_QUANTITY, 'sum'),
                    Transaction_Count=(COL_QUANTITY, 'size')
                ).reset_index()

                # 5. Create combined identifier for the chart lines
                daily_gpd_df['Pump_ID'] = daily_gpd_df[COL_STORE].astype(str) + ' - Pump ' + daily_gpd_df[COL_FUEL_POSIT].astype(str)
                
                # Re-create the formatted date column for display later
                daily_gpd_df['Start Date Formatted'] = daily_gpd_df['Calendar_Date'].dt.strftime('%m/%d/%Y')

                # 6. CRITICAL FIX: Reindex to force all dates from the full dataset range
                if not daily_gpd_df.empty:
                    # Determine the full date range based on the category/store filter context
                    full_date_min = category_filtered_df['Calendar_Date'].min()
                    full_date_max = category_filtered_df['Calendar_Date'].max()
                    
                    full_date_range = pd.date_range(start=full_date_min, end=full_date_max, freq='D')
                    
                    # Prepare the multi-index: all combinations of Pump_ID and Date
                    all_pump_ids = daily_gpd_df['Pump_ID'].unique()
                    multi_index = pd.MultiIndex.from_product(
                        [full_date_range, all_pump_ids],
                        names=['Calendar_Date', 'Pump_ID']
                    )

                    # Reindex the daily data against the full multi-index, filling missing values (downtime) with 0
                    daily_gpd_df = daily_gpd_df.set_index(['Calendar_Date', 'Pump_ID'])
                    daily_gpd_df = daily_gpd_df.reindex(multi_index, fill_value=0).reset_index()
                    
                    # Re-create the formatted date column based on the date index for the tooltips
                    daily_gpd_df['Start Date Formatted'] = daily_gpd_df['Calendar_Date'].dt.strftime('%m/%d/%Y')
                    
                    # Ensure numeric columns that were not grouped are 0
                    daily_gpd_df['Gallons_Dispensed'] = daily_gpd_df['Gallons_Dispensed'].fillna(0)
                    daily_gpd_df['Transaction_Count'] = daily_gpd_df['Transaction_Count'].fillna(0)
                    
                    # Cleanup residual columns from aggregation grouping that are not needed after reindex
                    daily_gpd_df = daily_gpd_df.drop(columns=[COL_STORE, COL_FUEL_POSIT], errors='ignore')

                
                # 7. Visualization: Time Series Line Chart (Downtime Monitoring)
                
                # Sort by Calendar_Date (datetime object) for true chronology
                fig_daily_gpd = px.line(
                    daily_gpd_df.sort_values(by='Calendar_Date', ascending=True), 
                    x='Calendar_Date', # Use datetime object for correct axis scaling
                    y='Gallons_Dispensed',
                    color='Pump_ID',
                    title=f'{selected_category} Dispensed Per Day by Store + Pump Position',
                    labels={'Gallons_Dispensed': 'Gallons Dispensed (Total)', 'Calendar_Date': 'Date'},
                    hover_data={
                        'Transaction_Count': True, 
                        'Calendar_Date': "|%m/%d/%Y" # Format the date in the tooltip
                    },
                    template="plotly_white",
                    markers=True,
                    height=CHART_HEIGHT
                )

                # Ensure x-axis shows the full range, even for inactive pumps
                if not daily_gpd_df.empty:
                     fig_daily_gpd.update_xaxes(range=[daily_gpd_df['Calendar_Date'].min(), daily_gpd_df['Calendar_Date'].max()])
                
                st.plotly_chart(fig_daily_gpd, use_container_width=True)
                st.markdown("---")

                # 8. Summary Table (Average GPD across all days for quick comparison)
                st.subheader(f"Summary: Average GPD by Pump for {selected_category}")
                
                # The summary table should now correctly include the days with 0 transactions from the reindexed data
                summary_gpd_df = daily_gpd_df.groupby('Pump_ID').agg(
                    Average_GPD=('Gallons_Dispensed', 'mean'),
                    Total_Days_Observed=('Calendar_Date', 'nunique'),
                    Total_Transactions=('Transaction_Count', 'sum')
                ).reset_index()
                
                # Sort by average GPD for easy detection of low performers
                summary_gpd_df = summary_gpd_df.sort_values(by='Average_GPD', ascending=True)

                st.dataframe(summary_gpd_df.style.format({
                    'Average_GPD': '{:,.0f}',
                    'Total_Transactions': '{:,.0f}'
                }))


# --------------------------------------------------------------------------
# --- TAB 3: HISTORICAL FUEL SALES (Macro View) ---
# --------------------------------------------------------------------------
with tab3:
    st.header("Historical Fuel Sales: Gallons Dispensed")
    st.info("This view aggregates sales by **Store** and **Product Category** over the selected date range. For Year-over-Year (YOY) comparison, select a range that spans multiple years.")
    
    # --- 1. Date Range Filter ---
    st.subheader("Date Range Selection")
    
    # Default selection: Last 60 days from max date
    default_start = max_date - pd.Timedelta(days=60).to_pytimedelta() if (max_date - pd.Timedelta(days=60).to_pytimedelta()) > min_date else min_date
    
    date_cols = st.columns(2)
    with date_cols[0]:
        start_date = st.date_input("Start Date", value=default_start, min_value=min_date, max_value=max_date)
    with date_cols[1]:
        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.error("Error: End Date must fall after Start Date.")
        st.stop()
        
    # --- 2. Store Multi-Select Filter ---
    available_stores = sorted(df[COL_STORE].unique().astype(str).tolist())
    selected_stores_macro = st.multiselect(
        'Select Store(s) to View:',
        options=available_stores,
        default=available_stores # Default to showing all stores
    )

    # --- 3. Product Category Toggle (The Bubble) ---
    selected_category_macro = st.radio(
        "Filter by Fuel Type:",
        ('Gasoline', 'Diesel'),
        horizontal=True
    )
    st.markdown("---")
        
    # --- 4. Filter Data ---
    # Filter using Python date objects for comparison AND the new category filter
    macro_filtered_df = df[
        (df['Calendar_Date'].dt.date >= start_date) & 
        (df['Calendar_Date'].dt.date <= end_date) &
        (df[COL_STORE].astype(str).isin(selected_stores_macro)) &
        (df['Product_Category'] == selected_category_macro) # APPLYING NEW FILTER HERE
    ].copy()
    
    if macro_filtered_df.empty:
        st.warning(f"No {selected_category_macro} data found for the selected stores and date range.")
    else:
        # --- 5. Aggregation ---
        # Group by Store, Date, and Product Category
        daily_sales_df = macro_filtered_df.groupby([
            'Calendar_Date', COL_STORE, 'Product_Category', 'Year'
        ]).agg(
            Gallons_Dispensed=(COL_QUANTITY, 'sum')
        ).reset_index()

        # Product_Key will now reflect Store and the Category (which is filtered)
        daily_sales_df['Product_Key'] = daily_sales_df[COL_STORE].astype(str) + ' - ' + daily_sales_df['Product_Category']
        
        # --- 6. Visualization: Absolute Date Trend ---
        st.subheader(f"{selected_category_macro} Sales Trend by Day (Absolute Dates)")

        fig_absolute = px.line(
            daily_sales_df.sort_values(by='Calendar_Date', ascending=True),
            x='Calendar_Date',
            y='Gallons_Dispensed',
            color='Product_Key',
            title=f'Daily Sales Volume by Store ({selected_category_macro})',
            labels={'Gallons_Dispensed': 'Gallons Dispensed (Total)', 'Calendar_Date': 'Date'},
            template="plotly_white",
            height=CHART_HEIGHT
        )
        
        st.plotly_chart(fig_absolute, use_container_width=True)
        st.markdown("---")

        # --- 7. Visualization: YOY Comparison Trend Overlay ---
        st.subheader(f"{selected_category_macro} Year-over-Year Comparison Trend Overlay")
        
        # Aggregate data again, using DayOfYear to align the trends across years
        yoy_df = macro_filtered_df.groupby([
            'DayOfYear', COL_STORE, 'Product_Category', 'Year'
        ]).agg(
            Gallons_Dispensed=(COL_QUANTITY, 'sum')
        ).reset_index()
        
        yoy_df['Product_Year_Key'] = (
            yoy_df[COL_STORE].astype(str) + ' - ' + 
            yoy_df['Product_Category'] + ' - ' + 
            yoy_df['Year'].astype(str)
        )
        
        fig_yoy = px.line(
            yoy_df.sort_values(by='DayOfYear', ascending=True),
            x='DayOfYear',
            y='Gallons_Dispensed',
            color='Product_Year_Key',
            title=f'Sales Trend Overlay ({selected_category_macro}, X-Axis = Day Index in the Year)',
            labels={'Gallons_Dispensed': 'Gallons Dispensed (Total)', 'DayOfYear': 'Day Index in the Year'},
            template="plotly_white",
            height=CHART_HEIGHT
        )
        
        st.plotly_chart(fig_yoy, use_container_width=True)
        st.markdown(
            "**Note on Comparison:** Retailers typically use the **4-5-4 Calendar** or **Day-of-Week Matching** to ensure accurate YOY comparisons."
            " The chart above uses the **Day Index in the Year** to overlay trends from different years, providing a visual comparison of the seasonal shape."
        )
