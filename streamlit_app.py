import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import re
import zipfile
import io
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from reportlab.graphics.barcode import code128
from reportlab.lib.enums import TA_LEFT

# --- Global Constants & Configuration ---
LOCATION_ORDER = ["Heat", "Cold Room", "Powder",
                  "Tower", "Pour drum", "Production", "Warehouse"]
FONT_NORMAL = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
ALLERGEN_COLOR = colors.pink


def natural_sort_key(s):
    """Returns a tuple for natural sorting (hashable)."""
    return tuple(int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s)))


def preprocess_location_map(uploaded_map_file):
    """
    Reads the single-column location mapping CSV, parses it, and creates a structured DataFrame.
    This version simplifies the classification to be quantity-agnostic.
    """
    df_map = pd.read_csv(uploaded_map_file, header=1)
    df_map.columns = ['full_location']
    df_map.dropna(subset=['full_location'], inplace=True)
    df_map['full_location'] = df_map['full_location'].astype(str)

    pat = re.compile(r'([^()]+)(?:\s*\(([^)]+)\))?')

    def parse_location(full_loc):
        match = pat.match(full_loc)
        if match:
            code = match.group(1).strip()
            name_part = f"({match.group(2).strip()})" if match.group(2) else ''
            return code, name_part
        return full_loc.strip(), ''

    parsed_data = df_map['full_location'].apply(parse_location)
    df_map['Code'] = parsed_data.apply(lambda x: x[0])
    df_map['Name'] = parsed_data.apply(lambda x: x[1])

    df_map = df_map[~df_map['Name'].str.contains(
        '(WIP|NC)', case=False, na=False)]

    def get_base_location_type(row):
        code = str(row['Code'])
        name = str(row['Name']).upper()
        if '(HOTBOX)' in name:
            return "Heat"
        if '(POWDER)' in name:
            return "Powder"
        if code.startswith('CR'):
            return "Cold Room"
        if code.startswith(('PFL-CP', 'PFR-CP')):
            return "Pour drum"  # Base type
        if code.startswith(('KXFL', 'KXFR')):
            return "Tower"  # Base type
        if code.startswith('Z'):
            return "Tower"  # Z is a type of Tower
        if code.startswith(('PFL', 'PFR')):
            return "Production"
        if code.startswith(('WFL', 'WFR', 'LOFL', 'LOFR')):
            return "Warehouse"
        return "Unknown"

    df_map['Location Type'] = df_map.apply(get_base_location_type, axis=1)
    df_map['Is Allergen'] = df_map['Name'].str.contains(
        '(ALLERGEN)', case=False, na=False)

    df_map.set_index('Code', inplace=True)
    return df_map


def classify_location(base_type, qty_required, max_tower_qty, max_pour_drum_qty):
    """
    Refines the base location type based on quantity limits.
    This function now holds all the quantity-dependent logic.
    """
    if base_type == "Tower" and qty_required >= max_tower_qty:
        return "Tower overweight"
    if base_type == "Pour drum" and qty_required >= max_pour_drum_qty:
        return "Pour drum overweight"
    return base_type  # Return the original type if no rules are met


def format_output_df(df_priority):
    """
    Formats a DataFrame for PDF export, applying sorting.
    This version is now robust and handles cases where no standard locations are found.
    """
    if df_priority.empty:
        return pd.DataFrame()

    # Define the final schema of the output DataFrame
    final_columns = ["Location", "Location Description", "RM name", "RM code", "Batch number",
                     "Available Quantity", "Quantity required", "Expiry Status", "Needs Highlighting",
                     "Is Allergen", "Location Priority"]

    output_columns = ["Location Type", "Location Description", "Description", "Component", "Batch Nr.1",
                      "Available Quantity", "Quantity required", "Expiry Status", "Needs Highlighting", "Is Allergen", "Location Priority"]
    df_final = df_priority[output_columns].copy()
    df_final.rename(columns={"Location Type": "Location", "Description": "RM name",
                    "Component": "RM code", "Batch Nr.1": "Batch number"}, inplace=True)
    df_final['RM name'] = df_final['RM name'].str[:20]

    formatted_rows = []
    for location in LOCATION_ORDER:
        subset = df_final[df_final["Location"] == location]
        if not subset.empty:
            header_row = {"Location": location, "Location Description": "", "RM name": "", "RM code": "", "Batch number": "", "Available Quantity": "",
                          "Quantity required": "", "Expiry Status": "", "Needs Highlighting": False, "Is Allergen": False, "Location Priority": 0}
            formatted_rows.append(header_row)

            subset['sort_key'] = subset['Location Description'].apply(
                natural_sort_key)
            subset = subset.sort_values(
                by=['sort_key', 'RM code', 'Batch number']).drop(columns=['sort_key'])
            for _, row in subset.iterrows():
                formatted_rows.append(row.to_dict())

    # --- FIXED LOGIC ---
    # If no rows were formatted (e.g., only 'overweight' locations existed),
    # return an empty DataFrame but WITH the correct columns.
    if not formatted_rows:
        return pd.DataFrame(columns=final_columns)

    df_output = pd.DataFrame(formatted_rows)
    return df_output


def process_data(df, location_map, max_tower_qty, max_pour_drum_qty):
    """
    Processes the production ticket using the location map and dynamic quantity limits.
    """
    first_row = df.iloc[0]
    production_date = pd.to_datetime(
        first_row["Current date marked the beginning"], format='%d%m%Y', errors='coerce')
    product_info = {"Production Ticket Nr": first_row["Production Ticket Nr"], "Wording": first_row["Wording"], "Product Code": first_row["Product Code"],
                    "Quantity Launched": df["Quantity launched Theoretical"].astype(float).max(), "Production Date": production_date.date()}
    unique_materials = df.drop_duplicates(subset=["Component", "Description"])
    product_info["Quantity Produced"] = unique_materials["Quantity required"].astype(
        float).sum()
    product_info["Raw Material Count"] = df["Component"].nunique()

    df_filtered = df[df['Location Description'].isin(
        location_map.index)].copy()

    df_filtered["Quantity required"] = pd.to_numeric(
        df_filtered["Quantity required"], errors='coerce').fillna(0)
    df_filtered["Available Quantity"] = pd.to_numeric(df_filtered["Available Quantity"].astype(
        str).str.replace(',', '.'), errors='coerce').fillna(0)

    # Map base properties from the location map
    df_filtered['Base Location Type'] = df_filtered['Location Description'].map(
        location_map['Location Type'])
    df_filtered['Is Allergen'] = df_filtered['Location Description'].map(
        location_map['Is Allergen'])

    # --- UPDATED: Refine location type based on quantity limits ---
    df_filtered['Location Type'] = df_filtered.apply(
        lambda row: classify_location(
            row['Base Location Type'], row['Quantity required'], max_tower_qty, max_pour_drum_qty),
        axis=1
    )

    df_filtered['DLUO_dt'] = pd.to_datetime(
        df_filtered['DLUO'], format='%d%m%Y', errors='coerce')
    one_month_later = production_date + pd.DateOffset(months=1)
    conditions = [df_filtered['DLUO_dt'] < production_date,
                  (df_filtered['DLUO_dt'] >= production_date) & (df_filtered['DLUO_dt'] < one_month_later)]
    df_filtered['Expiry Status'] = np.select(
        conditions, ['Expired', 'Expiring Soon'], default='OK')

    # --- UPDATED: Priority map includes new "overweight" category ---
    priority_map = {loc: i + 1 for i, loc in enumerate(LOCATION_ORDER)}
    priority_map['Tower overweight'] = 8
    priority_map['Pour drum overweight'] = 8  # Assign low priority
    df_filtered["Location Priority"] = df_filtered["Location Type"].map(
        priority_map)

    df_filtered.sort_values(
        by=["Component", "Location Priority", "Batch Nr.1"], inplace=True)

    # (The rest of the ranking and priority assignment logic remains the same)
    df_filtered['Rank'] = df_filtered.groupby('Component').cumcount() + 1
    all_priority_groups = []
    for component_code, group in df_filtered.groupby("Component"):
        required_qty, collected_qty = group["Quantity required"].iloc[0], 0
        group_copy = group.copy()
        first_priority_indices = []
        for index, row in group_copy.iterrows():
            if collected_qty < required_qty:
                first_priority_indices.append(index)
                collected_qty += row['Available Quantity']
            else:
                break
        group_copy.loc[first_priority_indices,
                       'Assigned Priority'] = 'First Priority'

        remaining_rows = group_copy[group_copy['Assigned Priority'].isnull()].copy(
        )
        if not remaining_rows.empty:
            start_rank = remaining_rows['Rank'].min()
            remaining_rows['Priority_Rank'] = remaining_rows['Rank'] - \
                start_rank + 2

            def assign_by_rank(rank):
                if rank == 2:
                    return 'Second Priority'
                if rank == 3:
                    return 'Third Priority'
                return 'Leftovers'
            group_copy.loc[remaining_rows.index, 'Assigned Priority'] = remaining_rows['Priority_Rank'].apply(
                assign_by_rank)

        group_copy['Needs Highlighting'] = len(first_priority_indices) > 1
        all_priority_groups.append(group_copy)

    if not all_priority_groups:
        return product_info, {}
    df_with_priorities = pd.concat(all_priority_groups)

    priority_dfs_raw = {p_level: df_with_priorities[df_with_priorities['Assigned Priority'] == p_level] for p_level in [
        'First Priority', 'Second Priority', 'Third Priority', 'Leftovers']}
    priority_dfs_formatted = {name: format_output_df(
        df_raw) for name, df_raw in priority_dfs_raw.items() if not df_raw.empty}
    return product_info, priority_dfs_formatted

# (The generate_pdf function remains unchanged)


def generate_pdf(product_info, priority_dfs, barcode_locations, file_configs, content_to_include):
    # This function is correct from the previous version. No changes needed.
    generated_files = []
    for config in file_configs:
        file_num, total_files, locations_for_this_file = config[
            'file_num'], config['total_files'], config['locations']
        pdf_filename = f"{product_info['Production Ticket Nr']}_{file_num}_of_{total_files}.pdf"
        doc = SimpleDocTemplate(pdf_filename, pagesize=A4, leftMargin=0.5*inch,
                                rightMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='LeftAlign',
                   alignment=TA_LEFT, fontName=FONT_NORMAL))
        styles['Normal'].fontName = FONT_NORMAL
        styles['h1'].fontName = FONT_BOLD
        elements, content_added_to_this_pdf, is_first_content_block = [], False, True
        title_text = f"Production Ticket Information - {product_info['Production Ticket Nr']}"
        if total_files > 1:
            title_text += f" ({file_num} / {total_files})"
        elements.append(Paragraph(title_text, styles['h1']))
        info_copy = {k: v for k, v in product_info.items() if k !=
                     "Production Ticket Nr"}
        elements.append(Table([[key, str(value)] for key, value in info_copy.items()], colWidths=[2*inch, 5*inch], style=[('ALIGN', (0, 0), (-1, -1), 'LEFT'), ('FONTNAME',
                        (0, 0), (0, -1), FONT_BOLD), ('FONTNAME', (1, 0), (1, -1), FONT_NORMAL), ('BOTTOMPADDING', (0, 0), (-1, 0), 12), ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
        elements.append(Spacer(1, 0.2*inch))
        for priority_name in content_to_include:
            if priority_name in priority_dfs:
                df_output = priority_dfs[priority_name][priority_dfs[priority_name]['Location'].isin(
                    locations_for_this_file)]
                if not df_output.empty:
                    content_added_to_this_pdf = True
                    if not is_first_content_block:
                        elements.append(PageBreak())
                    is_first_content_block = False
                    elements.append(Paragraph(priority_name, styles['h1']))
                    elements.append(Spacer(1, 0.1*inch))
                    headers = ["Location", "Location\nDescription", "RM name", "RM code\n& Barcode",
                               "Batch\nnumber", "Available\nQuantity", "Quantity\nRequired"]
                    table_data = [headers]
                    for _, row in df_output.iterrows():
                        if row['RM code'] == '':
                            table_data.append(list(row.drop(
                                ['Expiry Status', 'Needs Highlighting', 'Is Allergen', 'Location Priority'])))
                            continue
                        barcode_cell = Paragraph(
                            str(row['RM code']), styles['Normal'])
                        if row['Location'] in barcode_locations:
                            barcode_cell = [barcode_cell, Spacer(10, 0), code128.Code128(
                                str(row['RM code']), barHeight=0.2*inch, barWidth=0.008*inch)]
                        table_data.append([row['Location'], Paragraph(str(row['Location Description']), styles['Normal']), Paragraph(str(
                            row['RM name']), styles['LeftAlign']), barcode_cell, Paragraph(str(row['Batch number']), styles['Normal']), row['Available Quantity'], row['Quantity required']])
                    output_table = Table(table_data, colWidths=[
                                         0.8*inch, 1*inch, 2*inch, 1*inch, 1*inch, 0.8*inch, 0.8*inch], repeatRows=1)
                    base_style = [('BACKGROUND', (0, 0), (-1, 0), colors.darkblue), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                  ('FONTNAME', (0, 0), (-1, 0), FONT_BOLD), ('GRID', (0, 0), (-1, -1), 1, colors.black), ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('ALIGN', (2, 1), (2, -1), 'LEFT')]
                    dynamic_styles = []
                    for i, row_data in enumerate(table_data[1:], 1):
                        if row_data[1] == '':
                            dynamic_styles.extend([('BACKGROUND', (0, i), (-1, i), colors.lightgrey), (
                                'TEXTCOLOR', (0, i), (-1, i), colors.black), ('FONTNAME', (0, i), (0, i), FONT_BOLD)])
                        else:
                            try:
                                original_row = df_output[df_output['Batch number'] == str(
                                    row_data[4].text)].iloc[0]
                                if original_row['Is Allergen']:
                                    dynamic_styles.append(
                                        ('BACKGROUND', (2, i), (2, i), ALLERGEN_COLOR))
                                if original_row['Expiry Status'] == 'Expired':
                                    dynamic_styles.append(
                                        ('BACKGROUND', (2, i), (2, i), colors.lightcoral))
                                elif original_row['Expiry Status'] == 'Expiring Soon':
                                    dynamic_styles.append(
                                        ('BACKGROUND', (2, i), (2, i), colors.moccasin))
                                if original_row['Needs Highlighting']:
                                    dynamic_styles.append(
                                        ('BACKGROUND', (5, i), (6, i), colors.yellow))
                            except (IndexError, AttributeError):
                                pass
                    output_table.setStyle(TableStyle(
                        base_style + dynamic_styles))
                    elements.append(output_table)
        if content_added_to_this_pdf:
            doc.build(elements)
            generated_files.append(pdf_filename)
    return generated_files


# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("Production Ticket Processor")

st.header("Step 1: Upload Files")
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader(
        "Upload Production Ticket (XLS/XLSX)", type=["xls", "xlsx"])
with col2:
    location_map_file = st.file_uploader(
        "Upload Location Mapping File (CSV)", type=["csv"])

st.sidebar.header("Step 2: Configure PDF Output")

# --- NEW: UI for Quantity Limits ---
st.sidebar.markdown("---")
st.sidebar.write("Set Quantity Limits:")
# Ensure all numeric arguments are floats
max_tower_qty = st.sidebar.number_input(
    "Max Qty for Tower:", value=2.0, step=0.1, format="%.3f")
max_pour_drum_qty = st.sidebar.number_input(
    "Max Qty for Pour drum:", value=20.0, step=1.0, format="%.2f")  # Changed 10 to 10.0
st.sidebar.markdown("---")

split_option = st.sidebar.radio(
    "PDF Splitting:", ("Single File", "Split into 2 Files", "Split into 3 Files"))
# (Splitting UI logic remains the same)
file_configs, is_valid_config = [], True
if split_option == "Single File":
    file_configs = [
        {'file_num': 1, 'total_files': 1, 'locations': LOCATION_ORDER}]
else:
    num_splits = 2 if "2" in split_option else 3
    assignments = []
    for i in range(num_splits):
        assigned_so_far = sum(assignments, [])
        available_options = [
            loc for loc in LOCATION_ORDER if loc not in assigned_so_far]
        selection = st.sidebar.multiselect(
            f"Locations for File {i+1}:", available_options)
        assignments.append(selection)
    if len(sum(assignments, [])) != len(LOCATION_ORDER):
        st.sidebar.warning("All locations must be assigned to a file.")
        is_valid_config = False
    else:
        file_configs = [{'file_num': i+1, 'total_files': num_splits,
                         'locations': assignments[i]} for i in range(num_splits)]

st.sidebar.markdown("---")
barcode_locations_selection = st.sidebar.multiselect(
    "Generate barcodes for which locations?", LOCATION_ORDER, default=["Tower", "Pour drum", "Production"])
st.sidebar.markdown("---")
st.sidebar.write("Select Content to Include:")
include_p2 = st.sidebar.checkbox("Include Second Priority", True)
include_p3 = st.sidebar.checkbox("Include Third Priority", True)
include_leftovers = st.sidebar.checkbox("Include Leftovers", True)
content_to_include = ['First Priority']
if include_p2:
    content_to_include.append('Second Priority')
if include_p3:
    content_to_include.append('Third Priority')
if include_leftovers:
    content_to_include.append('Leftovers')

if uploaded_file is not None and location_map_file is not None:
    try:
        location_map = preprocess_location_map(location_map_file)
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = xls.sheet_names[0]
        df_preview = pd.read_excel(xls, sheet_name=sheet_name)
        header_row_index = df_preview.apply(lambda row: row.astype(
            str).str.contains("Batch Nr").any(), axis=1).idxmax() + 1
        string_converters = {'Production Ticket Nr': str, 'Product Code': str, 'Batch Nr': str,
                             'Current date marked the beginning': str, 'Component': str, 'Batch Nr.1': str, 'DLUO': str}
        df = pd.read_excel(xls, sheet_name=sheet_name,
                           header=header_row_index, converters=string_converters)

        # --- UPDATED: Pass the new quantity limits to the processing function ---
        product_info, priority_dataframes = process_data(
            df, location_map, max_tower_qty, max_pour_drum_qty)

        st.header("Step 3: Review and Generate")
        st.subheader("Production Information")
        st.json(product_info, expanded=False)
        st.subheader("First Priority Picking List (Preview)")

        if 'First Priority' in priority_dataframes and not priority_dataframes['First Priority'].empty:
            # --- FIXED: Use a clean selection of columns for the preview ---
            preview_df = priority_dataframes['First Priority']

            # These are the final, user-facing column names from format_output_df
            columns_to_show = [
                'Location', 'Location Description', 'RM name', 'RM code',
                'Batch number', 'Available Quantity', 'Quantity required'
            ]

            # Filter the DataFrame to only these columns for a clean preview
            st.dataframe(preview_df[columns_to_show])

            if is_valid_config and st.sidebar.button("Generate Full Picking PDF"):
                with st.spinner('Creating PDF(s)...'):
                    pdf_filenames = generate_pdf(
                        product_info, priority_dataframes, barcode_locations_selection, file_configs, content_to_include)
                    if not pdf_filenames:
                        st.sidebar.error(
                            "No content available for the selected locations/priorities.")
                    elif len(pdf_filenames) == 1:
                        with open(pdf_filenames[0], "rb") as f:
                            st.sidebar.download_button(
                                "Download PDF", f, file_name=pdf_filenames[0], mime="application/pdf")
                        os.remove(pdf_filenames[0])
                    else:
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zf:
                            for f in pdf_filenames:
                                zf.write(f, os.path.basename(f))
                        st.sidebar.download_button("Download Picking Lists (ZIP)", zip_buffer.getvalue(
                        ), f"{product_info['Production Ticket Nr']}_picking_lists.zip", "application/zip")
                        for f in pdf_filenames:
                            os.remove(f)
        else:
            st.warning(
                "No valid first-priority items were found based on the provided files.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error(
            "Please check that both files are correct and that all locations in the ticket exist in the mapping file.")
else:
    st.info(
        "Please upload both a Production Ticket and a Location Mapping File to begin.")
