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
ALLERGEN_COLOR = colors.lightblue


def natural_sort_key(s):
    """Returns a tuple for natural sorting (hashable)."""
    return tuple(int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s)))


def preprocess_location_map(uploaded_map_file):
    """
    Reads the mapping CSV and correctly assigns detailed Sub-Types for prioritization,
    including the new Powder hierarchy.
    """
    df_map = pd.read_csv(uploaded_map_file, skiprows=2,
                         header=None, names=['full_location'])
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

    df_map.drop_duplicates(subset=['Code'], keep='first', inplace=True)
    df_map = df_map[~df_map['Name'].str.contains(
        '(WIP|NC)', case=False, na=False)]

    def get_location_attributes(row):
        code = str(row['Code'])
        name = str(row['Name']).upper()

        # This function now returns a detailed SubType for prioritization
        sub_type = "Unknown"
        if '(HOTBOX)' in name or code.startswith(('HTFL', 'HTFR')):
            sub_type = "Heat"
        # --- NEW HIERARCHY LOGIC ---
        elif code.startswith(('PFL-PW', 'PFR-PW')):
            sub_type = "Powder (PW)"  # Specific, high-priority powder
        elif '(POWDER)' in name:
            sub_type = "Powder"  # General, lower-priority powder
        # ---
        elif code.startswith('CR'):
            sub_type = "Cold Room"
        elif code.startswith(('KXFL', 'KXFR')):
            sub_type = "Tower"
        elif code.startswith(('PFL-CP', 'PFR-CP')):
            sub_type = "Pour drum"
        elif code.startswith(('PFL', 'PFR')):
            sub_type = "Production"
        elif code.startswith(('WFL', 'WFR', 'LOFL', 'LOFR')):
            sub_type = "Warehouse"

        is_allergen = '(ALLERGEN)' in name
        return sub_type, is_allergen

    attributes = df_map.apply(get_location_attributes, axis=1)
    df_map['Location SubType'] = attributes.apply(lambda x: x[0])
    df_map['Is Allergen'] = attributes.apply(lambda x: x[1])

    # Create the simplified 'Location Type' for display in the final PDF
    def map_subtype_to_main_type(subtype):
        if 'Powder' in str(subtype):
            return 'Powder'
        return subtype
    df_map['Location Type'] = df_map['Location SubType'].apply(
        map_subtype_to_main_type)

    df_map.set_index('Code', inplace=True)
    return df_map


def classify_location(base_type, qty_required, max_tower_qty, max_pour_drum_qty):
    """
    Refines the base location type based on quantity limits.
    """
    if base_type == "Tower" and qty_required >= max_tower_qty:
        return "Tower overweight"
    if base_type == "Pour drum" and qty_required >= max_pour_drum_qty:
        return "Pour drum overweight"
    return base_type


def format_output_df(df_priority):
    """
    Formats a DataFrame for PDF export. This version now correctly finds and
    includes 'Unknown' locations (for 'To Be Prod.' items).
    """
    if df_priority.empty:
        return pd.DataFrame(columns=["Location", "Location Description", "RM name", "RM code", "Batch number",
                                     "Available Quantity", "Quantity required", "Expiry Status", "Needs Highlighting",
                                     "Is Allergen", "Location Priority", "To Be Prod."])

    output_columns = ["Location Type", "Location Description", "Description", "Component", "Batch Nr.1",
                      "Available Quantity", "Quantity required", "Expiry Status", "Needs Highlighting",
                      "Is Allergen", "Location Priority", "To Be Prod."]

    df_final = df_priority[output_columns].copy()
    df_final.rename(columns={"Location Type": "Location", "Description": "RM name",
                    "Component": "RM code", "Batch Nr.1": "Batch number"}, inplace=True)
    df_final['RM name'] = df_final['RM name'].str[:20]

    formatted_rows = []
    # 1. Process all the standard, known locations first
    for location in LOCATION_ORDER:
        subset = df_final[df_final["Location"] == location]
        if not subset.empty:
            header_row = {
                "Location": location, "Location Description": "", "RM name": "", "RM code": "",
                "Batch number": "", "Available Quantity": "", "Quantity required": "",
                "Expiry Status": "", "Needs Highlighting": False, "Is Allergen": False,
                "Location Priority": 0, "To Be Prod.": False
            }
            formatted_rows.append(header_row)

            if location in ["Tower", "Pour drum", "Powder"]:
                subset['sort_key'] = subset['Location Description'].apply(
                    natural_sort_key)
                subset = subset.sort_values(
                    by=['sort_key', 'RM code', 'Batch number']).drop(columns=['sort_key'])
            else:
                subset = subset.sort_values(by=['RM name', 'Batch number'])

            for _, row in subset.iterrows():
                formatted_rows.append(row.to_dict())

    # --- FIXED LOGIC: After processing standard locations, check for 'Unknown' ---
    unknown_subset = df_final[df_final["Location"] == "Unknown"]
    if not unknown_subset.empty:
        # Add a header for this special section
        header_row = {
            "Location": "To Be Prod.", "Location Description": "", "RM name": "", "RM code": "",
            "Batch number": "", "Available Quantity": "", "Quantity required": "",
            "Expiry Status": "", "Needs Highlighting": False, "Is Allergen": False,
            "Location Priority": 99, "To Be Prod.": True
        }
        formatted_rows.append(header_row)

        # Sort and add the 'Unknown' items
        unknown_subset = unknown_subset.sort_values(
            by=['RM name', 'Batch number'])
        for _, row in unknown_subset.iterrows():
            # Update the 'Location' field for better display
            row_dict = row.to_dict()
            row_dict['Location'] = 'To Be Prod.'
            formatted_rows.append(row_dict)

    if not formatted_rows:
        return pd.DataFrame(columns=df_final.columns)

    df_output = pd.DataFrame(formatted_rows)
    return df_output


def process_data(df, location_map, max_tower_qty, max_pour_drum_qty):

    debug_data = {}

    first_row = df.iloc[0]
    production_date = pd.to_datetime(
        first_row["Current date marked the beginning"], format='%d%m%Y', errors='coerce')
    product_info = {
        "Production Ticket Nr": first_row["Production Ticket Nr"],
        "Batch Nr": first_row["Batch Nr"],
        "Wording": first_row["Wording"],
        "Product Code": first_row["Product Code"],
        "Quantity Launched": df["Quantity launched Theoretical"].astype(float).max(),
        "Production Date": production_date.date()
    }

    unique_materials = df.drop_duplicates(subset=["Component", "Description"])
    product_info["Quantity Produced"] = unique_materials["Quantity required"].astype(
        float).sum()
    product_info["Raw Material Count"] = df["Component"].nunique()

    df_processed = df.copy()

    pat = re.compile(r'([^()]+)(?:\s*\(([^)]+)\))?')

    def parse_ticket_location(full_loc):
        match = pat.match(str(full_loc))
        if match:
            return match.group(1).strip()
        return str(full_loc).strip()

    df_processed['Location Code'] = df_processed['Location Description'].apply(
        parse_ticket_location)

    # Map properties, leaving unmapped locations as NaN for now
    df_processed['Location SubType'] = df_processed['Location Code'].map(
        location_map['Location SubType'])
    df_processed['Is Allergen'] = df_processed['Location Code'].map(
        location_map['Is Allergen'])

    # Fill NaN for unmapped locations with 'Unknown'
    df_processed['Location SubType'].fillna('Unknown', inplace=True)
    df_processed['Is Allergen'].fillna(False, inplace=True)

    debug_data['1_mapped_ticket'] = df_processed.copy()  # New debug step

    df_processed["Quantity required"] = pd.to_numeric(
        df_processed["Quantity required"], errors='coerce').fillna(0)
    df_processed["Available Quantity"] = pd.to_numeric(df_processed["Available Quantity"].astype(
        str).str.replace(',', '.'), errors='coerce').fillna(0)

    def map_subtype_to_main_type(subtype):
        if 'Powder' in str(subtype):
            return 'Powder'
        return subtype
    df_processed['Location Type'] = df_processed['Location SubType'].apply(
        map_subtype_to_main_type)

    def refine_location_subtype(row):
        subtype = row['Location SubType']
        qty_required = row['Quantity required']
        if subtype == "Tower" and qty_required >= max_tower_qty:
            return "Tower overweight"
        if subtype == "Pour drum" and qty_required >= max_pour_drum_qty:
            return "Pour drum overweight"
        return subtype
    df_processed['Final SubType'] = df_processed.apply(
        refine_location_subtype, axis=1)

    df_processed['DLUO_dt'] = pd.to_datetime(
        df_processed['DLUO'], format='%d%m%Y', errors='coerce')
    one_month_later = production_date + pd.DateOffset(months=1)
    conditions = [df_processed['DLUO_dt'] < production_date,
                  (df_processed['DLUO_dt'] >= production_date) & (df_processed['DLUO_dt'] < one_month_later)]
    df_processed['Expiry Status'] = np.select(
        conditions, ['Expired', 'Expiring Soon'], default='OK')

    priority_map = {loc: i + 1 for i, loc in enumerate(LOCATION_ORDER)}
    priority_map.update({'Powder (PW)': 3.1, 'Powder': 3.2, 'Tower overweight': 8,
                        'Pour drum overweight': 8, 'Unknown': 99})  # Add Unknown with lowest priority

    df_processed["Location Priority"] = df_processed["Final SubType"].map(
        priority_map)
    debug_data['2_classified_data'] = df_processed.copy()

    df_processed.sort_values(
        by=["Component", "Location Priority", "Batch Nr.1"], inplace=True)
    debug_data['3_sorted_by_priority'] = df_processed.copy()

    df_processed['Rank'] = df_processed.groupby('Component').cumcount() + 1
    all_priority_groups = []
    for component_code, group in df_processed.groupby("Component"):
        group_copy = group.copy()
        required_qty = group_copy["Quantity required"].iloc[0]

        # --- NEW: Logic for "To Be Prod." highlighting ---
        group_copy['To Be Prod.'] = False  # Default to False
        # Check if the highest-ranked item (Rank 1) is an "Unknown" location
        if not group_copy.empty and group_copy.iloc[0]['Location Priority'] == 99:
            group_copy.loc[group_copy.index[0], 'To Be Prod.'] = True

        group_copy['Cumulative Qty'] = group_copy['Available Quantity'].cumsum()
        first_pick_mask = group_copy['Cumulative Qty'] < required_qty
        first_priority_indices = group_copy[first_pick_mask].index.tolist()
        if first_pick_mask.sum() < len(group_copy):
            first_priority_indices.append(
                group_copy.index[first_pick_mask.sum()])

        group_copy['Assigned Priority'] = 'Leftovers'
        group_copy.loc[first_priority_indices,
                       'Assigned Priority'] = 'First Priority'

        remaining_rows = group_copy[group_copy['Assigned Priority'] == 'Leftovers'].copy(
        )
        if not remaining_rows.empty:
            def assign_by_rank(rank):
                if rank == len(first_priority_indices) + 1:
                    return 'Second Priority'
                if rank == len(first_priority_indices) + 2:
                    return 'Third Priority'
                return 'Leftovers'
            group_copy.loc[remaining_rows.index, 'Assigned Priority'] = remaining_rows['Rank'].apply(
                assign_by_rank)

        group_copy['Needs Highlighting'] = len(first_priority_indices) > 1
        all_priority_groups.append(group_copy)

    if not all_priority_groups:
        return product_info, {}, {}
    df_with_priorities = pd.concat(all_priority_groups)

    # Final filter: now we remove any 'Unknown' location that ISN'T a 'To Be Prod.' item
    df_with_priorities = df_with_priorities[~((df_with_priorities['Location Priority'] == 99) & (
        df_with_priorities['To Be Prod.'] == False))]

    debug_data['4_final_assignments'] = df_with_priorities.copy()

    priority_dfs_raw = {p_level: df_with_priorities[df_with_priorities['Assigned Priority'] == p_level] for p_level in [
        'First Priority', 'Second Priority', 'Third Priority', 'Leftovers']}
    priority_dfs_formatted = {name: format_output_df(
        df_raw) for name, df_raw in priority_dfs_raw.items() if not df_raw.empty}

    return product_info, priority_dfs_formatted  # , debug_data


def generate_pdf(product_info, priority_dfs, barcode_locations, file_configs, content_to_include):
    """
    Generates PDF(s) with the Color Code Legend and the corrected filtering logic
    to ensure "To Be Produced" items are always included.
    """
    generated_files = []

    # Define colors here for consistency
    ALLERGEN_COLOR = colors.lightblue
    EXPIRED_COLOR = colors.lightcoral
    EXPIRING_SOON_COLOR = colors.moccasin
    MULTI_PICK_COLOR = colors.yellow
    TO_BE_PRODUCED_COLOR = colors.lightgreen

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

        elements = []

        # 1. Main Title
        title_text = f"Production Ticket Information - {product_info['Production Ticket Nr']}"
        if total_files > 1:
            title_text += f" ({file_num} / {total_files})"
        elements.append(Paragraph(title_text, styles['h1']))
        elements.append(Spacer(1, 0.1*inch))

        # 2. Left Column: Production Info Table
        info_copy = {k: v for k, v in product_info.items() if k !=
                     "Production Ticket Nr"}
        info_data = [[key, str(value)] for key, value in info_copy.items()]
        info_table = Table(info_data, colWidths=[1.5*inch, 2.5*inch])
        info_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), FONT_BOLD),
            ('FONTNAME', (1, 0), (1, -1), FONT_NORMAL),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6)
        ]))

        # 3. Right Column: Color Code Legend Table
        legend_data = [
            ['Color', 'Meaning'],
            ['', 'Allergen Material'],
            ['', 'Expired Material'],
            ['', 'Expires Within 1 Month'],
            ['', 'Multi-location Pick'],
            ['', 'To Be Produced / Located']
        ]
        legend_table = Table(legend_data, colWidths=[0.5*inch, 2.2*inch])
        legend_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), FONT_BOLD),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BACKGROUND', (0, 1), (0, 1), ALLERGEN_COLOR),
            ('BACKGROUND', (0, 2), (0, 2), EXPIRED_COLOR),
            ('BACKGROUND', (0, 3), (0, 3), EXPIRING_SOON_COLOR),
            ('BACKGROUND', (0, 4), (0, 4), MULTI_PICK_COLOR),
            ('BACKGROUND', (0, 5), (0, 5), TO_BE_PRODUCED_COLOR),
        ]))

        # 4. Combine Info and Legend into a single parent table
        parent_table_data = [[info_table, legend_table]]
        parent_table = Table(parent_table_data, colWidths=[
                             4.2*inch, 2.9*inch], style=[('VALIGN', (0, 0), (-1, -1), 'TOP')])

        elements.append(parent_table)
        elements.append(Spacer(1, 0.2*inch))

        content_added_to_this_pdf, is_first_content_block = False, True
        for priority_name in content_to_include:
            if priority_name in priority_dfs:
                # --- FIXED LOGIC: Keep a row if its location is in the list OR if it's a "To Be Produced" item ---
                df_to_filter = priority_dfs[priority_name]
                df_output = df_to_filter[
                    df_to_filter['Location'].isin(locations_for_this_file) |
                    (df_to_filter['Location'] == 'To Be Prod.')
                ]

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
                                ['Expiry Status', 'Needs Highlighting', 'Is Allergen', 'To Be Prod.', 'Location Priority'])))
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
                                if original_row['To Be Prod.']:
                                    dynamic_styles.append(
                                        ('BACKGROUND', (0, i), (-1, i), TO_BE_PRODUCED_COLOR))
                                if original_row['Is Allergen']:
                                    dynamic_styles.append(
                                        ('BACKGROUND', (2, i), (2, i), ALLERGEN_COLOR))
                                if original_row['Expiry Status'] == 'Expired':
                                    dynamic_styles.append(
                                        ('BACKGROUND', (2, i), (2, i), EXPIRED_COLOR))
                                elif original_row['Expiry Status'] == 'Expiring Soon':
                                    dynamic_styles.append(
                                        ('BACKGROUND', (2, i), (2, i), EXPIRING_SOON_COLOR))
                                if original_row['Needs Highlighting']:
                                    dynamic_styles.append(
                                        ('BACKGROUND', (5, i), (6, i), MULTI_PICK_COLOR))
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
st.sidebar.markdown("---")
st.sidebar.write("Set Quantity Limits:")
max_tower_qty = st.sidebar.number_input(
    "Max Qty for Tower:", value=2.0, step=0.1, format="%.3f")
max_pour_drum_qty = st.sidebar.number_input(
    "Max Qty for Pour drum:", value=20.0, step=1.0, format="%.2f")
st.sidebar.markdown("---")
split_option = st.sidebar.radio(
    "PDF Splitting:", ("Single File", "Split into 2 Files", "Split into 3 Files"))
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
    "Generate barcodes for which locations?", LOCATION_ORDER, default=["Tower"])
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
        # with st.expander("Debug View: Processed Location Map"):
        #     st.caption("This is how the application has interpreted your location mapping file. Check the 'Location Type' and 'Is Allergen' columns to ensure all locations (Powder, Allergen, etc.) are being classified correctly.")
        #     st.dataframe(location_map)
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = xls.sheet_names[0]
        df_preview = pd.read_excel(xls, sheet_name=sheet_name)
        header_row_index = df_preview.apply(lambda row: row.astype(
            str).str.contains("Batch Nr").any(), axis=1).idxmax() + 1
        string_converters = {'Production Ticket Nr': str, 'Product Code': str, 'Batch Nr': str,
                             'Current date marked the beginning': str, 'Component': str, 'Batch Nr.1': str, 'DLUO': str}
        df = pd.read_excel(xls, sheet_name=sheet_name,
                           header=header_row_index, converters=string_converters)

        # debug - product_info, priority_dataframes, debug_dfs = process_data(df, location_map, max_tower_qty, max_pour_drum_qty)
        product_info, priority_dataframes = process_data(
            df, location_map, max_tower_qty, max_pour_drum_qty)

        st.header("Step 3: Review and Generate")
        st.subheader("Production Information")
        st.json(product_info, expanded=False)
        st.subheader("First Priority Picking List (Preview)")

        if 'First Priority' in priority_dataframes and not priority_dataframes['First Priority'].empty:
            preview_df = priority_dataframes['First Priority']

            # Define the columns we want to show the user in the preview
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

        # # --- NEW: Debugging Section ---
        # st.markdown("---")
        # st.header("Debug Information (Intermediate Steps)")

        # with st.expander("Step 1: Raw Data After Filtering"):
        #     st.caption(
        #         "This shows the rows from the production ticket that matched a valid location in your mapping file. Rows with (WIP) or (NC) are excluded.")
        #     if '1_filtered_ticket' in debug_dfs and not debug_dfs['1_filtered_ticket'].empty:
        #         st.dataframe(debug_dfs['1_filtered_ticket'])

        # with st.expander("Step 2: After Classification"):
        #     st.caption("This is the MOST IMPORTANT step. Check the 'Location Type' and 'Location Priority' columns. A lower priority number is better. Ensure Powder/Allergen locations are correctly typed and prioritized.")
        #     if '2_classified_data' in debug_dfs and not debug_dfs['2_classified_data'].empty:
        #         st.dataframe(debug_dfs['2_classified_data'])

        # with st.expander("Step 3: After Sorting by Priority"):
        #     st.caption("This shows the full list of all possible items, sorted by Component, then by the 'Location Priority'. The top items for each component will be picked first.")
        #     if '3_sorted_by_priority' in debug_dfs and not debug_dfs['3_sorted_by_priority'].empty:
        #         st.dataframe(debug_dfs['3_sorted_by_priority'])

        # with st.expander("Step 4: After Final Priority Assignment"):
        #     st.caption("This is the final result before formatting. Check the 'Assigned Priority' column to see which items were selected for 'First Priority' based on the quantity needed.")
        #     if '4_final_assignments' in debug_dfs and not debug_dfs['4_final_assignments'].empty:
        #         st.dataframe(debug_dfs['4_final_assignments'])

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error(
            "Please check that both files are correct and that all locations in the ticket exist in the mapping file.")
else:
    st.info(
        "Please upload both a Production Ticket and a Location Mapping File to begin.")
