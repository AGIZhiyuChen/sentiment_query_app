import json
import gradio as gr
import pandas as pd
import tempfile


import matplotlib.pyplot as plt
from typing import Optional, Union, List 
from plotting.plot import CustomLinePlot # adj lineplot not gradio one

from databricks_interactions.SentimentAggregator import SentAgg 

import traceback 
import openpyxl 

import logging, sys
                                                                                                                                                                                                 
import itertools     
import threading


import time                                                                                                                                                                                                      

# Control the names of the downloads to not be random
# tempfile._get_candidate_names = lambda: itertools.repeat('_tmp')   


import logging
import sys

from databricks.connect import DatabricksSession 

from fuzzywuzzy import fuzz
from difflib import SequenceMatcher

# 1. Get a logger instance. The name "APP" will be used by %(name)s
logger = logging.getLogger("APP")

# 4. Check if the logger already has handlers to prevent duplicate logs
#    This is a best practice, especially in environments like Jupyter
#    where cells can be run multiple times.
if not logger.hasHandlers():
    # Set the lowest-level of message this logger will handle
    logger.setLevel(logging.DEBUG)
    
    # 2. Create a handler to direct messages to the console (standard output)
    handler = logging.StreamHandler(sys.stdout)
    
    # 3. Create a formatter and set it for the handler
    #    This defines the final look of your log messages.
    fmt = "[%(asctime)s] [%(name)s] %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    
    formatter = logging.Formatter(fmt, datefmt=date_fmt)
    handler.setFormatter(formatter)
    
    # 5. Add the configured handler to the logger
    logger.addHandler(handler)





css = """
#button.green {background-color: #bae8e8}
#button.red {background-color: #f44336}
"""

base_param_df = pd.DataFrame(columns  =["Example Labels:", "Tariffs", "Inflation", "US Politics"],
                             data = [["Topic Filter Set #1:", "NA", "NA", "NA"],
                             ["Topic Filter Set #2:", "NA", "NA", "NA"]])


base_output_df = pd.DataFrame(columns  =["Example Labels:", "Tariffs", "Inflation", "US Politics"],
                             data = [["pd.Timestamp(2003,1,1)", "NA", "NA", "NA"],
                             ["pd.Timestamp(2003,1,2)", "NA", "NA", "NA"]])

try:
    with open("input_validation_data/description_to_rcsCode.json", "r") as jf:
        imported_topics_initial = json.load(jf)
        filter_choices_state = list(imported_topics_initial.keys())
    with open("input_validation_data/name_to_permid.json", "r") as jf:
        name_permid_import = json.load(jf)
        filter_choices_state += list(name_permid_import.keys())
        
        
except FileNotFoundError:
    logger.warning("WARNING: description_to_rcsCode.json not found. Using empty filter choices.")
    imported_topics_initial = {}
except json.JSONDecodeError:
    logger.warning("WARNING: description_to_rcsCode.json is not valid JSON. Using empty filter choices.")
    imported_topics_initial = {}
    

def fuzzy_match(query, options, max_results=50, threshold=0.1):
    if not query.strip():
        return options[:max_results]  # Return first 50 if no query
    
    query_lower = query.lower()
    matches = []
    
    # First pass: exact substring matches (fastest)
    for option in options:
        if query_lower in option.lower():
            matches.append((option, 1.0))  # High score for exact matches
            if len(matches) >= max_results * 2:  # Get extra for sorting
                break
    
    # If we don't have enough matches, do fuzzy matching
    if len(matches) < max_results:
        remaining_options = [opt for opt in options if not any(opt == match[0] for match, _ in matches)]
        
        for option in remaining_options[:min(1000, len(remaining_options))]:  # Limit fuzzy search scope
            ratio = SequenceMatcher(None, query_lower, option.lower()).ratio()
            if ratio > threshold:
                matches.append((option, ratio))
            
            if len(matches) >= max_results * 3:  # More candidates for better sorting
                break
    
    # Sort by score (descending) and return top results
    matches.sort(key=lambda x: x[1], reverse=True)
    return [match[0] for match in matches[:max_results]]
    
    
def update_dropdown_choices(f_dropdown_selection, search_query):
    start_time = time.time()
    
    if not search_query or len(search_query.strip()) < 1:
        # Return first 50 options if no search query
        filtered_options = filter_choices_state[:50]
    else:
        # Use fuzzy matching to filter options
        filtered_options = fuzzy_match(search_query, filter_choices_state, max_results=50)
    
    end_time = time.time()
    search_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    # Since we enforce string option matching

    filtered_options +=  f_dropdown_selection
    
    # Create new dropdown with filtered options
    new_dropdown = gr.Dropdown(
        choices=filtered_options,
        value=f_dropdown_selection,
        label=f"Filtered Options ({len(filtered_options)} shown, search took {search_time:.1f}ms)",
        interactive=True,
        allow_custom_value=False, # default defined anyways
        multiselect=True, 
    )
    
    return new_dropdown, gr.update(info = f"Time taken to filter topics {search_time} ms")



# global helpers
from contextlib import contextmanager
from databricks.connect import DatabricksSession

@contextmanager
def managed_databricks_session(connection_string: str):
    """
    A context manager to automatically create and stop a Databricks Spark session.
    """
    start_time = time.time()
    logger.info(f"INFO: started context spark manager")
    spark = None
    try:
        try:
          
            spark = DatabricksSession.builder.remote(connection_string).getOrCreate()
            yield spark  
        finally:
            logger.info(f"INFO: retrieval took {((time.time() - start_time) / 60):.2f} minutes")
            if spark:
                print("Closing Spark session via context manager...")
                spark.stop()
    except Exception as e:
        logger.warning(f"WARNING: Spark connection not working as expceted {traceback.format_exc()} \nretrieval stopped after {((time.time() - start_time) / 60):.2%} minutes")
                

def handle_dynamic_dropdown_change(new_values, label, index, current_params):
    """
    Callback for when a dynamic filter dropdown's value changes.
    Updates the params_state.
    """
    if label and label in current_params and 0 <= index < len(current_params[label]):
        current_params[label][index] = new_values
        
    if not new_values: # Check if new_values is empty list
        if label and label in current_params and 0 <= index < len(current_params[label]):
            # Potentially remove the entry if it becomes empty, or handle as needed.
            # The original code implies deletion if new_values is [], but that might be too aggressive.
            # For now, let's assume an empty list of filters is valid for a set.
            # If the intent was to delete the filter set if cleared, `handle_dynamic_remove` logic would be here.
            # The original del current_params[label][index] was if new_values == [].
            # Corrected to reflect this:
            current_params[label][index] = [] # Set to empty, rather than delete
            # If you want to delete the filter set if it becomes empty:
            # del current_params[label][index]
            # And then you'd need to re-render the UI potentially.
            # For safety, just setting to empty list.
            
    return current_params

def handle_dynamic_remove(label, index, current_params):
    """
    Callback for when a 'Remove' button for a filter set is clicked.
    Updates the params_state by removing the specified filter set.
    """
    if label and label in current_params and 0 <= index < len(current_params[label]):
        del current_params[label][index]
        gr.Info(f"Filter set #{index + 1} for '{label}' removed.")
    return current_params


# --- Main Gradio App ---

task_status = {}
with gr.Blocks(theme= gr.themes.Soft(), css=css) as demo:
    # --- States ---
    params_state = gr.State(value={})
    dropdown_labels_state = gr.State(value=set())
    #filter_choices_state = gr.State(value=imported_topics_initial)
    params_freq = gr.State(value = "Weekly (M-SUN)")
    # analysis_is_running = gr.State(value = False) # This state wasn't used actively for UI control, run_btn interactive state serves this
    result_data_state = gr.State(value =dict())
    running_logs = gr.State(value = "")
    #is_running = False #gr.State(value  = False)
    user_task_ids = gr.State(value = [])
    

    # Title
    gr.Markdown("# Topic Tracking Interface")
    gr.Markdown(
        """
        This interface presents a frontend for filtering a daily-updated stream of news.
        News content, tags, and sentiment scores are obtained by pooling Thomson Reuters News Analytics (TRNA) alongside the original articles.
        Each item is then embedded using a sentence embedder, enabling content search via retrieval-augmented generation (RAG).
        Once the filter configuration is defined, it is sent to the backend, which handles filtering and aggregating relevant news by
        combining RAG embeddings with TRNA pre-tags.
        """
    )
    gr.Markdown("## Filter Configuration")
    gr.Markdown(
        """
        ### How the filters work:
        - **Topic Labels**
          Each set of filters corresponds to an overarching topic label.
        - **Combining Filters:**
          - **“Horizontal” (OR):**
            Filters on the same row form an OR relationship—an item is selected if it meets *any* of those filters.
          - **“Vertical” (AND):**
            Filters stacked in a column form an AND relationship—an item is selected only if it satisfies *all* of those filters in sequence.
        """
    )
        


    with gr.Row(equal_height=True):
        curr_label_dropdown = gr.Dropdown(
            choices=[], value = "U.S. Tariffs", allow_custom_value=True,
            label="Current Filter Label", info="Select an existing label or type a new one and click 'Set/Add Label'.", scale=6
        )
        add_label_btn = gr.Button("Set/Add Label", scale=1)
        delete_label_btn = gr.Button("DELETE Label", scale=1)

    gr.Markdown("---")
    add_new_filter_set_btn = gr.Button("+ Add New Filter Set to Current Label")
    gr.Markdown(
                """
                ### Filter Sets for Selected Label:
                """
                )
    
    
    
    @gr.render(inputs=[curr_label_dropdown, params_state])
    def render_dynamic_filters_ui(selected_label, current_params):
        ui_components_to_render = []
        if not selected_label:
            ui_components_to_render.append(gr.Markdown("Select or create a label above to manage its filters."))
            return ui_components_to_render
        if selected_label not in current_params:
            # This state can happen if a label is typed but not yet added via "Set/Add Label"
            # Or if a label was deleted and the dropdown still holds its value briefly.
            ui_components_to_render.append(gr.Markdown(f"Label '{selected_label}' is not yet active or has been removed. Please 'Set/Add Label' or select an existing one."))
            return ui_components_to_render
            
        filter_sets_for_this_label = current_params.get(selected_label, [])
        if not filter_sets_for_this_label:
            ui_components_to_render.append(gr.Markdown(f"No filter sets defined for '{selected_label}'. Click '+ Add New Filter Set' above."))
            return ui_components_to_render

        for idx, filter_values_for_set in enumerate(filter_sets_for_this_label):
            with gr.Group() as filter_set_ui_group: # type: ignore
                with gr.Row(equal_height=True):
                    
                    filter_searchbox = gr.Textbox(value = '', 
                                                  placeholder = 'Write your Filter name: i.e. Tariffs, or Apple Inc', 
                                                  label= 'This is the Filter Finder that searches around 80000 Filters for your Filter using fuzzy string matching'
                                                  )
                    
                    filter_dropdown = gr.Dropdown(choices = list(filter_values_for_set), value = list(filter_values_for_set),
                                                  multiselect=True, label=f"Filter Set #{idx + 1}", scale=4)

                
                    remove_this_set_btn = gr.Button(f"DELETE", min_width=120, scale=0)
                    
                    
                filter_searchbox.change(fn=update_dropdown_choices, inputs=[filter_dropdown, filter_searchbox], outputs=[filter_dropdown, filter_searchbox])
                filter_dropdown.change(fn=handle_dynamic_dropdown_change, inputs=[filter_dropdown, gr.State(selected_label), gr.State(idx), params_state], outputs=[params_state], show_progress="hidden")
                remove_this_set_btn.click(fn=handle_dynamic_remove, inputs=[gr.State(selected_label), gr.State(idx), params_state], outputs=[params_state], show_progress="hidden")
            ui_components_to_render.append(filter_set_ui_group)
        return ui_components_to_render


    def append_label_action(new_label_text, current_params, current_labels_set):
        if not new_label_text or not new_label_text.strip():
            gr.Warning("Label cannot be empty!")
            return current_params, current_labels_set, gr.update(choices=sorted(list(current_labels_set)))
        new_label_text = new_label_text.strip()
        current_labels_set.add(new_label_text)
        if new_label_text not in current_params: current_params[new_label_text] = []
        updated_choices = sorted(list(current_labels_set))
        return current_params, current_labels_set, gr.update(choices=updated_choices, value=new_label_text)

    def delte_label_action(rm_label, current_params, current_labels_set):
        if not rm_label:
            gr.Info("No label selected to delete.")
            return current_params, current_labels_set, gr.update(choices=sorted(list(current_labels_set)))
        if rm_label not in current_params:
            gr.Info(f"Label '{rm_label}' does not exist in active parameters.")
            if rm_label in current_labels_set: # Exists in dropdown but not params (e.g. typed but not added)
                 current_labels_set.remove(rm_label)
            new_val = sorted(list(current_labels_set))[0] if current_labels_set else "" # Default to first or empty
            return current_params, current_labels_set, gr.update(choices=sorted(list(current_labels_set)), value=new_val)
        
        # Label exists in current_params
        gr.Warning(f"Deleting label '{rm_label}' and all its filter sets.")
        current_params.pop(rm_label)
        if rm_label in current_labels_set:
            current_labels_set.remove(rm_label)
        
        new_val = sorted(list(current_labels_set))[0] if current_labels_set else "" # Default to first or empty
        return current_params, current_labels_set, gr.update(choices=sorted(list(current_labels_set)), value=new_val)

    add_label_btn.click(fn=append_label_action, inputs=[curr_label_dropdown, params_state, dropdown_labels_state], outputs=[params_state, dropdown_labels_state, curr_label_dropdown])
    delete_label_btn.click(fn=delte_label_action, inputs=[curr_label_dropdown, params_state, dropdown_labels_state], outputs=[params_state, dropdown_labels_state, curr_label_dropdown])

    def add_new_filter_set_to_label_action(current_selected_label, current_params):
        if not current_selected_label:
            gr.Warning("Please select or set a label first...")
            return current_params
        if current_selected_label not in current_params:
            # This can happen if a label is typed in dropdown but "Set/Add Label" hasn't been clicked.
            # Initialize it implicitly.
            gr.Info(f"Label '{current_selected_label}' was not active. Initializing and adding a new filter set.")
            current_params[current_selected_label] = []
            # Note: dropdown_labels_state should also be updated if we auto-add here.
            # For consistency, it's better if append_label_action is always called first.
            # The current UI flow implies curr_label_dropdown's value should already be in params_state.
            # If not, it means the user hasn't pressed "Set/Add Label" for a new custom value.

        current_params[current_selected_label].append([]) # Add an empty list representing a new filter set
        gr.Info(f"Added a new empty filter set placeholder for '{current_selected_label}'.")
        return current_params

    add_new_filter_set_btn.click(fn=add_new_filter_set_to_label_action, inputs=[curr_label_dropdown, params_state], outputs=[params_state])
    
    def display_params_as_df(current_params_state: dict):
        if not current_params_state: return gr.update(value=base_param_df)
        processed_data = {}
        labels = list(current_params_state.keys())
        if not labels: return gr.update(value=base_param_df)
        
        max_len = 0
        for label in labels:
            # Filter sets; each set is a list of filter strings
            filter_sets_for_label = current_params_state.get(label, [])
            
            # String representation for each filter set
            str_filter_sets = []
            if not filter_sets_for_label: # No filter sets for this label
                str_filter_sets.append("No filter sets defined")
            else:
                for f_set in filter_sets_for_label:
                    if not f_set: # An empty filter set
                        str_filter_sets.append("Empty filter set (matches all if OR, or specific backend logic)")
                    else:
                        str_filter_sets.append(" AND ".join(map(str, f_set))) # Items within a set are ANDed
            
            processed_data[label] = str_filter_sets
            if len(str_filter_sets) > max_len:
                max_len = len(str_filter_sets)
        
        if max_len == 0: # No filter sets across all labels
            # Create a DataFrame indicating no filters for existing labels
            df_dict = {lbl: ["No filter sets defined"] for lbl in labels}
            max_len = 1 # For proper row indexing
        else:
            # Pad shorter lists with empty strings for DataFrame consistency
            df_dict = {lbl: (data + [""] * (max_len - len(data))) for lbl, data in processed_data.items()}

        df = pd.DataFrame(df_dict)
        # The problem statement implies "Topic Filter Set #" are rows, and labels are columns.
        # "Topic Labels:" should be the first column, acting as row headers.
        df["Topic Filter Set:"] = [f"Set #{i+1}" for i in range(max_len)]
        # Reorder columns to have "Topic Filter Set:" first
        df = df[["Topic Filter Set:"] + labels]
        # df.index is not needed if "Topic Filter Set:" is a column
        return gr.update(value=df)


    gr.Markdown("---")
    gr.Markdown("### Filter View")
    with gr.Tabs():
        with gr.TabItem("DataFrame View"):
            params_display_df = gr.DataFrame(value = base_param_df, label="Current Params (DataFrame View)")
        with gr.TabItem("JSON View"):
            params_display_json = gr.JSON(label="Current Params (JSON)")
        
    params_state.change(display_params_as_df, inputs=[params_state], outputs=[params_display_df])
    params_state.change(lambda p: p, inputs=[params_state], outputs=[params_display_json], show_progress="hidden")
    
    with gr.Row(equal_height=True):
        freq_btn_dd = gr.Dropdown(choices=["Monthly", "Weekly (M-SUN)", "Daily"], value=params_freq.value, multiselect=False,
                               label="Frequency Aggregation of Returned Data", info="Default is Weekly...", scale=3)
        run_btn = gr.Button("Fetch & Return Data", scale=1, elem_id="button", elem_classes=["green"])

    freq_btn_dd.change(fn=lambda v: v, inputs=[freq_btn_dd], outputs=[params_freq])
    
    def df_to_lineplot_data(df: pd.DataFrame, selected_cols=None): # Renamed to avoid conflict
        if df is None or df.empty or "Timestamp" not in df.columns: return None
        if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
            try: df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            except Exception as e:
                logger.error(f"ERROR converting Timestamp to datetime: {e}")
                return None
        
        all_num_cols = [col for col in df.columns if col != "Timestamp" and pd.api.types.is_numeric_dtype(df[col])]
        
        plot_cols = []
        if selected_cols:
            plot_cols = [c for c in (selected_cols if isinstance(selected_cols, list) else [selected_cols]) if c in all_num_cols]
        
        if not plot_cols: # If no valid selection or selection is empty, plot all numeric columns
            plot_cols = all_num_cols

        if not plot_cols: return None # No numeric columns to plot
        
        df_melted = pd.melt(df, id_vars=["Timestamp"], value_vars=plot_cols, var_name="SentScore Label", value_name="Score").sort_values(by="Timestamp")
        return df_melted
    
    def df_to_tableview(df : pd.DataFrame, key : Optional[str]):
        
        if key is None: 
            logging.debug(f"{key} not in fetched data dictionary {df.keys()}")
            
        else:
            df = df.get(key, base_output_df)
            
        
        if df is None or df.empty: return base_output_df
        # If df contains only an "Error" column
        if "Error" in df.columns and len(df.columns) == 1:
            return df # Display the error DataFrame as is
        if "Timestamp" not in df.columns :
            # If no Timestamp and not an error DF, it's unexpected data
            return pd.DataFrame({'Error': ["Unexpected data structure received. Timestamp column missing."]})
        
        # Standard case: reorder to put Timestamp first
        cols = df.columns.to_list()
        cols.remove("Timestamp")
        return df[["Timestamp"] + cols]

    def make_plot_sentiment(df : dict, key : str, selected_cols = None, plot_args : dict = {}):
        
        if df is None or key not in df.keys(): 
            logger.error(f"ERROR: occured when calling make_plot with key parameter : {key}")
            return None
        
        df_melted = df_to_lineplot_data(df.get(key), selected_cols) 
        if df_melted is None or df_melted.empty: return None
        # cutom line plot that actually has better dimension then the generic gradio lineplot
        if not plot_args:
            return CustomLinePlot(value=df_melted, title_font_size=28, axis_title_font_size=18, axis_label_font_size=14)
        else: 
            return CustomLinePlot(value=df_melted, title_font_size=28, axis_title_font_size=18, axis_label_font_size=14,
                                  **plot_args)
            

    def make_plot_counts(df : dict, key : str, selected_cols = None):
        
        if df is None or key not in df.keys(): 
            logger.error(f"ERROR: occured when calling make_plot with key parameter : {key}")
            return None
        
        df_melted = df_to_lineplot_data(df.get(key), selected_cols) 
        if df_melted is None or df_melted.empty: return None
        
        return CustomLinePlot(value=df_melted, title_font_size=28, axis_title_font_size=18, axis_label_font_size=14,
                                y_label = "News Count", title="News Counts associated with Topic")
            
    
    gr.Markdown("---")      
    gr.Markdown("## Results")
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            running_logs_display = gr.Textbox(label="Query Logs", value="", lines=15, interactive=False, max_lines=20)
            check_completion = gr.Button(value = "Check if Query is Complete")
        with gr.Column(scale=8):                                     
            with gr.Tabs():
                with gr.TabItem("Sentiment Chart", elem_id="sentiment-tab"):
                    scores_output_lineplot = gr.Plot() # Let Gradio decide the plot type based on output of make_plot
                    scores_dd = gr.Dropdown(choices=[], multiselect=True, label="Select Topics to Display")
                    
                with gr.TabItem("Topic Linked News Counts", elem_id="counts-tab"):
                    counts_output_lineplot = gr.Plot() 
                    counts_dd = gr.Dropdown(choices=[], multiselect=True, label="Select Topics to Display")
                    
                with gr.TabItem("Sentiment Table"):
                    score_output_table = gr.DataFrame(label="News Sentiment over Time", value=base_output_df, elem_id="scores_dataframe_output", row_count=(30, 'fixed'))
                
                with gr.TabItem("News Count Table"):
                    counts_output_table = gr.DataFrame(label="News Counts over Time", value=base_output_df, elem_id="counts_dataframe_output", row_count=(30, 'fixed'))
                
                #with gr.TabItem("Raw Aggregated Data Table"): pass # Placeholder
                with gr.TabItem("Download Data"):
                    with gr.Row():
                        
                        scores_download_excel_btn = gr.DownloadButton("Download Scores Table as .xlsx File", visible=True)
                        counts_download_excel_btn = gr.DownloadButton("Download Counts Table as .xlsx File", visible=True)
                        all_data_excel_btn = gr.DownloadButton("All Data as .xlsx File", visible=True)
                            
            # with gr.Row("Country Associated Counts"): pass
            # with gr.Row("Company Associated Counts"): pass

    # Note: The @run_btn.click decorator implicitly handles creating a generator from the function if it yields.
    @run_btn.click(inputs=[user_task_ids, params_state, params_freq, running_logs],
            outputs=[user_task_ids, run_btn, result_data_state, score_output_table, 
                        counts_output_table, running_logs_display],
            queue=True)
    def asynchronous_data_fetcher(user_task_ids_, current_query_params, current_params_freq_val, logs_state_value: str):
        current_accumulated_logs = logs_state_value if isinstance(logs_state_value, str) else ""
        current_accumulated_logs = f"{current_accumulated_logs}\n--- New Query Run ---\nStarting query process...".strip()

        # FIXED: Generate unique task ID for this operation
        import uuid
        task_id = str(uuid.uuid4())[:8]
        
        # FIXED: Initialize task status in global dict
        task_status[task_id] = {
            'status': 'running',
            'start_time': time.time(),
            'logs': current_accumulated_logs,
            'result': None,
            'error': None,
            'params': current_query_params
        }
        
        user_task_ids_ += [task_id]

        logger.info(f"New query run started with task id {task_id} out of {user_task_ids_}")
        yield {
            user_task_ids : gr.update(value = user_task_ids_),
            run_btn: gr.update(value=f"Running... (ID: {task_id})", interactive=False, elem_classes=["red"]),
            running_logs_display: gr.update(value=current_accumulated_logs)
        }

        ui_log_msg = "\nAttempting to initialize SentAgg API...".strip()
        current_accumulated_logs += ui_log_msg
        task_status[task_id]['logs'] = current_accumulated_logs  # FIXED: Update task status
        logger.info("Attempting to initialize SentAgg API...")
        yield {running_logs_display: gr.update(value=current_accumulated_logs)}

        try:
            data_api = SentAgg()
            ui_log_msg = "\nSentAgg API initialized successfully.".strip()
            current_accumulated_logs += ui_log_msg
            task_status[task_id]['logs'] = current_accumulated_logs  # FIXED: Update task status
            logger.info("SentAgg API initialized successfully.")
            yield {running_logs_display: gr.update(value=current_accumulated_logs)}
        except Exception as e:
            error_details = traceback.format_exc()
            error_message = f"FATAL ERROR: Failed to initialize SentAgg API.\n\nDetails:\n{error_details}"
            
            logger.error(error_message)
            current_accumulated_logs += f"\n{error_message}"
            
            # FIXED: Update task status with error
            task_status[task_id]['status'] = 'failed'
            task_status[task_id]['error'] = error_message
            task_status[task_id]['logs'] = current_accumulated_logs
            
            yield {
                run_btn: gr.update(value="Fetch & Return Data", interactive=True, elem_classes=["green"]),
                running_logs_display: gr.update(value=current_accumulated_logs)
            }
            return

        ui_log_msg = "\nInitializing query_news generator from SentAgg...".strip()
        current_accumulated_logs += ui_log_msg
        task_status[task_id]['logs'] = current_accumulated_logs  # FIXED: Update task status
        logger.info("Initializing query_news generator from SentAgg...")
        yield {running_logs_display: gr.update(value=current_accumulated_logs)}

        # FIXED: Proper background task function that updates global task_status
        def background_task():
            try:
                with managed_databricks_session() as spark:
                    try:
                        gen_instance = data_api.query_news(current_query_params, current_params_freq_val, spark)
                        
                        # Update task status
                        task_status[task_id]['logs'] += "\nquery_news generator created. Starting iteration..."
                        logger.info("query_news generator created. Starting iteration...")
                        
                    except Exception as e:
                        error_details = traceback.format_exc()
                        error_message = f"FATAL ERROR: Failed to call SentAgg.query_news.\n\nDetails:\n{error_details}"
                        
                        logger.error(error_message)
                        task_status[task_id]['status'] = 'failed'
                        task_status[task_id]['error'] = error_message
                        task_status[task_id]['logs'] += f"\n{error_message}"
                        return

                    # FIXED: Direct generator iteration in background thread
                    fetched_data = None
                    try:
                        while True:
                            try:
                                intermediate_item = next(gen_instance)

                                if isinstance(intermediate_item, str):
                                    ui_log_msg = f"\n[SentAgg Log] {intermediate_item}".strip()
                                    task_status[task_id]['logs'] += ui_log_msg
                                    logger.info(f"[SentAgg Log] {intermediate_item}")

                            except StopIteration as e:
                                fetched_data = e.value if isinstance(e.value, dict) else None
                                task_status[task_id]['logs'] += "\n[Generator Loop] Generator finished. Final data received."
                                logger.info("Generator finished successfully.")
                                break

                            except Exception as gen_execution_error:
                                error_details = traceback.format_exc()
                                error_message = f"ERROR during generator execution: {gen_execution_error}\n\nDetails:\n{error_details}"
                                
                                logger.error(error_message)
                                task_status[task_id]['status'] = 'failed'
                                task_status[task_id]['error'] = error_message
                                task_status[task_id]['logs'] += f"\n{error_message}"
                                fetched_data = {"Sentiment Scores": pd.DataFrame({"Error": [f"Error during generation: {gen_execution_error}"]})}
                                break

                    except Exception as e: 
                        fetched_data = {"Sentiment Scores": pd.DataFrame({"Error": [f"Outer exception during generation: {e}"]})}
                        task_status[task_id]['status'] = 'failed'
                        task_status[task_id]['error'] = str(e)

                # FIXED: Mark task as completed and store result
                task_status[task_id]['status'] = 'completed'
                task_status[task_id]['result'] = fetched_data
                task_status[task_id]['end_time'] = time.time()
                logger.info("Query processing complete in background thread.")
                
            except Exception as e:
                # FIXED: Handle any unexpected errors in background task
                task_status[task_id]['status'] = 'failed'
                task_status[task_id]['error'] = str(e)
                logger.error(f"Background task failed: {e}")

        # FIXED: Start the background task in a daemon thread (like your example)
        thread = threading.Thread(target=background_task)
        thread.daemon = True
        thread.start()

        # FIXED: Return immediately with task ID and status message
        final_message = f"Task started in background (ID: {task_id})\n\nThe query is running asynchronously. Check the logs or result tables periodically.\nThis avoids the 60-second timeout limitation."
        current_accumulated_logs += f"\n{final_message}"
        
        yield {
            run_btn: gr.update(value="Fetch & Return Data", interactive=True, elem_classes=["green"]),
            running_logs_display: gr.update(value=current_accumulated_logs)
        }


    # FIXED: Add a new function to check task status
    def check_task_status(user_task_ids, running_logs_state_value):
        """Return the current status of all background tasks."""
        current_accumulated_logs = running_logs_state_value if isinstance(running_logs_state_value, str) else ""
        
        user_task_history = {task_id : task_status.get(task_id) for task_id in user_task_ids}
        
        if not task_status:
            current_accumulated_logs += "\nNo background tasks found."
            return [
                gr.update(value="Fetch & Return Data", interactive=True, elem_classes=["green"]),
                gr.update(value=None),
                gr.update(value=current_accumulated_logs),
                pd.DataFrame(),
                pd.DataFrame()
            ]
        
        status_text = "\n\n## Background Task Status\n\n"
        
        # Check if any tasks are completed and get the most recent result
        curr_task = task_status.get(user_task_ids[-1])
        completed_task  = curr_task if curr_task.get('status', None) == 'completed' else None
        result = None 
        
        for task_id, info in user_task_history.items():
            status_text += f"**Task {task_id}:**\n"
            for f_key in info['params'].keys():
                status_text += f"{f_key}\n"
            
            if info['status'] == 'running':
                elapsed = time.time() - info['start_time']
                status_text += f"- Running... (Elapsed {elapsed:.0f} seconds)\n"
                
            elif info['status'] == 'completed':
                if 'end_time' in info:
                    duration = info['end_time'] - info['start_time']
                    status_text += f"- Completed (Duration {duration:.0f} seconds)\n"
                else:
                    status_text += f"- Completed\n"
                    
            elif info['status'] == 'failed':
                status_text += f"- Failed: {info['error']}\n"
            
            status_text += "\n"
        
        current_accumulated_logs += status_text
        
        # Get the most recent completed result if available
        if completed_task:
            result = completed_task['result']
            current_accumulated_logs = completed_task['logs'] + status_text
            
            return [
                gr.update(value="Fetch & Return Data", interactive=True, elem_classes=["green"]),
                result,
                current_accumulated_logs,
                df_to_tableview(result, "Sentiment Scores") if result else pd.DataFrame(),
                df_to_tableview(result, "News Counts") if result else pd.DataFrame()
            ]
        else:
            return [
                gr.update(value="Fetch & Return Data", interactive=True, elem_classes=["green"]),
                gr.update(value=None),
                current_accumulated_logs,
                pd.DataFrame(),
                pd.DataFrame()
            ]
        
    check_completion.click(fn=check_task_status, 
                        inputs=[user_task_ids, running_logs], 
                        outputs=[run_btn, result_data_state, running_logs_display,
                                    score_output_table, counts_output_table])

    
    result_data_state.change(fn=make_plot_sentiment, inputs=[result_data_state, gr.State(value = "Sentiment Scores"), scores_dd], outputs=scores_output_lineplot)
    result_data_state.change(fn=make_plot_counts, inputs=[result_data_state, gr.State(value = "News Counts"), counts_dd], outputs=counts_output_lineplot)
    
    def update_output_dd_choices(result_state: dict, key: str) -> dict:
        logger.info("INFO: updating output dropdown choices")
        out = gr.update(
            choices=(
                
                [c for c in result_state.get(key, pd.DataFrame()).columns if c != 'Timestamp']
                # Ensure we have a valid dictionary and it's not an error response
                if isinstance(result_state, dict) and key in result_state and not result_state[key].empty and "Error" not in result_state[key].columns
                else []
            ),
            value = []
        )
        return out
        
    
    result_data_state.change(
        fn=update_output_dd_choices,
        inputs=[result_data_state, gr.State( value = "Sentiment Scores") ],
        outputs=scores_dd
        )

    result_data_state.change(
        fn=update_output_dd_choices,
        inputs=[result_data_state, gr.State( value = "News Counts") ],
        outputs=counts_dd
        )
    
     
    scores_dd.change(fn=make_plot_sentiment, inputs=[result_data_state, gr.State(value = "Sentiment Scores"), scores_dd, 
                                           gr.State(value = {})], outputs=scores_output_lineplot)
    counts_dd.change(fn=make_plot_counts, inputs=[result_data_state, gr.State(value = "News Counts"), counts_dd,
                                           gr.State(value = {"y_label": "Number of News Articles", 
                                                             "title": "Number of Articles matched with selected Topic"})], outputs=counts_output_lineplot)
    
    def create_and_serve_excel(df: pd.DataFrame, key: str) -> Optional[str]:
        logger.info(f"INFO: serving excel for {key}")
        if key not in df.keys():
            logger.error(f"ERROR: create_serve_excel - key {key} is invalid")
            return pd.DataFrame()
        
        
        df = df.get(key, pd.DataFrame())
        
        if df is not None:
            if "Error" in df.columns and len(df.columns) == 1:
                logger.warning(f"WARNING: create_serve_excel error or no values in dataframe for key {key}")

        if df is None or df.empty or ("Error" in df.columns and len(df.columns) == 1):
            logger.warning(f"WARNING: create_serve_excel {key} returns empty df or None")
            return None # Critical: Return None to DownloadButton if no file

        try:
            with tempfile.NamedTemporaryFile(prefix = key, suffix=".xlsx", delete=False) as tmpfile:
                tmp_path = tmpfile.name

            cols_to_write = df.columns.tolist()
            if "Timestamp" in cols_to_write: # Ensure Timestamp is first if it exists
                cols_to_write.remove("Timestamp")
                cols_to_write = ["Timestamp"] + cols_to_write
            
            df_to_save = df[cols_to_write].copy() # Work on a copy

            # Handle timezone-aware datetimes for openpyxl compatibility
            if "Timestamp" in df_to_save.columns and pd.api.types.is_datetime64_any_dtype(df_to_save["Timestamp"]):
                if df_to_save["Timestamp"].dt.tz is not None:
                    df_to_save["Timestamp"] = df_to_save["Timestamp"].dt.tz_localize(None)
            
            logger.info(f"Attempting to write DataFrame to Excel file: {tmp_path}")
            df_to_save.to_excel(tmp_path, index=False, engine="openpyxl")
            logger.info(f"Successfully wrote DataFrame to Excel file: {tmp_path}")
            
            return tmp_path # Return the path for Gradio's DownloadButton
            
        except Exception as e:
            gr.Error(f"An error occurred while creating the Excel file: {str(e)}")
            logger.error(f"ERROR: full traceback for Excel creation error:\n{traceback.format_exc()}")
            return None

    scores_download_excel_btn.click(
        fn=create_and_serve_excel,
        inputs=[result_data_state, gr.State(value = "Sentiment Scores")],
        outputs=[scores_download_excel_btn] 
    )

    counts_download_excel_btn.click(
        fn=create_and_serve_excel,
        inputs=[result_data_state, gr.State(value = "News Counts")],
        outputs=[counts_download_excel_btn] 
    )
    
    all_data_excel_btn.click(
        fn=create_and_serve_excel,
        inputs=[result_data_state, gr.State(value = "All Agg Data")],
        outputs=[all_data_excel_btn] 
    )
    
    def load_initial_app_data(s_params, s_labels_set, current_logs_val):
        current_keys_from_params = set(s_params.keys())
        updated_labels_set = s_labels_set.union(current_keys_from_params)
        for lbl in updated_labels_set: s_params.setdefault(lbl, []) 
        
        sorted_choices_for_dropdown = sorted(list(updated_labels_set))
        # Set a default value for curr_label_dropdown if choices exist, otherwise empty or a placeholder
        default_label_val = sorted_choices_for_dropdown[0] if sorted_choices_for_dropdown else "U.S. Tariffs" # Fallback to original default
        if not sorted_choices_for_dropdown and "U.S. Tariffs" not in updated_labels_set: # If U.S. Tariffs was the default but no params yet
            updated_labels_set.add("U.S. Tariffs")
            s_params["U.S. Tariffs"] = []
            sorted_choices_for_dropdown = ["U.S. Tariffs"]


        initial_log_message = "Application loaded. Ready."
        updated_logs = f"{current_logs_val}\n{initial_log_message}".strip() if current_logs_val else initial_log_message
        
        initial_scores_plot = make_plot_sentiment({"Sentiment Scores" : pd.DataFrame()}, "Sentiment Scores") # Make an empty plot
        initial_counts_plot = make_plot_counts({"News Counts" : pd.DataFrame()}, "News Counts") # Make an empty plot
        init_s_or_c_dd = [] # No columns to choose from initially

        return (
            s_params, 
            updated_labels_set, 
            gr.update(choices=sorted_choices_for_dropdown, value=default_label_val), 
            updated_logs, 
            initial_scores_plot, 
            initial_counts_plot,
            gr.update(choices=init_s_or_c_dd, value=[]), # Value must be a list for multiselect
            gr.update(choices=init_s_or_c_dd, value=[]) # Value must be a list for multiselect
        )

    demo.load(
        fn=load_initial_app_data,
        inputs=[params_state, dropdown_labels_state, running_logs],
        outputs=[params_state, dropdown_labels_state, curr_label_dropdown, running_logs, 
                 scores_output_lineplot, counts_output_lineplot, scores_dd, counts_dd]
    )

if __name__ == "__main__":
    
    demo.queue()

    demo.launch(
        server_name="0.0.0.0",  
        share=False
    )


