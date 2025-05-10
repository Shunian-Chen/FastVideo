import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd # Import pandas for easier data handling
from tqdm import tqdm

def load_face_bin_mapping(metadata_file_path):
    """
    Loads the mapping from sample_id to face_percent_bin from the metadata file.
    Assumes the metadata file is a JSON list of objects.

    Args:
        metadata_file_path (str): Path to the videos2caption_cleaned.json file.

    Returns:
        dict: A dictionary mapping sample_id (str) to face_percent_bin (int).
              Returns an empty dict if the file cannot be processed.
    """
    mapping = {}
    try:
        with open(metadata_file_path, 'r') as f:
            # Updated based on user's change: Assume it's a JSON list, not JSON Lines
            data = json.load(f)
            for item in data:
                if 'latent_path' in item and 'face_percent_bin' in item:
                    # Derive sample_id from latent_path (remove .pt)
                    sample_id = item['latent_path'].replace('.pt', '')
                    mapping[sample_id] = item['face_percent_bin']
                else:
                     print(f"Warning: Skipping item due to missing keys in metadata file: {item}")
        print(f"Successfully loaded face bin mapping for {len(mapping)} samples.")
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_file_path}")
        return {}
    except json.JSONDecodeError:
         print(f"Error: Invalid JSON format in metadata file {metadata_file_path}")
         return {}
    except Exception as e:
        print(f"Error reading metadata file {metadata_file_path}: {e}")
        return {}
    return mapping

def load_and_optionally_merge_json_data(directory_path, merged_file_path, force_recreate_merged=False):
    """
    Loads loss data. If force_recreate_merged is True or the merged_file_path
    does not exist, it reads all individual JSON files from directory_path,
    merges them, saves to merged_file_path, and returns the data.
    Otherwise, it loads data directly from merged_file_path.

    Args:
        directory_path (str): Path to the directory containing individual JSON loss files.
        merged_file_path (str): Path to the (to be created or existing) merged JSON file.
        force_recreate_merged (bool): If True, always recreates the merged file.

    Returns:
        list: A list of dictionaries, where each dictionary is the content of a JSON file.
              Returns an empty list if no data can be loaded or created.
    """
    if not force_recreate_merged and os.path.exists(merged_file_path):
        print(f"Loading pre-merged loss data from: {merged_file_path}")
        try:
            with open(merged_file_path, 'r') as f:
                all_loss_data = json.load(f)
            print(f"Successfully loaded {len(all_loss_data)} records from merged file.")
            return all_loss_data
        except Exception as e:
            print(f"Error loading merged file {merged_file_path}: {e}. Will attempt to recreate.")
            # Fall through to recreate logic

    print(f"Reading individual JSON files from: {directory_path} to create/update merged file: {merged_file_path}")
    json_files = glob.glob(os.path.join(directory_path, '*.json'))
    if not json_files:
        print(f"No JSON files found in {directory_path}")
        return []

    all_loss_data = []
    for file_path in tqdm(json_files, desc="Reading and merging JSON files"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict): # Basic check
                    all_loss_data.append(data)
                else:
                    print(f"Warning: Content of {os.path.basename(file_path)} is not a dictionary. Skipping.")
        except json.JSONDecodeError:
            print(f"Skipping {os.path.basename(file_path)} due to Invalid JSON format.")
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")

    if all_loss_data:
        print(f"Successfully read {len(all_loss_data)} individual JSON files.")
        try:
            # Ensure the directory for the merged file exists
            output_dir_for_merged_file = os.path.dirname(merged_file_path)
            if output_dir_for_merged_file: # Create directory only if it's not empty (i.e. not current dir)
                 os.makedirs(output_dir_for_merged_file, exist_ok=True)

            with open(merged_file_path, 'w') as f:
                json.dump(all_loss_data, f, indent=4) # Save with indent for readability
            print(f"Merged data saved to {merged_file_path}")
        except Exception as e:
            print(f"Error saving merged data to {merged_file_path}: {e}")
    else:
        print("No data collected from individual JSON files.")

    return all_loss_data

def analyze_average_losses(all_loss_data, output_plot_dir):
    """
    Analyzes loss values from the provided data, calculates overall averages,
    and plots them as a bar chart.

    Args:
        all_loss_data (list): A list of dictionaries, each representing a loaded JSON loss file.
        output_plot_dir (str): The directory where the plot will be saved.
    """
    if not all_loss_data:
        print("No loss data provided for average loss analysis.")
        return

    total_losses = {'background_loss': 0.0, 'face_loss': 0.0, 'lip_loss': 0.0}
    file_count = 0

    for data in tqdm(all_loss_data, desc="Calculating overall average losses"):
        if 'background_loss' in data and 'face_loss' in data and 'lip_loss' in data:
            total_losses['background_loss'] += data['background_loss']
            total_losses['face_loss'] += data['face_loss']
            total_losses['lip_loss'] += data['lip_loss']
            file_count += 1
        # else:
            # Optional: print message for items missing keys, but might be verbose
            # sample_id_info = f" (sample_id: {data.get('sample_id', 'N/A')})" if isinstance(data, dict) else ""
            # print(f"Skipping item{sample_id_info} for average calc: Missing required loss keys.")

    if file_count == 0:
        print("No valid JSON files with required loss keys found for average calculation.")
        return

    average_losses = {k: v / file_count for k, v in total_losses.items()}

    print("\nOverall Average Losses:")
    for key, value in average_losses.items():
        print(f"  {key}: {value:.4f}")
    # Print the total count used for this overall average
    print(f"Total valid samples considered for overall average: {file_count}")

    # Plotting (Bar chart)
    labels = list(average_losses.keys())
    values = list(average_losses.values())

    plt.figure(figsize=(8, 6)) # Create a new figure for the bar chart
    bars = plt.bar(labels, values, color=['blue', 'green', 'red'])
    plt.ylabel('Average Loss')
    plt.title('Overall Average Loss Comparison by Region')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center')

    # Save the plot
    os.makedirs(output_plot_dir, exist_ok=True) # Ensure output directory exists
    plot_filename = os.path.join(output_plot_dir, 'overall_average_loss_comparison.png')
    plt.savefig(plot_filename)
    print(f"Bar chart saved to {plot_filename}")
    plt.close() # Close the figure

def analyze_losses_by_step(all_loss_data, output_plot_dir):
    """
    Analyzes loss values from the provided data, groups by step, calculates average loss
    and counts per step for each region, and plots line graphs.

    Args:
        all_loss_data (list): A list of dictionaries, each representing a loaded JSON loss file.
        output_plot_dir (str): The directory where the plot will be saved.
    """
    if not all_loss_data:
        print("\nNo loss data provided for step analysis.")
        return

    processed_data_for_df = []

    for data in tqdm(all_loss_data, desc="Processing data for step analysis"):
        if all(k in data for k in ['step', 'background_loss', 'face_loss', 'lip_loss']):
            processed_data_for_df.append({
                'step': data['step'],
                'background_loss': data['background_loss'],
                'face_loss': data['face_loss'],
                'lip_loss': data['lip_loss']
            })
        # else:
            # sample_id_info = f" (sample_id: {data.get('sample_id', 'N/A')})" if isinstance(data, dict) else ""
            # print(f"Skipping item{sample_id_info} for step analysis: Missing required keys (step or loss types).")

    if not processed_data_for_df:
        print("No valid data found in JSON files for step analysis.")
        return

    df = pd.DataFrame(processed_data_for_df)

    # Group by step and calculate both mean and count
    grouped = df.groupby('step')
    step_stats = grouped.agg(
        background_loss_avg=('background_loss', 'mean'),
        face_loss_avg=('face_loss', 'mean'),
        lip_loss_avg=('lip_loss', 'mean'),
        count=('step', 'size')  # Get count for each step group
    ).reset_index()

    # Sort by step for printing and plotting
    step_stats = step_stats.sort_values(by='step')

    # print("\nAverage Losses and Counts by Step:")
    # print(step_stats.to_string())

    # Define a window for rolling average
    rolling_window = 5  # You can adjust this window size
    if len(step_stats) >= rolling_window:
        step_stats['background_loss_avg_smooth'] = step_stats['background_loss_avg'].rolling(window=rolling_window, center=True, min_periods=1).mean()
        step_stats['face_loss_avg_smooth'] = step_stats['face_loss_avg'].rolling(window=rolling_window, center=True, min_periods=1).mean()
        step_stats['lip_loss_avg_smooth'] = step_stats['lip_loss_avg'].rolling(window=rolling_window, center=True, min_periods=1).mean()
    else:
        # Not enough data points for rolling average, use original data
        step_stats['background_loss_avg_smooth'] = step_stats['background_loss_avg']
        step_stats['face_loss_avg_smooth'] = step_stats['face_loss_avg']
        step_stats['lip_loss_avg_smooth'] = step_stats['lip_loss_avg']

    # Plotting line graphs (using the new column names)
    plt.rc('lines', markersize=3) # 减小后续绘图中标记点的大小
    plt.figure(figsize=(12, 7))
    
    plt.plot(step_stats['step'], step_stats['background_loss_avg_smooth'], marker='o', linestyle='-', label='Background Loss Avg (Smoothed)')
    plt.plot(step_stats['step'], step_stats['face_loss_avg_smooth'], marker='s', linestyle='--', label='Face Loss Avg (Smoothed)')
    plt.plot(step_stats['step'], step_stats['lip_loss_avg_smooth'], marker='^', linestyle=':', label='Lip Loss Avg (Smoothed)')

    # Optionally, plot count on a secondary y-axis if desired
    # ax2 = plt.gca().twinx()
    # ax2.plot(step_stats['step'], step_stats['count'], color='grey', marker='.', linestyle='-', label='Sample Count')
    # ax2.set_ylabel('Sample Count', color='grey')
    # ax2.tick_params(axis='y', labelcolor='grey')
    # plt.gca().legend(loc='upper left') # Adjust legend location if using secondary axis
    # ax2.legend(loc='upper right')

    plt.xlabel('Step')
    plt.ylabel('Average Loss')
    plt.title('Average Loss per Region over Steps (with Sample Counts)')
    plt.legend() # Show legend for loss lines
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(output_plot_dir, exist_ok=True) # Ensure output directory exists
    plot_filename = os.path.join(output_plot_dir, 'average_loss_by_step_line.png')
    plt.savefig(plot_filename)
    print(f"\nLine plot by step saved to {plot_filename}")
    plt.close() # Close the figure

def analyze_losses_by_face_bin(all_loss_data, face_bin_mapping, output_plot_dir):
    """
    Analyzes losses based on face percentage bins using pre-loaded data,
    calculates averages and counts, and plots results.

    Args:
        all_loss_data (list): A list of dictionaries, each representing a loaded JSON loss file.
        face_bin_mapping (dict): Dictionary mapping sample_id to face_percent_bin.
        output_plot_dir (str): The directory where the plot will be saved.
    """
    if not all_loss_data:
        print(f"\nNo loss data provided for bin analysis")
        return
    if not face_bin_mapping:
        print("\nFace bin mapping is empty. Cannot perform analysis by bin.")
        return

    binned_data = []
    skipped_count = 0

    for data in tqdm(all_loss_data, desc="Processing data for face bin analysis"):
        sample_id = data.get('sample_id')

        if not sample_id:
            # print(f"Skipping an item: Missing 'sample_id'.")
            skipped_count += 1
            continue

        face_bin = face_bin_mapping.get(sample_id)
        if face_bin is None:
            skipped_count += 1
            continue

        if all(k in data for k in ['background_loss', 'face_loss', 'lip_loss']):
            binned_data.append({
                'face_percent_bin': face_bin,
                'background_loss': data['background_loss'],
                'face_loss': data['face_loss'],
                'lip_loss': data['lip_loss']
            })
        else:
            # print(f"Skipping {os.path.basename(file_path)}: Missing required loss keys.")
            skipped_count += 1

    if not binned_data:
        print("No valid data found for bin analysis after matching samples.")
        return

    if skipped_count > 0:
        print(f"\nNote: Skipped {skipped_count} loss files during bin analysis due to missing sample_id, no matching bin, or other errors.")

    df = pd.DataFrame(binned_data)

    # Group by face bin and calculate both mean and count
    grouped = df.groupby('face_percent_bin')
    bin_stats = grouped.agg(
        background_loss_avg=('background_loss', 'mean'),
        face_loss_avg=('face_loss', 'mean'),
        lip_loss_avg=('lip_loss', 'mean'),
        count=('face_percent_bin', 'size')
    ).reset_index()

    print("\nAverage Losses and Counts by Face Percent Bin:")
    print(bin_stats.to_string())

    # Plotting average losses by bin
    if not bin_stats.empty:
        # Set bin as index for easier plotting of multiple bars per category
        bin_stats_plot = bin_stats.set_index('face_percent_bin')

        fig, ax1 = plt.subplots(figsize=(12, 7))

        # Plot bars for average losses
        bin_stats_plot[['background_loss_avg', 'face_loss_avg', 'lip_loss_avg']].plot(kind='bar', ax=ax1, position=0.9, width=0.4)
        ax1.set_title('Average Loss and Sample Count by Face Percent Bin')
        ax1.set_xlabel('Face Percent Bin')
        ax1.set_ylabel('Average Loss', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.tick_params(axis='x', rotation=0)
        ax1.legend(title='Avg Loss Type', loc='upper left')

        # Create a secondary y-axis for the count
        ax2 = ax1.twinx()
        # Plot count as a line or points on secondary axis
        bar_width = 0.4 # Approximate width from the loss bars
        # Offset the count points slightly to align better visually if needed
        ax2.plot(ax1.get_xticks(), bin_stats_plot['count'], color='grey', marker='o', linestyle='--', label='Sample Count')
        ax2.set_ylabel('Sample Count', color='grey')
        ax2.tick_params(axis='y', labelcolor='grey')
        # Ensure count axis starts near zero or min count
        ax2.set_ylim(bottom=0)
        ax2.legend(loc='upper right')

        fig.tight_layout() # Adjust layout to prevent overlap
        os.makedirs(output_plot_dir, exist_ok=True) # Ensure output directory exists
        plot_filename = os.path.join(output_plot_dir, 'average_loss_count_by_face_bin.png')
        plt.savefig(plot_filename)
        print(f"\nCombined bar/line chart for losses and counts by face bin saved to {plot_filename}")
        plt.close(fig) # Close the figure

if __name__ == '__main__':
    # Define paths
    loss_dir = '/data/nas/yexin/workspace/shunian/model_training/FastVideo/data/outputs/Hunyuan-Audio-Finetune-Hunyuan-ai2v-49frames-200_hour_480p_49frames_eng-0503-debug-yexin3/sample_losses'
    metadata_file = '/data/nas/yexin/workspace/shunian/model_training/FastVideo/data/200_hour_480p_49frames_eng/videos2caption_cleaned.json'
    
    # Determine the parent directory of loss_dir for saving merged data and plots
    # This assumes loss_dir is a valid path and has a parent.
    output_parent_dir = os.path.dirname(loss_dir)
    if not output_parent_dir: # Handle cases like loss_dir = "sample_losses" (relative path in current dir)
        output_parent_dir = "." # Save merged file/plots in the current directory

    merged_data_filename = "merged_loss_data.json"
    merged_data_filepath = os.path.join(output_parent_dir, merged_data_filename)

    # Load face bin mapping first
    face_bin_map = load_face_bin_mapping(metadata_file)
    # print(f"Face bin mapping: {face_bin_map}")

    # --- Data Loading ---
    # Set force_recreate_merged to True if you always want to re-read individual files
    # and overwrite the merged_loss_data.json file.
    # Otherwise, if merged_loss_data.json exists, it will be loaded directly.
    force_recreate = True #显式要求时修改为True
    
    all_json_data = load_and_optionally_merge_json_data(
        loss_dir, 
        merged_data_filepath, 
        force_recreate_merged=force_recreate
    )

    if not all_json_data:
        print("No data loaded. Exiting analysis.")
    else:
        # Run the analyses using the loaded data
        # analyze_average_losses(all_json_data, output_parent_dir) # Overall average bar chart
        analyze_losses_by_step(all_json_data, output_parent_dir) # Loss by step line chart
        # analyze_losses_by_face_bin(all_json_data, face_bin_map, output_parent_dir) # Loss by face bin analysis

        print("\nAnalysis complete. Plots saved.")
        # plt.show() # Call once if interactive display is needed