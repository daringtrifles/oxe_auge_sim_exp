#!/usr/bin/env python3
"""
Script to categorize experiments in evaluation results CSV.
Adds a 'Category' column based on experiment naming patterns.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
from typing import Dict, List, Optional, Tuple

plt.style.use('default')  # Start with clean slate
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'font.size': 25,
    'axes.titlesize': 25,
    'axes.labelsize': 25,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 30,
    
    # High-quality rendering
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    
    # Professional styling
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,  # Disable global grid, enable selectively per plot
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
    'grid.linestyle': '--',
    
    # Better spacing
    'figure.autolayout': True,
    'axes.axisbelow': True,
})

def categorize_task_name(exp_name: str) -> str:
    name = exp_name.lower()
    if 'can' in name:
        return "can"
    elif 'square' in name:
        return "square"
    elif 'two_piece_assembly' in name:
        return "two_piece_assembly"
    elif 'lift' in name:
        return "lift"
    elif 'stack' in name:
        return "stack"
    
def categorize_experiment_standard_mode(exp_name: str, robot: str) -> str:

    name = exp_name.lower()
    robot_key = robot.lower()

    if robot_key == "panda":
        return "NA"
    all_two_robots = ['panda_jaco', 'panda_sawyer', 'panda_ur5e', 'panda_kinova3']
    if "all_minus" in name:
        return "n-1"
    elif name.startswith("all_"):
        return "n"
    elif robot_key in name and 'panda' in name:
        return "two_robots_seen"
    elif any(alias in name for alias in all_two_robots):
        return "two_robots_unseen"
    elif name.startswith("panda"):
        return "source"
    elif robot_key in name:
        return "target"

    
def categorize_experiment_noise(exp_name: str, robot: str) -> str:

    name = exp_name.lower()
    all_two_robots = ['panda_jaco', 'panda_sawyer', 'panda_ur5e', 'panda_kinova3']
    if name.startswith("all_"):
        return "n"
    elif any(alias in name for alias in all_two_robots):
        return "two_robots"
    elif name.startswith("panda"):
        return "panda"

PLOT_PATH: str = 'results/plots'

PLOT_COLOURS = {
    # First three: earthy tones (Yellow, Brown, Orange)
    'target': '#E6C229',               # Yellow (earthy, slightly muted)
    'n': '#8C564B',      # Brown (earthy, reddish undertone)
    'two_robots_seen': '#E68A00',                    # Orange (earthy, warm)

    # Next three: cool blue-green/turquoise tones
    'source': '#B2E2E2',               # Light blue-green (airy)
    'two_robots_unseen': '#4DAF9C',    # Turquoise (balanced mid-tone)
    'n-1': '#2C7FB8',                  # Blue-green (darker anchor)

    'panda': '#B2E2E2',         
    'two_robots': '#E68A00'      

}

# Category display names for better readability
CATEGORY_LABELS = {
    'target': 'Target-Only (Seen)',
    'two_robots_seen': '1+1 (Seen)',
    'n': 'N (Seen)',
    'source': 'Source-Only (Unseen)', 
    'two_robots_unseen': '1+1 (Unseen, Avg. across 3 robots)',
    'n-1': 'N-1 (Unseen)',
    'panda': 'Source-Only',
    'two_robots': '1+1 (Avg. across 4 robots)',
}


def compute_category_averages(
    data: pd.DataFrame,
    primary_keys: List[str],
    category_order: List[str],
    primary_column: str = 'Robot'
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Compute category averages and error bars for any primary grouping.
    
    Returns:
        Tuple of (averages_dict, errors_dict) where each maps primary_key -> category -> value
    """
    averages: Dict[str, Dict[str, float]] = {}
    errors: Dict[str, Dict[str, float]] = {}
    
    for primary_key in primary_keys:
        primary_df = data[data[primary_column] == primary_key]
        averages[primary_key] = {}
        errors[primary_key] = {}
        
        for cat in category_order:
            cat_data = primary_df[primary_df['Category'] == cat]
            

            success_rate = float(cat_data['Success Rate'].mean())
            num_rollouts = int(cat_data['Num Rollouts'].sum())
        
            p = success_rate
            error = math.sqrt(max(p * (1.0 - p), 0.0) / float(num_rollouts - 1))
                
            averages[primary_key][cat] = success_rate
            errors[primary_key][cat] = error

    
    return averages, errors


def compute_centered_offsets(
    num_bars: int,
    bar_width: float,
    split_index: Optional[int] = None,
    gap: float = 0.0,
) -> List[float]:
    """Compute centered bar offsets with optional gap after split_index.

    Ensures bars within a group do not overlap by spacing centers using
    an effective step larger than bar_width.
    """
    raw_offsets: List[float] = []

    # Use a slightly larger step than bar_width to avoid overlap of adjacent bars
    effective_step = bar_width * 1.1

    for i in range(num_bars):
        offset = i * effective_step
        # Add gap after the split_index to visually separate groups
        if split_index is not None and i >= split_index:
            offset += gap
        raw_offsets.append(offset)

    # Center all offsets around 0 based on the midpoint of the first and last
    if len(raw_offsets) <= 1:
        return [0.0 for _ in raw_offsets]

    center_point = (raw_offsets[0] + raw_offsets[-1]) / 2.0
    return [o - center_point for o in raw_offsets]


def add_bar_value_labels(ax, x_positions: np.ndarray, values: List[float]) -> None:
    for xj, yj in zip(x_positions, values):
        y = yj + 0.01 if yj > 0 else yj + 0.02
        ax.text(xj, y, f"{yj:.2f}", ha='center', va='bottom', fontsize=10, rotation=0)


def format_task_label(task_name: str) -> str:
    return task_name.replace('_', ' ').title()


def get_tasks(df: pd.DataFrame) -> List[str]:
    return ['lift', 'stack', 'two_piece_assembly', 'square', 'can']


def create_bar_plot(
    data_dict: Dict[str, Dict[str, float]], 
    x_labels: List[str],
    category_order: List[str],
    title: str,
    output_path: str,
    figsize: tuple = (12, 8),
    bar_width: float = 0.18,
    x_spacing: float = 1.0,
    split_index: Optional[int] = None,
    error_dict: Optional[Dict[str, Dict[str, float]]] = None,
    n_dict: Optional[Dict[str, Dict[str, int]]] = None,
    show_values: bool = True,
    show_n_values: bool = False
) -> None:
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up positions with proper spacing for wider bars
    x = np.arange(len(x_labels)) * x_spacing
    n_cats = len(category_order)
    offsets = compute_centered_offsets(n_cats, bar_width, split_index=split_index, gap=bar_width * 0.6)
    
    # Get colors for categories
    colors = [PLOT_COLOURS.get(cat, '#666666') for cat in category_order]
    labels = [CATEGORY_LABELS.get(cat, cat) for cat in category_order]
    
    bars_list = []
    for i, (cat, color, label) in enumerate(zip(category_order, colors, labels)):
        y_vals = [data_dict[x_label].get(cat, 0.0) for x_label in x_labels]
        x_pos = x + offsets[i]
        
        # Create bars with professional styling
        errors = [error_dict[x_label].get(cat, 0.0) for x_label in x_labels] if error_dict else None
        
        bars = ax.bar(x_pos, y_vals, bar_width, 
                     label=label,
                     color=color,
                     alpha=0.9,
                     edgecolor='white',
                     linewidth=0.8,
                     yerr=errors,
                     capsize=4,
                     ecolor='#333333',
                     error_kw={'alpha': 0.8})
        
        bars_list.extend(bars)
        
        # Add value labels on bars
        if show_values:
            for xj, yj in zip(x_pos, y_vals):
                idx = list(x_pos).index(xj)
                error_val = errors[idx] if errors else 0.0
                # Place text just above the bar (or error bar) with a small extra gap
                offset = error_val + (0.01 if yj > 0 else 0.02)
                if split_index is not None:
                    ax.text(xj, yj + offset,
                        f'{yj:.2f}',
                        ha='center', va='bottom',
                        fontsize=9,
                        color='#333333')
                else:
                    ax.text(xj, yj + offset,
                        f'{yj:.2f}',
                        ha='center', va='bottom',
                        fontsize=13,
                        color='#333333')

        
        # Add sample size labels
        if show_n_values and n_dict is not None:
            n_vals = [n_dict[x_label].get(cat, 0) for x_label in x_labels]
            for xj, nj in zip(x_pos, n_vals):
                if nj > 0:
                    ax.text(xj, -0.08, f'n={nj}', 
                           ha='center', va='top', 
                           fontsize=8, style='italic',
                           color='#666666')
    
    # Professional styling
    if split_index is None:
        ax.set_title(title, fontweight='bold', pad=30, fontsize=35)
    else:
        ax.set_title(title, fontweight='bold', pad=30)
    
    # Set x-axis without tick marks
    ax.set_xticks(x)
    display_labels = [format_task_label(lbl) for lbl in x_labels]
    ax.set_xticklabels(display_labels, fontweight='medium')
    ax.tick_params(axis='x', length=0)  # Remove x-axis tick marks
    
    # Set y-axis to always be 0 to 1
    ax.set_ylim(0, 1.0)
    
    # Remove percentage formatting - keep as decimal values
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
    
    # Professional legend placed below the plot
    legend = ax.legend(
                      loc='upper center',
                      bbox_to_anchor=(0.5, -0.08),
                      frameon=False,
                      fancybox=False,
                      shadow=False,
                      ncol=min(len(labels), 3),
                      borderpad=0.6,
                      columnspacing=0.8,
                      handletextpad=0.6)

    # Leave minimal extra space at bottom for legend
    plt.subplots_adjust(bottom=0.14)

    # Clean up the plot and add y-axis grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                pad_inches=0.2)
    plt.close()


def draw_grouped_bar_plot_on_ax(
    ax: plt.Axes,
    data_dict: Dict[str, Dict[str, float]],
    x_labels: List[str],
    category_order: List[str],
    bar_width: float = 0.18,
    x_spacing: float = 1.0,
    split_index: Optional[int] = None,
    error_dict: Optional[Dict[str, Dict[str, float]]] = None,
    n_dict: Optional[Dict[str, Dict[str, int]]] = None,
    show_values: bool = True,
    show_n_values: bool = False,
    format_x_labels: bool = True,
) -> List:
    """Draw grouped bar plot on provided axis and return legend handles.

    Returns list of bar containers (for legend extraction).
    """
    x = np.arange(len(x_labels)) * x_spacing
    n_cats = len(category_order)
    offsets = compute_centered_offsets(n_cats, bar_width, split_index=split_index, gap=bar_width * 0.8)

    colors = [PLOT_COLOURS.get(cat, '#666666') for cat in category_order]
    labels = [CATEGORY_LABELS.get(cat, cat) for cat in category_order]

    handles = []
    for i, (cat, color, label) in enumerate(zip(category_order, colors, labels)):
        y_vals = [data_dict[x_label].get(cat, 0.0) for x_label in x_labels]
        x_pos = x + offsets[i]

        errors = [error_dict[x_label].get(cat, 0.0) for x_label in x_labels] if error_dict else None

        bars = ax.bar(
            x_pos,
            y_vals,
            bar_width,
            label=label,
            color=color,
            alpha=0.9,
            edgecolor='white',
            linewidth=0.8,
            yerr=errors,
            capsize=4,
            ecolor='#333333',
            error_kw={'alpha': 0.8},
        )
        handles.append(bars[0])

        if show_values:
            for xj, yj in zip(x_pos, y_vals):
                idx = list(x_pos).index(xj)
                error_val = errors[idx] if errors else 0.0
                offset = error_val + (0.01 if yj > 0 else 0.02)
                ax.text(
                    xj,
                    yj + offset,
                    f'{yj:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    color='#333333',
                )

        if show_n_values and n_dict is not None:
            n_vals = [n_dict[x_label].get(cat, 0) for x_label in x_labels]
            for xj, nj in zip(x_pos, n_vals):
                if nj > 0:
                    ax.text(
                        xj,
                        -0.08,
                        f'n={nj}',
                        ha='center',
                        va='top',
                        fontsize=10,
                        style='italic',
                        color='#666666',
                    )

    ax.set_xticks(x)
    display_labels = [format_task_label(lbl) for lbl in x_labels] if format_x_labels else x_labels
    ax.set_xticklabels(display_labels, fontweight='medium')
    ax.tick_params(axis='x', length=0)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))

    # Keep grid visible (y-axis) while removing bolded axes/borders
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Add vertical grid lines for every other x-axis position (standard grouped plots only)
    for i in range(0, len(x)):  # Every other position (0, 2, 4, ...)
        ax.axvline(x=x[i]+x_spacing/2, color='gray', linestyle='--', alpha=0.7, linewidth=0.6)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    return handles


def create_combined_standard_figure(
    df: pd.DataFrame,
    category_order: List[str],
    robot_order: List[str],
    plot_path: str,
) -> None:
    """Create a 2x3 grid of standard plots with a single shared legend."""
    # Overall + 5 tasks = 6 plots total
    tasks = get_tasks(df)
    tasks = [t for t in tasks if t in ['lift', 'stack', 'two_piece_assembly', 'square', 'can']]

    robot_category_avg, robot_category_errors = compute_category_averages(df, robot_order, category_order, 'Robot')

    fig, axes = plt.subplots(3, 2, figsize=(25, 12))
    axes = axes.flatten()

    # Plot 1: overall
    grouped_bar_width = 0.6
    grouped_x_spacing = 6
    handles = draw_grouped_bar_plot_on_ax(
        axes[5],
        robot_category_avg,
        robot_order,
        category_order,
        bar_width=grouped_bar_width,
        x_spacing=grouped_x_spacing,
        split_index=3,
        error_dict=robot_category_errors,
        show_values=True,
        show_n_values=False,
        format_x_labels=False,
    )
    # Light orange harmonious highlight for the overall subplot background
    axes[5].add_patch(
        plt.Rectangle(
            (0.0, 0.0), 1.0, 1.0,
            transform=axes[5].transAxes,
            facecolor='#4DAF9C',  # harmonious green from sky/nature palette
            alpha=0.1,
            edgecolor='none',
            zorder=0,
        )
    )
    axes[5].set_title('Average Success Rates Across 5 Tasks', fontweight='bold', pad=10)

    # Next 5: one subplot per task
    for idx, task in enumerate(tasks, start=0):
        task_df = df[df['Task'] == task]
        task_avg, task_err = compute_category_averages(task_df, robot_order, category_order, 'Robot')
        draw_grouped_bar_plot_on_ax(
            axes[idx],
            task_avg,
            robot_order,
            category_order,
            bar_width=grouped_bar_width,
            x_spacing=grouped_x_spacing,
            split_index=3,
            error_dict=task_err,
            show_values=True,
            show_n_values=False,
            format_x_labels=False,

        )
        axes[idx].set_title(f'{format_task_label(task)}', fontweight='bold', pad=10)

    # Shared legend below all subplots
    labels = [CATEGORY_LABELS.get(cat, cat) for cat in category_order]
    fig.legend(
        handles,
        labels,
        loc='lower center',
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
        borderpad=0.6,
        columnspacing=0.8,
        handletextpad=0.6,
    )

    plt.subplots_adjust(bottom=0.12, hspace=0.4)

    os.makedirs(plot_path, exist_ok=True)
    output_path = f'{plot_path}/standard_combined.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close(fig)


def create_grouped_bar_plot(*args, **kwargs):
    """Legacy wrapper - use create_bar_plot for new code."""
    return create_bar_plot(*args, **kwargs)

standard_x_spacing = 2
standard_bar_width = 0.23
standard_figsize = (14, 10)
def plot_standard_overall(df: pd.DataFrame, category_order: List[str], robot_order: List[str], plot_path: str) -> None:
    robot_category_avg, robot_category_errors = compute_category_averages(df, robot_order, category_order, 'Robot')
    
    create_bar_plot(
        data_dict=robot_category_avg,
        x_labels=robot_order,
        category_order=category_order,
        title='Average Success Rates Across 5 Tasks',
        output_path=f'{plot_path}/all_robots.png',
        figsize=standard_figsize,
        bar_width=standard_bar_width,
        split_index=3,
        x_spacing=standard_x_spacing,
        error_dict=robot_category_errors,
        show_values=True,
        show_n_values=False
    )


def plot_standard_per_task(df: pd.DataFrame, category_order: List[str], robot_order: List[str], plot_path: str) -> None:
    tasks = get_tasks(df)
    for task in tasks:
        task_df = df[df['Task'] == task]
        if task_df.empty:
            continue

        robot_category_avg, robot_category_errors = compute_category_averages(task_df, robot_order, category_order, 'Robot')
        
        create_bar_plot(
            data_dict=robot_category_avg,
            x_labels=robot_order,
            category_order=category_order,
            title=f'{format_task_label(task)}',
            output_path=f"{plot_path}/{task}_per_robot.png",
            figsize=standard_figsize,
            bar_width=standard_bar_width,
            x_spacing=standard_x_spacing,
            split_index=3,
            error_dict=robot_category_errors,
            show_values=True,
            show_n_values=False
        )


def plot_single_robot_across_tasks(df: pd.DataFrame, category_order: List[str], plot_path: str, title: str) -> None:
    tasks = get_tasks(df)
        
    task_category_avg, task_category_errors = compute_category_averages(df, tasks, category_order, 'Task')
    create_bar_plot(
        data_dict=task_category_avg,
        x_labels=[(t) for t in tasks],
        category_order=category_order,
        title=title,
        output_path=f"{plot_path}/plot.png",
        figsize=(15, 8),
        bar_width=0.25,
        x_spacing=1.5,
        error_dict=task_category_errors,
        show_values=True,
        show_n_values=False
    )

def main() -> None:
    for MODE in ['standard', 'patch', 'lighting']:

        try:
            input_path = f'results/evaluation_results_{MODE}_mode.csv'
            df = pd.read_csv(input_path)
            plot_path = PLOT_PATH + '/' + MODE
            os.makedirs(plot_path, exist_ok=True)
            if MODE == 'standard':
                CATEGORY_LABELS['n'] = 'N (Seen)'
                PLOT_COLOURS['n'] = '#8C564B'
                df['Category'] = df.apply(lambda row: categorize_experiment_standard_mode(row['Exp Name'], row['Robot']), axis=1)
                category_order = ['target', 'two_robots_seen', 'n', 'source', 'two_robots_unseen', 'n-1']
                robot_order  = ['UR5e', 'Kinova3', 'Sawyer', 'Jaco']
                df['Task'] = df.apply(lambda row: categorize_task_name(row['Exp Name']), axis=1)
                df.sort_values(by=['Robot', 'Task'], inplace=True)
                df.to_csv(f'results/evaluation_results_{MODE}_mode_categorized.csv', index=False)
                create_combined_standard_figure(df, category_order, robot_order, plot_path)
                
            else:
                CATEGORY_LABELS['n'] = 'N'
                mpl.rcParams.update({'xtick.labelsize': 25, 'ytick.labelsize': 25, 'legend.fontsize': 25})
                df['Category'] = df.apply(lambda row: categorize_experiment_noise(row['Exp Name'], row['Robot']), axis=1)
                category_order = ['panda', 'two_robots', 'n']
                df['Task'] = df.apply(lambda row: categorize_task_name(row['Exp Name']), axis=1)
                df.sort_values(by=['Robot', 'Task'], inplace=True)
                df.to_csv(f'results/evaluation_results_{MODE}_mode_categorized.csv', index=False)

                title = 'Source Robot Under Visual Occlusion' if MODE == 'patch' else 'Source Robot Under Different Lighting'
                plot_single_robot_across_tasks(df, category_order, plot_path, title)
        except Exception as e:
            print(f"Error creating {MODE} figure: {e}")

if __name__ == "__main__":
    exit(main())