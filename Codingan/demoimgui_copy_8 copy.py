import numpy as np
from imgui_bundle import implot, imgui, immapp, ImVec2, portable_file_dialogs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter
from enum import IntFlag
import time
import math
import threading
import serial.tools.list_ports
import os
from datetime import datetime

# Define input columns (must match CSV file)
input_columns = [
    'Timestamp', 'Suspension1', 'Suspension2', 'Suspension3', 'Suspension4',
    'Steering', 'RPM_FrontRight', 'RPM_FrontLeft', 'RPM_RearRight', 'RPM_RearLeft',
    'Theta', 'Derajat', 'TPS', 'GPSS', 'RPM'
]

# Define calculated columns that will be added during processing
calculated_columns = [
    'CMA_RPM_FrontRight', 'CMA_RPM_FrontLeft', 'CMA_RPM_RearRight', 'CMA_RPM_RearLeft',
    'KMPH_FrontRight', 'KMPH_FrontLeft', 'AverageSpeed'
]

# All columns (used for DataFrame initialization)
columns = input_columns + calculated_columns

# Initialize global DataFrame
data = pd.DataFrame(columns=columns)

S1 = 3000 # (START)SESUAIKAN DENGAN INDEX SUSPENSI YANG MAU DIJADIKAN ACUAN NOL
S2 = 4000 # (END)
class ValueBarFlags(IntFlag):
    NONE = 0
    VERTICAL = 1
    CENTER_ZERO = 2

def value_bar(label, value, size, min_value=0.0, max_value=1.0, flags=ValueBarFlags.NONE):
    is_horizontal = not (flags & ValueBarFlags.VERTICAL)
    is_center_zero = bool(flags & ValueBarFlags.CENTER_ZERO)

    draw_list = imgui.get_window_draw_list()
    cursor_pos = imgui.get_cursor_screen_pos()
    frame_height = imgui.get_frame_height()
    label_size = imgui.calc_text_size(label) if label else ImVec2(0, 0)
    
    # FIRST, compute normalized fraction in [0, 1], 
    # where 0 => min_value, 1 => max_value.
    full_range = (max_value - min_value)
    fraction = (value - min_value) / full_range  # 0..1
    
    if is_horizontal:
        rect_size = ImVec2(size.x if size.x > 0 else imgui.calc_item_width(), frame_height)
        rect_start = cursor_pos
        rect_end   = ImVec2(rect_start.x + rect_size.x, rect_start.y + rect_size.y)
        
        # Draw background
        draw_list.add_rect_filled(
            rect_start, rect_end,
            imgui.get_color_u32(imgui.Col_.frame_bg),
            imgui.get_style().frame_rounding
        )

        if is_center_zero:
            # For horizontal, 0.5 fraction = center.
            center_fraction = 0.5
            center_x = rect_start.x + rect_size.x * center_fraction
            # Shift from center based on (fraction - 0.5).
            fill_offset = (fraction - 0.5) * rect_size.x
            # If fill_offset >= 0 => fill right side, else fill left side
            fill_min_x = min(center_x, center_x + fill_offset)
            fill_max_x = max(center_x, center_x + fill_offset)
            
            draw_list.add_rect_filled(
                ImVec2(fill_min_x, rect_start.y),
                ImVec2(fill_max_x, rect_end.y),
                imgui.get_color_u32(imgui.Col_.plot_histogram),
                imgui.get_style().frame_rounding
            )
        else:
            # Normal fill from left to fraction
            fill_end_x = rect_start.x + rect_size.x * fraction
            draw_list.add_rect_filled(
                rect_start, ImVec2(fill_end_x, rect_end.y),
                imgui.get_color_u32(imgui.Col_.plot_histogram),
                imgui.get_style().frame_rounding
            )
            
        # Draw text, label, etc. (unchanged) ...
        value_text = f"{value:.2f}" if is_horizontal else f"{value:.1f}"
        text_size = imgui.calc_text_size(value_text)
        text_pos = ImVec2(rect_start.x + (rect_size.x - text_size.x) / 2, rect_start.y + (rect_size.y - text_size.y) / 2)
        draw_list.add_text(text_pos, imgui.get_color_u32(imgui.Col_.text), value_text)
    # Draw label text
        if label:
            if is_horizontal:
                label_pos = ImVec2(
                    rect_start.x + rect_size.x + imgui.get_style().item_inner_spacing.x,
                    rect_start.y + (rect_size.y - label_size.y) / 2
                )
            else:
                label_pos = ImVec2(
                    rect_start.x + (rect_size.x - label_size.x) / 2,
                    rect_start.y + rect_size.y + imgui.get_style().item_inner_spacing.y
                )
            draw_list.add_text(label_pos, imgui.get_color_u32(imgui.Col_.text), label)

    else:
        # VERTICAL case
        rect_size = ImVec2(frame_height * 2, size.y if size.y > 0 else size.y - label_size.y)
        rect_start = ImVec2(cursor_pos.x + max(0.0, (label_size.x - rect_size.x) / 2), cursor_pos.y)
        rect_end   = ImVec2(rect_start.x + rect_size.x, rect_start.y + rect_size.y)

        # Draw background
        draw_list.add_rect_filled(
            rect_start, rect_end,
            imgui.get_color_u32(imgui.Col_.frame_bg),
            imgui.get_style().frame_rounding
        )

        if is_center_zero:
            # For vertical, 0.5 fraction = center
            center_fraction = 0.5
            center_y = rect_start.y + rect_size.y * center_fraction
            # offset from center based on (fraction - 0.5)
            fill_offset = (fraction - 0.5) * rect_size.y
            # if fill_offset >= 0 => fill upward, else fill downward
            fill_min_y = min(center_y, center_y - fill_offset)
            fill_max_y = max(center_y, center_y - fill_offset)
            
            draw_list.add_rect_filled(
                ImVec2(rect_start.x, fill_min_y),
                ImVec2(rect_end.x, fill_max_y),
                imgui.get_color_u32(imgui.Col_.plot_histogram),
                imgui.get_style().frame_rounding
            )
        else:
            # Normal fill from bottom to fraction
            fill_start_y = rect_start.y + rect_size.y * (1.0 - fraction)
            draw_list.add_rect_filled(
                ImVec2(rect_start.x, fill_start_y),
                rect_end,
                imgui.get_color_u32(imgui.Col_.plot_histogram),
                imgui.get_style().frame_rounding
            )

        # Draw text, label, etc. (unchanged) ...
        value_text = f"{value:.2f}" if is_horizontal else f"{value:.1f}"
        text_size = imgui.calc_text_size(value_text)
        text_pos = ImVec2(rect_start.x + (rect_size.x - text_size.x) / 2, rect_start.y + (rect_size.y - text_size.y) / 2)
        draw_list.add_text(text_pos, imgui.get_color_u32(imgui.Col_.text), value_text)
    # Draw label text
        if label:
            if is_horizontal:
                label_pos = ImVec2(
                    rect_start.x + rect_size.x + imgui.get_style().item_inner_spacing.x,
                    rect_start.y + (rect_size.y - label_size.y) / 2
                )
            else:
                label_pos = ImVec2(
                    rect_start.x + (rect_size.x - label_size.x) / 2,
                    rect_start.y + rect_size.y + imgui.get_style().item_inner_spacing.y
                )
            draw_list.add_text(label_pos, imgui.get_color_u32(imgui.Col_.text), label)
###############CUSTOMDEFINED################

# Load the data
file_name = "log_126withDegreecopy.csv"
def process_dataframe(df):
    """Apply all the filtering and calculations to a DataFrame"""
    try:
        # Create a copy to avoid modifying the original
        processed = df.copy()

        # Ensure all input columns exist
        for col in input_columns:
            if col not in processed.columns:
                processed[col] = 0.0

        # Only process if we have enough data
        N = min(17, len(processed))
        if N % 2 == 0:  # Ensure N is odd
            N -= 1
        
        # Process suspension data
        for col in ['Suspension1', 'Suspension2', 'Suspension3', 'Suspension4']:
            processed[col] = (processed[col] - 26139)/-334.73
            if len(processed) > S2:
                processed[col] = processed[col] - processed[col].iloc[S1:S2].mean()
            if len(processed) >= N:
                processed[col] = medfilt(processed[col], kernel_size=N)

        # Zero out RPM outliers
        rpm_cols = ['RPM_FrontRight', 'RPM_FrontLeft', 'RPM_RearRight', 'RPM_RearLeft']
        for col in rpm_cols:
            processed.loc[processed[col] > 1370, col] = 0

        # Create CMA columns explicitly
        processed['CMA_RPM_FrontRight'] = processed['RPM_FrontRight']
        processed['CMA_RPM_FrontLeft'] = processed['RPM_FrontLeft']
        processed['CMA_RPM_RearRight'] = processed['RPM_RearRight']
        processed['CMA_RPM_RearLeft'] = processed['RPM_RearLeft']

        # Apply median filter if enough data
        if len(processed) >= N:
            for col in ['CMA_RPM_FrontRight', 'CMA_RPM_FrontLeft', 
                       'CMA_RPM_RearRight', 'CMA_RPM_RearLeft']:
                processed[col] = medfilt(processed[col].to_numpy(), kernel_size=N)

            # Apply Savitzky-Golay if enough data
            window_length = min(11, len(processed))
            if window_length > 2 and window_length % 2 == 1:
                for col in ['CMA_RPM_FrontRight', 'CMA_RPM_FrontLeft', 
                           'CMA_RPM_RearRight', 'CMA_RPM_RearLeft']:
                    processed[col] = savgol_filter(processed[col], window_length=window_length, polyorder=2)

        # Calculate speeds
        processed['KMPH_FrontRight'] = processed['CMA_RPM_FrontRight'] * 0.0766
        processed['KMPH_FrontLeft'] = processed['CMA_RPM_FrontLeft'] * 0.0766
        processed['AverageSpeed'] = (processed['KMPH_FrontRight'] + processed['KMPH_FrontLeft']) / 2

        # Use forward fill followed by backward fill (updated method)
        processed = processed.ffill().bfill()

        return processed

    except Exception as e:
        print(f"Error processing data: {e}")
        # Ensure all required columns exist even on error
        for col in columns:
            if col not in df.columns:
                df[col] = 0.0
        return df

# Modify read_file to use the new function
def read_file(file_name):
    global data
    # Read CSV with only input columns
    data = pd.read_csv(file_name, delimiter=',', decimal='.')
    
    # Verify column count matches expected
    if len(data.columns) != len(input_columns):
        raise ValueError(f"CSV has {len(data.columns)} columns, expected {len(input_columns)}")
    
    # Assign input column names
    data.columns = input_columns
    
    # Process data (this will add calculated columns)
    data = process_dataframe(data)

print("FILE NAME : "+f'{file_name}')

slider_value = [0]
play_mode = [False]
sample_rate = [2.0]  # frames per second
time_state = {"last_time": time.time()}

_speed_state = {"current_speed": 0.0}

def lerp(start, end, t):
    return start + (end - start) * t

def color_uint8(r, g, b, a=255):
    """Convert RGBA to the packed integer format ImGui expects."""
    return (a << 24) | (b << 16) | (g << 8) | r

class Speedometer:
    def __init__(self, max_speed=150.0, radius=150.0, pos_offset=ImVec2(0, 0),
                 line_thickness=3.0, lerp_factor=0.1):
        self.max_speed = max_speed
        self.radius = radius
        self.pos_offset = pos_offset
        self.line_thickness = line_thickness
        self.lerp_factor = lerp_factor
        self.current_speed = 0.0

    def render(self, draw_list, target_speed):
        window_pos = imgui.get_window_pos()
        window_size = imgui.get_window_size()
        center = ImVec2(
            window_pos.x + window_size.x * 0.5 + self.pos_offset.x,
            window_pos.y + window_size.y * 0.5 + self.pos_offset.y
        )
        start_angle = -math.pi * 1.25
        end_angle   =  math.pi * 0.25

        # Smoothly interpolate speed
        self.current_speed = lerp(self.current_speed, target_speed, self.lerp_factor)

        # 1) Outer circle
        draw_list.add_circle(
            center, self.radius,
            color_uint8(150, 50, 50, 255),
            num_segments=100,
            thickness=self.line_thickness
        )

        # 2) Markings
        step_mark = 10
        for i in range(0, int(self.max_speed) + 1, step_mark):
            angle = start_angle + (end_angle - start_angle) * (i / self.max_speed)
            mark_start = ImVec2(
                center.x + math.cos(angle) * (self.radius - 10),
                center.y + math.sin(angle) * (self.radius - 10)
            )
            mark_end = ImVec2(
                center.x + math.cos(angle) * self.radius,
                center.y + math.sin(angle) * self.radius
            )
            draw_list.add_line(mark_start, mark_end, color_uint8(200, 200, 200, 255), self.line_thickness * 0.5)
            mark_text = f"{i}"
            text_size = imgui.calc_text_size(mark_text)
            text_pos = ImVec2(
                center.x + math.cos(angle) * (self.radius - 25),
                center.y + math.sin(angle) * (self.radius - 25)
            )
            text_pos.x -= text_size.x * 0.5
            text_pos.y -= text_size.y * 0.5
            draw_list.add_text(text_pos, color_uint8(255, 255, 255, 255), mark_text)

        # 3) Needle
        speed_angle = start_angle + (end_angle - start_angle) * (self.current_speed / self.max_speed)
        needle_end = ImVec2(
            center.x + math.cos(speed_angle) * self.radius,
            center.y + math.sin(speed_angle) * self.radius
        )
        start_thickness = self.line_thickness
        end_thickness   = self.line_thickness * 0.5
        t = 0.0
        while t <= 1.0:
            segment_point = ImVec2(
                center.x + (needle_end.x - center.x) * t,
                center.y + (needle_end.y - center.y) * t
            )
            thickness = start_thickness + (end_thickness - start_thickness) * t
            draw_list.add_line(center, segment_point, color_uint8(255, 55, 55, 255), thickness)
            t += 0.05

        # 4) Hub
        draw_list.add_circle_filled(
            center, self.line_thickness * 3.0,
            color_uint8(155, 155, 155, 255)
        )

        # 5) Speed text
        speed_text = f"{self.current_speed:.0f} km/h"
        text_size  = imgui.calc_text_size(speed_text)
        text_pos   = ImVec2(center.x - text_size.x * 0.5, center.y + text_size.y * 4.5)
        draw_list.add_text(text_pos, color_uint8(255, 255, 255, 255), speed_text)

def open_csv_dialog():
    dialog = portable_file_dialogs.open_file(
        title="Select CSV File",
        default_path="",
        filters=["*.csv", "*.*"],
        options=portable_file_dialogs.opt.none
    )
    # 'dialog' is an object. We call .result() to get the user’s selection.
    selected_files = dialog.result()
    if not selected_files:
        return None
    # Now selected_files is typically a list (with one entry when opt.none is used)
    return selected_files[0]

def update_plot_data():
    global x_values, y1_values, y2_values, y3_values, y4_values, y5_values, y6_values
    x_values = np.array(data.index, dtype=np.float32)
    y1_values = data['CMA_RPM_FrontRight'].to_numpy(dtype=np.float32)
    y2_values = data['CMA_RPM_FrontLeft'].to_numpy(dtype=np.float32)
    y3_values = data['CMA_RPM_RearRight'].to_numpy(dtype=np.float32)
    y4_values = data['CMA_RPM_RearLeft'].to_numpy(dtype=np.float32)
    y5_values = data['Derajat'].to_numpy(dtype=np.float32)
    y6_values = data['RPM'].to_numpy(dtype=np.float32)
    
    # Move slider to newest data if in serial mode and auto-follow is on
    if use_serial[0] and auto_follow[0]:
        slider_value[0] = len(data) - 1
    
    return x_values, y1_values, y2_values, y3_values, y4_values, y5_values

auto_follow = [False]  # track whether auto-follow is on/off

# Indicate whether we use CSV (default) or Serial
use_serial = [False]
serial_thread = None
stop_serial_event = threading.Event()

# Add these near the top with other globals, after the imports
SERIAL_PORT = None  # Will be set when user selects a port
BAUD_RATE = None   # Will be set when user selects baud rate

# Example lists of baud rates you want to offer:
baud_list = [9600, 19200, 38400, 57600, 115200, 230400]
selected_baud_idx = [0]  # Default to 115200

# We'll gather available ports automatically:
available_ports = [p.device for p in serial.tools.list_ports.comports()]
if not available_ports:
    # Fallback if nothing found
    available_ports = ["COM1", "COM2", "/dev/ttyUSB0"]
selected_port_idx = [0]  # Default selection

data_lock = threading.Lock()  # For thread-safe data updates

LOG_DIR = "logs"  # Directory to store log files
current_log_file = None
logging_paused = [False]  # Use list for imgui compatibility
total_rows_logged = [0]  # Track total rows written
logging_status = ["Not logging"]  # Current status message

def start_serial_reading():
    global serial_thread, current_log_file, slider_value
    stop_serial_event.clear()
    
    # Reset slider and data when starting serial mode
    slider_value[0] = 0
    
    # Create logs directory if it doesn't exist
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    # Create new log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_log_file = os.path.join(LOG_DIR, f"telemetry_{timestamp}.csv")
    
    # Write header to the log file
    with open(current_log_file, 'w') as f:
        f.write(','.join(columns) + '\n')
    
    total_rows_logged[0] = 0
    logging_status[0] = f"Logging to: {os.path.basename(current_log_file)}"
    
    serial_thread = threading.Thread(target=serial_loop, daemon=True)
    serial_thread.start()

def stop_serial_reading():
    global serial_thread
    if serial_thread:
        stop_serial_event.set()
        serial_thread.join()
        serial_thread = None

def serial_loop():
    global data
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            while not stop_serial_event.is_set():
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue
                    values = line.split(',')
                    if len(values) != len(input_columns):  # Check against input_columns instead
                        print(f"Warning: Expected {len(input_columns)} input values, got {len(values)} -> {values}")
                        continue

                    try:
                        # Convert values to float before creating DataFrame
                        row = [float(v) for v in values]
                        # Create DataFrame with just input columns
                        df_row = pd.DataFrame([row], columns=input_columns)
                        
                        with data_lock:
                            if len(data) == 0:
                                data = df_row
                            else:
                                data = pd.concat([data, df_row], ignore_index=True)
                            
                            # Process the entire DataFrame to add calculated columns
                            data = process_dataframe(data)
                            
                            processed_row = process_dataframe(df_row)
                            
                            # Write to log file if logging isn't paused
                            if not logging_paused[0] and current_log_file:
                                processed_row.to_csv(current_log_file, mode='a', header=False, index=False)
                                total_rows_logged[0] += 1
                        
                        # Update plot data outside the lock
                        update_plot_data()
                            
                    except ValueError as e:
                        print(f"ValueError: Could not convert values to float: {values} - {e}")
                        continue

                except serial.SerialException as e:
                    print(f"SerialException: {e}")
                    break

    except serial.SerialException as e:
        print(f"Serial error: {e}")

def main():
    global data, auto_follow, use_serial, SERIAL_PORT, BAUD_RATE
    # By default, read from CSV
    file_name = "log_126withDegreecopy.csv"
    read_file(file_name)
    update_plot_data()
    speedometer1 = Speedometer()
    speedometer2 = Speedometer()

    def gui():
        global data, SERIAL_PORT, BAUD_RATE
        
        # Safety check for slider value at the start of gui()
        if len(data) == 0:
            slider_value[0] = 0
        elif slider_value[0] >= len(data):
            slider_value[0] = max(0, len(data) - 1)

        # ---------------- File Controls ----------------
        imgui.begin("File Controls")

        # Serial Port combo
        changed_port = False
        if imgui.begin_combo("Serial Port", available_ports[selected_port_idx[0]]):
            for i, port in enumerate(available_ports):
                is_selected = (selected_port_idx[0] == i)
                if imgui.selectable(port, is_selected):
                    selected_port_idx[0] = i
                    changed_port = True
                    SERIAL_PORT = available_ports[selected_port_idx[0]]  # Update immediately
                if is_selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()

        # Baud Rate combo
        changed_baud = False
        if imgui.begin_combo("Baud Rate", str(baud_list[selected_baud_idx[0]])):
            for i, baud in enumerate(baud_list):
                is_selected = (selected_baud_idx[0] == i)
                if imgui.selectable(str(baud), is_selected):
                    selected_baud_idx[0] = i
                    changed_baud = True
                    BAUD_RATE = baud_list[selected_baud_idx[0]]  # Update immediately
                if is_selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()

        # Button to toggle between CSV mode and Serial mode
        if imgui.button("Toggle CSV / Serial"):
            if use_serial[0]:
                stop_serial_reading()
                use_serial[0] = False
                auto_follow[0] = False
                print("Switched to CSV mode")
                # Reset slider when switching to CSV
                if len(data) > 0:
                    slider_value[0] = 0
            else:
                if SERIAL_PORT and BAUD_RATE:  # Only switch if port and baud are set
                    use_serial[0] = True
                    auto_follow[0] = True
                    data = pd.DataFrame(columns=columns)
                    # Reset slider when switching to Serial
                    slider_value[0] = 0
                    start_serial_reading()
                    print(f"Switched to Serial mode on {SERIAL_PORT} @ {BAUD_RATE} baud")
                else:
                    print("Please select both Port and Baud Rate first")

        # "Open CSV" only if not in Serial mode
        if not use_serial[0]:
            if imgui.button("Open CSV"):
                new_file = open_csv_dialog()
                if new_file:
                    read_file(new_file)
                    update_plot_data()

        # Toggle auto-follow
        if imgui.button("Toggle Auto-Follow"):
            auto_follow[0] = not auto_follow[0]

        imgui.end()
        # --------------- End File Controls ---------------

        # Playback controls
        imgui.begin("Playback Controls")
        if imgui.button("Play/Pause"):
            play_mode[0] = not play_mode[0]
        changed, sample_rate[0] = imgui.slider_float(
            "Sample Rate (Hz)", sample_rate[0], 0.1, 30.0
        )
        imgui.end()

        # Move slider automatically if playing
        current_time = time.time()
        if play_mode[0] and (current_time - time_state["last_time"] > 1.0 / sample_rate[0]):
            slider_value[0] += 1
            if slider_value[0] >= len(x_values):
                slider_value[0] = len(x_values) - 1
                play_mode[0] = False
            time_state["last_time"] = current_time

        # Modify all data access sections to check length first
        if len(data) > 0:  # Only try to plot/show data if we have any
            # Plot Window
            imgui.begin("Plot Window")
            imgui.push_item_width(-1)
            if not (use_serial[0] and auto_follow[0]):  # Only allow manual sliding when not auto-following
                changed, slider_value[0] = imgui.slider_int(
                    "Index", slider_value[0], 0, max(0, len(data) - 1)
                )
            else:
                imgui.text(f"Index: {slider_value[0]} (Auto-following)")
            imgui.pop_item_width()
            imgui.end()

            # Plots, bars, etc...
            imgui.begin("Plot")
            if len(x_values) > 0:  # Check array lengths too
                if auto_follow[0]:
                    center = float(slider_value[0])
                    half_width = 200
                    implot.set_next_axis_limits(implot.ImAxis_.x1, 
                        center - half_width, center + half_width, 
                        implot.Cond_.always)

                if implot.begin_plot("Plot"):
                    implot.plot_line("Front Right RPM", x_values, y1_values)
                    implot.plot_line("Front Left RPM", x_values, y2_values)
                    implot.plot_line("Rear Right RPM", x_values, y3_values)
                    implot.plot_line("Rear Left RPM", x_values, y4_values)
                    implot.plot_line("Derajat", x_values, y5_values)

                    # Vertical line
                    line_position = np.array([float(slider_value[0])], dtype=np.float64)
                    implot.plot_inf_lines("Slider Index Line", line_position)
                    implot.end_plot()
            imgui.end()

            imgui.begin("RPM Plot")
            if auto_follow[0]:
                center = float(slider_value[0])
                half_width = 200
                implot.set_next_axis_limits(implot.ImAxis_.x1, center - half_width, center + half_width, implot.Cond_.always)
            if implot.begin_plot("RPM"):
                implot.plot_line("RPM", x_values, y6_values)

                # Vertical line
                line_position = np.array([float(slider_value[0])], dtype=np.float64)
                implot.plot_inf_lines("Slider Index Line", line_position)
                implot.end_plot()
            imgui.end()

            # Steering angle as horizontal bar
            imgui.begin("Steering Angle")
            if len(data) > 0:
                derajat_value = data['Derajat'].iloc[slider_value[0]]
                window_width = imgui.get_window_size().x - imgui.get_style().window_padding.x * 2
                if derajat_value >= 0:
                    imgui.push_style_color(imgui.Col_.plot_histogram, (0.0, 0.0, 1.0, 1.0))  # Blue
                else:
                    imgui.push_style_color(imgui.Col_.plot_histogram, (1.0, 0.0, 0.0, 1.0))  # Red
                value_bar("Derajat", derajat_value, ImVec2(window_width, 400),
                        min_value=-60, max_value=60, flags=ValueBarFlags.NONE | ValueBarFlags.CENTER_ZERO)
                imgui.pop_style_color()  # Reset color to default
            imgui.end()

            # Display suspension values as vertical bars
            suspension_labels = ['Suspension1', 'Suspension2', 'Suspension3', 'Suspension4']
            for label in suspension_labels:
                imgui.begin(f"{label}")
                
                value = data[label].iloc[slider_value[0]]
                
                # Set color based on value
                if value >= 0:
                    imgui.push_style_color(imgui.Col_.plot_histogram, (0.0, 0.0, 1.0, 1.0))  # Blue
                else:
                    imgui.push_style_color(imgui.Col_.plot_histogram, (1.0, 0.0, 0.0, 1.0))  # Red

                # Adjust min_value and max_value symmetrically to center the bar at zero
                value_bar(label, value, ImVec2(50, 400), min_value=-6, max_value=6, flags=ValueBarFlags.VERTICAL | ValueBarFlags.CENTER_ZERO)

                imgui.pop_style_color()  # Reset color to default
                
                imgui.end()



            # Display speeds as vertical bars
            speed_labels = ['CMA_RPM_FrontRight', 'CMA_RPM_FrontLeft', 'CMA_RPM_RearRight', 'CMA_RPM_RearLeft']
            for label in speed_labels:
                imgui.begin(f"{label} Window")
                value = data[label].iloc[slider_value[0]]
                imgui.push_style_color(imgui.Col_.plot_histogram, (0.0, 0.0, 1.0, 1.0))  # Blue
                value_bar(label, value, ImVec2(50, 400), min_value=-100, max_value=1370, flags=ValueBarFlags.VERTICAL)
                imgui.pop_style_color()  # Reset color to default
                imgui.end()

            #TPS
            imgui.begin("TPS")
            value = data['TPS'].iloc[slider_value[0]]
            imgui.push_style_color(imgui.Col_.plot_histogram, (0.0, 1.0, 0.0, 1.0))  # Blue
            value_bar('TPS', value, ImVec2(50, 400), min_value=0, max_value=5, flags=ValueBarFlags.VERTICAL)
            imgui.pop_style_color()  # Reset color to default
            imgui.end()

            #RPM
            imgui.begin("RPM")
            value = data['RPM'].iloc[slider_value[0]]
            imgui.push_style_color(imgui.Col_.plot_histogram, (1.0, 1.0, 0.0, 1.0))  # Blue
            value_bar('RPM', value, ImVec2(50, 400), min_value=0, max_value=13500, flags=ValueBarFlags.VERTICAL)
            imgui.pop_style_color()  # Reset color to default
            imgui.end()

            # Speedometer
            imgui.begin("Speedometer")
            draw_list = imgui.get_window_draw_list()
            target_speed = data['AverageSpeed'].iloc[slider_value[0]]
            speedometer1.render(draw_list, target_speed)
            imgui.end()

            imgui.begin("GPS Speed")
            draw_list = imgui.get_window_draw_list()
            target_speed = data['GPSS'].iloc[slider_value[0]]
            speedometer2.render(draw_list, target_speed)
            imgui.end()
        
        else:
            # Show a message when no data is available
            imgui.begin("Status")
            imgui.text("Waiting for data...")
            imgui.end()

        # Add Logging Status Window
        imgui.begin("Logging Status")
        
        # Show current log file and rows logged
        if current_log_file:
            imgui.text(logging_status[0])
            imgui.text(f"Total rows logged: {total_rows_logged[0]}")
            
            # Pause/Resume button
            if imgui.button("Pause Logging" if not logging_paused[0] else "Resume Logging"):
                logging_paused[0] = not logging_paused[0]
                logging_status[0] = f"Logging to: {os.path.basename(current_log_file)}"
                if logging_paused[0]:
                    logging_status[0] += " (PAUSED)"
        else:
            imgui.text("Not logging")
        
        imgui.end()

    immapp.run(gui, fps_idle=144, with_implot=True, with_markdown=True)

if __name__ == "__main__":
    main()
