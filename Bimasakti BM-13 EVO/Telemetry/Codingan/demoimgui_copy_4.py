import numpy as np
from imgui_bundle import implot, imgui, immapp, ImVec2, portable_file_dialogs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter
from enum import IntFlag
import time
import math
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
file_name = "log_126withDegree.csv"
def read_file(file_name):
    global data 
    data = pd.read_csv(file_name, delimiter=',', decimal='.')

    # Ensure proper column names
    data.columns = [
        'Timestamp', 'Suspension1', 'Suspension2', 'Suspension3', 'Suspension4',
        'Steering', 'RPM_FrontRight', 'RPM_FrontLeft', 'RPM_RearRight', 'RPM_RearLeft',
        'Theta', 'Derajat'
    ]

    for col in ['Suspension1', 'Suspension2', 'Suspension3', 'Suspension4']:
        data[col] = (data[col] - 26139)/-334.73
        data[col] = data[col] - data[col].iloc[S1:S2].mean()  #CALIBRATE SUSPENSION DATA

    N=17
    data['Suspension1'] = medfilt(data['Suspension1'], kernel_size=N)
    data['Suspension2'] = medfilt(data['Suspension2'], kernel_size=N)
    data['Suspension3'] = medfilt(data['Suspension3'], kernel_size=N)
    data['Suspension4'] = medfilt(data['Suspension4'], kernel_size=N)

    # Zero out all outliers above 5000 in RPM columns
    data.loc[data['RPM_FrontRight'] > 1370, 'RPM_FrontRight'] = 0
    data.loc[data['RPM_FrontLeft'] > 1370, 'RPM_FrontLeft'] = 0
    data.loc[data['RPM_RearRight'] > 1370, 'RPM_RearRight'] = 0
    data.loc[data['RPM_RearLeft'] > 1370, 'RPM_RearLeft'] = 0

    # Median filter
    data['CMA_RPM_FrontRight'] = medfilt(data['RPM_FrontRight'], kernel_size=N)
    data['CMA_RPM_FrontLeft'] = medfilt(data['RPM_FrontLeft'], kernel_size=N)
    data['CMA_RPM_RearRight'] = medfilt(data['RPM_RearRight'], kernel_size=N)
    data['CMA_RPM_RearLeft'] = medfilt(data['RPM_RearLeft'], kernel_size=N)

    # Apply Savitzky-Golay filter 
    data['CMA_RPM_FrontRight'] = savgol_filter(data['CMA_RPM_FrontRight'], window_length=11, polyorder=2)
    data['CMA_RPM_FrontLeft'] = savgol_filter(data['CMA_RPM_FrontLeft'], window_length=11, polyorder=2)
    data['CMA_RPM_RearRight'] = savgol_filter(data['CMA_RPM_RearRight'], window_length=11, polyorder=2)
    data['CMA_RPM_RearLeft'] = savgol_filter(data['CMA_RPM_RearLeft'], window_length=11, polyorder=2)

    data['KMPH_FrontRight'] = data['CMA_RPM_FrontRight'] * 0.0766
    data['KMPH_FrontLeft'] = data['CMA_RPM_FrontLeft'] * 0.0766
    data['AverageSpeed'] = (data['KMPH_FrontRight'] + data['KMPH_FrontLeft']) / 2

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

def add_speedometer(
    draw_list,
    target_speed,
    radius=150.0,
    max_speed=150.0,
    pos_offset=ImVec2(0, 0),
    line_thickness=3.0,
    lerp_factor=0.1
):
    # Get the center of the current window + offset
    window_pos = imgui.get_window_pos()
    window_size = imgui.get_window_size()
    center = ImVec2(
        window_pos.x + window_size.x * 0.5 + pos_offset.x,
        window_pos.y + window_size.y * 0.5 + pos_offset.y
    )

    start_angle = -math.pi * 1.25
    end_angle   =  math.pi * 0.25

    # Smoothly interpolate speed toward target_speed
    _speed_state["current_speed"] = lerp(
        _speed_state["current_speed"], target_speed, lerp_factor
    )

    current_speed = _speed_state["current_speed"]

    # 1) Draw outer circle
    draw_list.add_circle(
        center, radius,
        color_uint8(150, 50, 50, 255),
        num_segments=100,
        thickness=line_thickness
    )

    # 2) Draw markings around the speedometer
    step_mark = 10
    for i in range(0, int(max_speed) + 1, step_mark):
        angle = start_angle + (end_angle - start_angle) * (i / max_speed)
        mark_start = ImVec2(
            center.x + math.cos(angle) * (radius - 10),
            center.y + math.sin(angle) * (radius - 10)
        )
        mark_end = ImVec2(
            center.x + math.cos(angle) * radius,
            center.y + math.sin(angle) * radius
        )

        draw_list.add_line(
            mark_start, mark_end,
            color_uint8(200, 200, 200, 255),
            line_thickness * 0.5
        )

        # Numeric label (i.e. the "20, 40, 60..." markings)
        mark_text = f"{i}"
        text_size = imgui.calc_text_size(mark_text)
        text_pos = ImVec2(
            center.x + math.cos(angle) * (radius - 25),
            center.y + math.sin(angle) * (radius - 25)
        )
        # Offset text so it's centered
        text_pos.x -= text_size.x * 0.5
        text_pos.y -= text_size.y * 0.5

        draw_list.add_text(
            text_pos,
            color_uint8(255, 255, 255, 255),
            mark_text
        )

    # 3) Draw the speedometer needle
    speed_angle = start_angle + (end_angle - start_angle) * (current_speed / max_speed)
    needle_end = ImVec2(
        center.x + math.cos(speed_angle) * radius,
        center.y + math.sin(speed_angle) * radius
    )

    start_thickness = line_thickness
    end_thickness   = line_thickness * 0.5  # Thinner at the tip

    # Draw the "growing line" in segments
    t = 0.0
    while t <= 1.0:
        segment_point = ImVec2(
            center.x + (needle_end.x - center.x) * t,
            center.y + (needle_end.y - center.y) * t
        )
        thickness = start_thickness + (end_thickness - start_thickness) * t

        draw_list.add_line(
            center, segment_point,
            color_uint8(255, 55, 55, 255),
            thickness
        )
        t += 0.05

    # 4) Draw a circle at the center (where the needle pivots)
    circle_radius = line_thickness * 3.0
    draw_list.add_circle_filled(
        center, circle_radius,
        color_uint8(155, 155, 155, 255)
    )

    # 5) Draw the speed text (e.g., "123 km/h")
    speed_text = f"{current_speed:.0f} km/h"
    text_size  = imgui.calc_text_size(speed_text)
    text_pos   = ImVec2(
        center.x - text_size.x * 0.5,
        center.y + text_size.y * 4.5
    )
    draw_list.add_text(
        text_pos,
        color_uint8(255, 255, 255, 255),
        speed_text
    )

    # 6) Optional: draw a "check engine" icon if speed > 30
    # if current_speed > 30.0:
    #     # Example unicode icon and offset
    #     check_engine_icon = "\ue16e"
    #     icon_offset = ImVec2(0, 85)  # shift downward
    #     # Slight shift for text center
    #     icon_pos = ImVec2(center.x - 10, center.y - 10)
    #     icon_pos.x += icon_offset.x
    #     icon_pos.y += icon_offset.y

    #     draw_list.add_text(
    #         icon_pos,
    #         color_uint8(255, 155, 16, 255),
    #         check_engine_icon
    #     )

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
    global x_values, y1_values, y2_values, y3_values, y4_values, y5_values
    x_values = np.array(data.index, dtype=np.float32)
    y1_values = data['CMA_RPM_FrontRight'].to_numpy(dtype=np.float32)
    y2_values = data['CMA_RPM_FrontLeft'].to_numpy(dtype=np.float32)
    y3_values = data['CMA_RPM_RearRight'].to_numpy(dtype=np.float32)
    y4_values = data['CMA_RPM_RearLeft'].to_numpy(dtype=np.float32)
    y5_values = data['Derajat'].to_numpy(dtype=np.float32)
    slider_value[0]=0
    play_mode[0]=False
    return x_values, y1_values, y2_values, y3_values, y4_values, y5_values

def main():
    global data
    read_file(file_name)
    update_plot_data()

    # x_values = np.array(data.index, dtype=np.float32)
    # y1_values = data['CMA_RPM_FrontRight'].to_numpy(dtype=np.float32)
    # y2_values = data['CMA_RPM_FrontLeft'].to_numpy(dtype=np.float32)
    # y3_values = data['CMA_RPM_RearRight'].to_numpy(dtype=np.float32)
    # y4_values = data['CMA_RPM_RearLeft'].to_numpy(dtype=np.float32)
    # y5_values = data['Derajat'].to_numpy(dtype=np.float32)





    def gui():
        global data
        # File-open control


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

        imgui.begin("Plot Window")
        imgui.push_item_width(-1) 
        changed, slider_value[0] = imgui.slider_int(
            "Index", slider_value[0], 0, len(x_values) - 1
        )
        imgui.pop_item_width()

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

        # Steering angle as horizontal bar
        imgui.begin("Steering Angle")
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
            value_bar(label, value, ImVec2(50, 400), min_value=-100, max_value=1370, flags=ValueBarFlags.VERTICAL)
            imgui.end()

        # Speedometer
        imgui.begin("Speedometer")
        draw_list = imgui.get_window_draw_list()
        target_speed = data['AverageSpeed'].iloc[slider_value[0]]
        add_speedometer(draw_list, target_speed)
        imgui.end()

        imgui.begin("File Controls")
        if imgui.button("Open CSV"):
            new_file = open_csv_dialog()
            if new_file:
                read_file(new_file)
                update_plot_data()
                # Optionally repeat your pre-processing steps here
        imgui.end()

    immapp.run(gui, fps_idle=144, with_implot=True, with_markdown=True)

if __name__ == "__main__":
    main()
