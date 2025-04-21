import numpy as np
from imgui_bundle import implot, imgui, immapp, ImVec2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter
from enum import IntFlag
import time

class ValueBarFlags(IntFlag):
    NONE = 0
    VERTICAL = 1

def value_bar(label, value, size, min_value=0.0, max_value=1.0, flags=ValueBarFlags.NONE):
    is_horizontal = not (flags & ValueBarFlags.VERTICAL)
    draw_list = imgui.get_window_draw_list()
    cursor_pos = imgui.get_cursor_screen_pos()
    fraction = (value - min_value) / (max_value - min_value)
    frame_height = imgui.get_frame_height()
    label_size = imgui.calc_text_size(label) if label else ImVec2(0, 0)
    
    if is_horizontal:
        rect_size = ImVec2(size.x if size.x > 0 else imgui.calc_item_width(), frame_height)
        rect_start = cursor_pos
        fill_start = rect_start  # Define fill_start for horizontal orientation
        fill_end = ImVec2(rect_start.x + rect_size.x * fraction, rect_start.y + rect_size.y)
        rounding_corners = imgui.ImDrawFlags_.round_corners_left
    else:
        rect_size = ImVec2(frame_height * 2, size.y if size.y > 0 else size.y - label_size.y)
        rect_start = ImVec2(cursor_pos.x + max(0.0, (label_size.x - rect_size.x) / 2), cursor_pos.y)
        fill_start = ImVec2(rect_start.x, rect_start.y + rect_size.y * (1 - fraction))
        fill_end = ImVec2(rect_start.x + rect_size.x, rect_start.y + rect_size.y)
        rounding_corners = imgui.ImDrawFlags_.round_corners_bottom
    
    # Draw background
    draw_list.add_rect_filled(
        rect_start,
        ImVec2(rect_start.x + rect_size.x, rect_start.y + rect_size.y),
        imgui.get_color_u32(imgui.Col_.frame_bg),
        imgui.get_style().frame_rounding
    )
    
    # Draw filled portion
    draw_list.add_rect_filled(
        fill_start, fill_end,
        imgui.get_color_u32(imgui.Col_.plot_histogram),
        imgui.get_style().frame_rounding,
        rounding_corners
    )
    
    # Draw value text
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
data = pd.read_csv(file_name, delimiter=',', decimal='.')

# Ensure proper column names
data.columns = [
    'Timestamp', 'Suspension1', 'Suspension2', 'Suspension3', 'Suspension4',
    'Steering', 'RPM_FrontRight', 'RPM_FrontLeft', 'RPM_RearRight', 'RPM_RearLeft',
    'Theta', 'Derajat'
]

for col in ['Suspension1', 'Suspension2', 'Suspension3', 'Suspension4']:
    data[col] = (data[col] - 26139)/-334.73
    data[col] = data[col] - data[col].iloc[0:1000].mean()  # adjust reference

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

print("FILE NAME : "+f'{file_name}')

slider_value = [0]
play_mode = [False]
sample_rate = [2.0]  # frames per second
time_state = {"last_time": time.time()}

def main():
    x_values = np.array(data.index, dtype=np.float32)
    y1_values = data['CMA_RPM_FrontRight'].to_numpy(dtype=np.float32)
    y2_values = data['CMA_RPM_FrontLeft'].to_numpy(dtype=np.float32)
    y3_values = data['CMA_RPM_RearRight'].to_numpy(dtype=np.float32)
    y4_values = data['CMA_RPM_RearLeft'].to_numpy(dtype=np.float32)
    y5_values = data['Derajat'].to_numpy(dtype=np.float32)

    def gui():
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
        value_bar("Derajat", derajat_value, ImVec2(window_width, 400),
                  min_value=-60, max_value=60, flags=ValueBarFlags.NONE)
        imgui.end()

        # Display suspension values as vertical bars
        suspension_labels = ['Suspension1', 'Suspension2', 'Suspension3', 'Suspension4']
        for label in suspension_labels:
            imgui.begin(f"{label} Window")
            value = data[label].iloc[slider_value[0]]
            value_bar(label, value, ImVec2(50, 500), min_value=-10, max_value=10, flags=ValueBarFlags.VERTICAL)
            imgui.end()

        # Display speeds as vertical bars
        speed_labels = ['CMA_RPM_FrontRight', 'CMA_RPM_FrontLeft', 'CMA_RPM_RearRight', 'CMA_RPM_RearLeft']
        for label in speed_labels:
            imgui.begin(f"{label} Window")
            value = data[label].iloc[slider_value[0]]
            value_bar(label, value, ImVec2(50, 500), min_value=-100, max_value=1370, flags=ValueBarFlags.VERTICAL)
            imgui.end()

    immapp.run(gui, with_implot=True, with_markdown=True, window_size=(600, 400))

if __name__ == "__main__":
    main()