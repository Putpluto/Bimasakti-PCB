import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter
from imgui_bundle import imgui, implot, implot3d, imgui_md, immapp, ImVec2, ImVec4

# Load the data
file_name = "log_179withDegree.csv"
data = pd.read_csv(file_name, delimiter=',', decimal='.')

# Ensure proper column names
data.columns = ['Timestamp', 'Suspension1', 'Suspension2', 'Suspension3', 'Suspension4',
                'Steering', 'RPM_FrontRight', 'RPM_FrontLeft', 'RPM_RearRight', 'RPM_RearLeft',
                'Theta', 'Derajat']

# Zero out all outliers above 5000 in RPM columns
data.loc[data['RPM_FrontRight'] > 1370, 'RPM_FrontRight'] = 0
data.loc[data['RPM_FrontLeft'] > 1370, 'RPM_FrontLeft'] = 0
data.loc[data['RPM_RearRight'] > 1370, 'RPM_RearRight'] = 0
data.loc[data['RPM_RearLeft'] > 1370, 'RPM_RearLeft'] = 0
# Scale 'Derajat'
data['Derajat'] = data['Derajat'] * 10

# Median Filter
N = 17  # Kernel size
data['CMA_RPM_FrontRight'] = medfilt(data['RPM_FrontRight'], kernel_size=N)
data['CMA_RPM_FrontLeft'] = medfilt(data['RPM_FrontLeft'], kernel_size=N)
data['CMA_RPM_RearRight'] = medfilt(data['RPM_RearRight'], kernel_size=N)
data['CMA_RPM_RearLeft'] = medfilt(data['RPM_RearLeft'], kernel_size=N)

# Apply Savitzky-Golay filter for final smoothing
data['CMA_RPM_FrontRight'] = savgol_filter(data['CMA_RPM_FrontRight'], window_length=11, polyorder=2)
data['CMA_RPM_FrontLeft'] = savgol_filter(data['CMA_RPM_FrontLeft'], window_length=11, polyorder=2)
data['CMA_RPM_RearRight'] = savgol_filter(data['CMA_RPM_RearRight'], window_length=11, polyorder=2)
data['CMA_RPM_RearLeft'] = savgol_filter(data['CMA_RPM_RearLeft'], window_length=11, polyorder=2)

data['KMPH_FrontRight'] = data['CMA_RPM_FrontRight'] * 0.0766
data['KMPH_FrontLeft'] = data['CMA_RPM_FrontLeft'] * 0.0766

def create_histogram(data, column, bins, color, title):
    counts, edges = np.histogram(data[column], bins=bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    mean_rpm = np.sum(counts * bin_centers) / np.sum(counts)
    
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.bar(edges[:-1], counts, color=color, width=100, alpha=1, label=column)
    ax.set_xlabel('RPM Range')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    
    canvas.draw()
    return canvas, mean_rpm

def create_rpm_plot(data):
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    
    ax.plot(data.index, data['RPM_FrontRight'], label='RPM_FrontRight (Raw)', color='darkorange', alpha=0)
    ax.plot(data.index, data['CMA_RPM_FrontRight'], label='CMA RPM_FrontRight', color='gold', alpha=0.9)
    ax.plot(data.index, data['RPM_FrontLeft'], label='RPM_FrontLeft (Raw)', color='deepskyblue', alpha=0)
    ax.plot(data.index, data['CMA_RPM_FrontLeft'], label='CMA RPM_FrontLeft', color='green', alpha=0.9)
    ax.plot(data.index, data['RPM_RearRight'], label='RPM_RearRight (Raw)', color='turquoise', alpha=0.2)
    ax.plot(data.index, data['CMA_RPM_RearRight'], label='CMA RPM_RearRight', color='blue', alpha=0.9)
    ax.plot(data.index, data['RPM_RearLeft'], label='RPM_RearLeft (Raw)', color='violet', alpha=0)
    ax.plot(data.index, data['CMA_RPM_RearLeft'], label='CMA RPM_RearLeft', color='red', alpha=0.9)
    ax.plot(data.index, data['Derajat'], label='Derajat CW is increment', color='black', alpha=1)
    ax.plot(data.index, data['KMPH_FrontRight'], linestyle='--', label='KMPH_FrontRight', color='orangered', alpha=0.9)
    ax.plot(data.index, data['KMPH_FrontLeft'], linestyle='--', label='KMPH_FrontLeft', color='gold', alpha=0.9)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Index')
    ax.set_ylabel('RPM')
    ax.set_title(f'{file_name} RPM Data with Centered Moving Average')
    ax.legend()
    
    canvas.draw()
    return canvas

def main():
    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    window = glfw.create_window(1280, 720, "RPM Data Analysis", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    imgui.create_context()
    impl = GlfwRenderer(window)

    bins = range(1, 1370, 100)
    histograms = {
        'FrontRight': create_histogram(data, 'CMA_RPM_FrontRight', bins, 'red', f'{file_name} Histogram of CMA_RPM_FrontRight'),
        'FrontLeft': create_histogram(data, 'CMA_RPM_FrontLeft', bins, 'blue', f'{file_name} Histogram of CMA_RPM_FrontLeft'),
        'RearRight': create_histogram(data, 'CMA_RPM_RearRight', bins, 'green', f'{file_name} Histogram of CMA_RPM_RearRight'),
        'RearLeft': create_histogram(data, 'CMA_RPM_RearLeft', bins, 'purple', f'{file_name} Histogram of CMA_RPM_RearLeft')
    }
    rpm_plot = create_rpm_plot(data)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()

        imgui.new_frame()

        imgui.begin("RPM Data Analysis", True)
        imgui.text(f"File Name: {file_name}")

        for key, (canvas, mean_rpm) in histograms.items():
            imgui.text(f"Average RPM_{key} (whole dataset excluding zero): {mean_rpm:.2f}")
            imgui.image(canvas.renderer.buffer_id, canvas.get_width_height()[0], canvas.get_width_height()[1])

        imgui.image(rpm_plot.renderer.buffer_id, rpm_plot.get_width_height()[0], rpm_plot.get_width_height()[1])

        imgui.end()

        gl.glClearColor(0.45, 0.55, 0.60, 1.00)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()

if __name__ == "__main__":
    main()