import time
import serial

# Initialize serial communication
try:
    ser = serial.Serial('COM3', 230400, timeout=1)  # Ensure COM3 is the correct port
    time.sleep(2)  # Wait for the serial connection to initialize
    print("Serial port COM3 opened successfully.")
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    exit()

print("TermissonBot starting at 30Hz...")

# Print header row
header = ("Timestamp,Suspension1,Suspension2,Suspension3,Suspension4,"
          "Steering,RPM_FrontRight,RPM_FrontLeft,RPM_RearRight,RPM_RearLeft,"
          "Theta,Derajat,TPS,GPSS,RPM")
print(header)

# Constants
SUSPENSION_MIN = 30
SUSPENSION_MAX = 26139
SUSPENSION_STEP = 50  # Change in suspension value per update
SAMPLE_INTERVAL = 1 / 22.0  # 30Hz update rate

# Initial suspension values and directions
suspension_values = [SUSPENSION_MIN] * 4
suspension_directions = [1] * 4  # 1 for increasing, -1 for decreasing

# Initial time
t = 0.0

try:
    while True:
        start_time = time.time()

        # Update suspension values
        for i in range(4):
            suspension_values[i] += suspension_directions[i] * SUSPENSION_STEP
            # Reverse direction if limits are reached
            if suspension_values[i] >= SUSPENSION_MAX:
                suspension_values[i] = SUSPENSION_MAX
                suspension_directions[i] = -10
            elif suspension_values[i] <= SUSPENSION_MIN:
                suspension_values[i] = SUSPENSION_MIN
                suspension_directions[i] = 10

        # Steering (smoother transitions, ±60 degrees)
        steering = (t % 120) - 60  # Cycles between -60 and 60

        # Wheel RPMs (more realistic values and relationships)
        base_rpm = abs((t % 20) - 10) * 100  # Cycles between 0 and 1000
        rpm_variations = [0.9, 1.0, 1.1, 1.05]  # Slight variations for each wheel
        rpm_front_right = base_rpm * rpm_variations[0]
        rpm_front_left = base_rpm * rpm_variations[1]
        rpm_rear_right = base_rpm * rpm_variations[2]
        rpm_rear_left = base_rpm * rpm_variations[3]

        # Vehicle orientation (smoother changes)
        theta = (t % 90) - 45  # Cycles between -45 and 45 degrees
        derajat = steering  # Match steering angle

        # Throttle Position Sensor (0-5V, smoother transitions)
        tps = ((t % 10) / 10) * 5  # Cycles between 0 and 5

        # GPS Speed (more realistic acceleration/deceleration)
        gpss = abs((t % 240) - 120) * 0.5  # Cycles between 0 and 60 km/h

        # Engine RPM (smoother transitions, realistic range)
        rpm = 1000 + abs((t % 20) - 10) * 1250  # Cycles between 1000 and 13500 RPM

        # Create CSV-formatted string
        row = (f"{t:.2f},{suspension_values[0]},{suspension_values[1]},{suspension_values[2]},{suspension_values[3]},"
               f"{steering:.1f},{rpm_front_right:.1f},{rpm_front_left:.1f},{rpm_rear_right:.1f},{rpm_rear_left:.1f},"
               f"{theta:.1f},{derajat:.1f},{tps:.2f},{gpss:.1f},{rpm:.1f}\n")

        # Send to serial port
        ser.write(row.encode('utf-8'))

        # Print to console (optional)
        print(row.strip())

        # Calculate sleep time to maintain 30Hz
        elapsed = time.time() - start_time
        sleep_time = max(0, SAMPLE_INTERVAL - elapsed)
        time.sleep(sleep_time)

        # Increment time
        t += SAMPLE_INTERVAL

except KeyboardInterrupt:
    print("\nTermissonBot stopped by user.")

finally:
    if ser.is_open:
        ser.close()
        print("Serial port COM3 closed.")
