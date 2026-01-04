import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import queue

# ========= CONFIG =========
PORT = 'COM9'       # <--- CHECK YOUR PORT
BAUD_RATE = 115200
WINDOW_SIZE = 6000
CHUNK_SIZE = 50     # How many samples to send per animation frame
# ==========================

# Queues for safe thread communication
data_queue = queue.Queue()
serial_queue = queue.Queue()
status_queue = queue.Queue()

def generate_ecg_data(condition):
    """Load data from our trusted text files"""
    filename = 'apnea_data.txt' if condition == 'APNEA' else 'normal_data.txt'
    try:
        with open(filename, 'r') as f:
            raw = f.read().replace('\n', ',')
            data = [float(x) for x in raw.split(',') if x.strip()]
        # Pad or truncate to 6000
        if len(data) > WINDOW_SIZE: data = data[:WINDOW_SIZE]
        else: data += [0.0] * (WINDOW_SIZE - len(data))
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return [0.0] * WINDOW_SIZE

# --- Serial Thread ---
def serial_worker():
    try:
        with serial.Serial(PORT, BAUD_RATE, timeout=1) as ser:
            time.sleep(2) # Wait for reset
            print(f"🔌 Connected to {PORT}")
            
            while True:
                # 1. Get next patient case from the main thread
                condition = serial_queue.get() 
                print(f"\n--- STARTING {condition} TEST ---")
                
                # 2. Wait for STM32 Ready
                ready = False
                while not ready:
                    line = ser.readline().decode(errors='ignore').strip()
                    if "READY" in line:
                        ready = True
                
                # 3. Load Data
                data = generate_ecg_data(condition)
                
                # 4. Stream Data & Update Plot Queue
                for i in range(0, len(data), CHUNK_SIZE):
                    chunk = data[i:i+CHUNK_SIZE]
                    
                    # Send to STM32
                    for val in chunk:
                        ser.write(f"{val:.8f}\n".encode())
                    
                    # Send to Plotter (for visualization)
                    data_queue.put(chunk)
                    
                    # SLOW DOWN: Change 0.01 to 0.05
                    time.sleep(0.05)

                print(" Data Sent. Waiting for result...")
                
                # 5. Get Result
                while True:
                    line = ser.readline().decode(errors='ignore').strip()
                    
                    if "HIL Result" in line:
                        print(f"Result: {line}")
                        
                        # Parse the score
                        try:
                            score = float(line.split(":")[-1].strip())
                            if score > 0.5:
                                status_queue.put("APNEA")
                            else:
                                status_queue.put("NORMAL")
                        except:
                            pass
                        break
                        
    except Exception as e:
        print(f"Serial Error: {e}")

# --- Plotter Class ---
class LiveECGPlot:
    def __init__(self):
        self.buffer = np.zeros(WINDOW_SIZE)
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        
        # Initialize Line (Green by default)
        self.line, = self.ax.plot(self.buffer, color='#00ff00', lw=1.5)
        
        # Initialize Status Text (Top Right Corner)
        self.status_text = self.ax.text(0.95, 0.90, "WAITING...", 
                                        transform=self.ax.transAxes,
                                        color='white', fontsize=14, 
                                        fontweight='bold', ha='right')
        
        self.ax.set_ylim(-5, 8)
        self.ax.set_title("Live HIL Apnea Monitor")
        self.ax.grid(True, alpha=0.2)
        
        # Dark Mode Styling
        self.ax.set_facecolor('#1e1e1e')
        self.fig.patch.set_facecolor('#1e1e1e')
        self.ax.tick_params(colors='gray')
        self.ax.spines['bottom'].set_color('gray')
        self.ax.spines['top'].set_color('gray')
        self.ax.spines['left'].set_color('gray')
        self.ax.spines['right'].set_color('gray')
        self.ax.title.set_color('white')

    def update(self, frame):
        # 1. Update Data (Scrolling)
        new_data = []
        while not data_queue.empty():
            new_data.extend(data_queue.get())
        
        if new_data:
            n = len(new_data)
            self.buffer = np.roll(self.buffer, -n)
            self.buffer[-n:] = new_data
            self.line.set_ydata(self.buffer)
        
        # 2. Update Status (Check for new diagnosis)
        if not status_queue.empty():
            diagnosis = status_queue.get()
            
            if diagnosis == "APNEA":
                self.line.set_color('#ff3333')  # Red Line
                self.status_text.set_text("⚠️ APNEA DETECTED")
                self.status_text.set_color('#ff3333')
            else:
                self.line.set_color('#00ff00')  # Green Line
                self.status_text.set_text(" NORMAL RHYTHM")
                self.status_text.set_color('#00ff00')
            
        return self.line, self.status_text

# --- Main ---
if __name__ == "__main__":
    # Start Serial Thread
    t = threading.Thread(target=serial_worker, daemon=True)
    t.start()
    
    # Queue up a sequence of tests
    serial_queue.put("NORMAL")
    serial_queue.put("APNEA")
    serial_queue.put("NORMAL")

    # Start Animation
    print("🚀 Opening Plotter...")
    viz = LiveECGPlot()
    ani = FuncAnimation(viz.fig, viz.update, interval=50, blit=True, cache_frame_data=False)
    plt.show()