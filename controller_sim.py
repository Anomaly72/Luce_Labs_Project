import time
import random
import threading
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque

class RealTimeEnclosureSim:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Enclosure Control Simulator")
        self.root.geometry("1100x750")

        # Track scheduled GUI callbacks for clean exit
        self.after_id = None
        self.is_running = True

        # --- THREAD LOCK FOR DATA SAFETY ---
        self.data_lock = threading.Lock()

        # --- SIMULATION STATE ---
        self.current_temp = 22.0
        self.current_hum = 55.0
        self.prev_temp = 22.0
        self.prev_hum = 55.0
        self.fan_state = False  # False = OFF, True = ON
        self.last_state_change = 0.0
        self.start_time = time.time()
        
        # Environmental rates
        self.ambient_temp = 16.0
        self.ambient_hum = 40.0
        self.heating_rate = 0.3    # °C/s from light source
        self.misting_rate = 0.5    # %/s from plants/mister
        self.cooling_rate = 0.6    # Exhaust fan cooling effect
        self.drying_rate = 0.8     # Exhaust fan drying effect

        # Data windows for live chart (stores up to 600 history points)
        self.max_points = 600
        self.time_history = deque(maxlen=self.max_points)
        self.temp_history = deque(maxlen=self.max_points)
        self.hum_history = deque(maxlen=self.max_points)
        self.fan_history = deque(maxlen=self.max_points)

        # --- CONTROL PARAMETERS (Tkinter Variables bound to sliders) ---
        self.target_temp = tk.DoubleVar(value=20.0)
        self.target_hum = tk.DoubleVar(value=60.0)
        self.temp_buffer = tk.DoubleVar(value=5.0)
        self.hum_buffer = tk.DoubleVar(value=5.0)
        self.short_cycle_delay = tk.DoubleVar(value=3.0)  # Seconds in real time

        # --- UI LAYOUT ---
        self._build_ui()

        # --- START THREADS ---
        self.sim_thread = threading.Thread(target=self._run_physics_loop, daemon=True)
        self.sim_thread.start()

        # Start GUI update loop (~20 FPS chart refresh)
        self._update_chart()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        # Left Panel: Sliders & Controls
        control_frame = ttk.LabelFrame(self.root, text=" Tuning Parameters ", padding=15)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Helper to create labeled sliders
        def add_slider(parent, label_text, var, from_, to_, resolution):
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=8)
            lbl = ttk.Label(frame, text=f"{label_text}: {var.get():.1f}")
            lbl.pack(anchor=tk.W)
            
            def on_scroll(val):
                lbl.config(text=f"{label_text}: {float(val):.1f}")
                
            slider = ttk.Scale(
                frame, from_=from_, to=to_, variable=var,
                command=on_scroll
            )
            slider.pack(fill=tk.X)
            return slider

        add_slider(control_frame, "Target Temp (°C)", self.target_temp, 10.0, 35.0, 0.5)
        add_slider(control_frame, "Temp Buffer (°C)", self.temp_buffer, 0.5, 10.0, 0.5)
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=5)
        
        add_slider(control_frame, "Target Humidity (%)", self.target_hum, 30.0, 90.0, 1.0)
        add_slider(control_frame, "Humidity Buffer (%)", self.hum_buffer, 0.5, 15.0, 0.5)
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=5)
        
        add_slider(control_frame, "Short Cycle Delay (s)", self.short_cycle_delay, 0.0, 10.0, 0.5)

        # Live Status Readouts
        status_frame = ttk.LabelFrame(control_frame, text=" Live Sensor Readings ", padding=10)
        status_frame.pack(fill=tk.X, pady=15)

        self.temp_readout = ttk.Label(status_frame, text="Filtered Temp: -- °C", font=("Helvetica", 10, "bold"))
        self.temp_readout.pack(anchor=tk.W)
        self.hum_readout = ttk.Label(status_frame, text="Filtered Hum: -- %", font=("Helvetica", 10, "bold"))
        self.hum_readout.pack(anchor=tk.W)
        self.fan_readout = ttk.Label(status_frame, text="Fan State: OFF", font=("Helvetica", 10, "bold"), foreground="gray")
        self.fan_readout.pack(anchor=tk.W, pady=(5, 0))

        # Right Panel: Matplotlib Figure
        chart_frame = ttk.Frame(self.root)
        chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig, (self.ax_temp, self.ax_hum) = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _run_physics_loop(self):
        """ Runs in a background thread at 10 Hz real-time ticks """
        dt = 0.1  # 100 ms step
        while self.is_running:
            current_time = time.time() - self.start_time

            # 1. Physics Engine
            if self.fan_state:
                # Cooling/Drying towards ambient conditions
                self.current_temp += (self.ambient_temp - self.current_temp) * self.cooling_rate * dt
                self.current_hum += (self.ambient_hum - self.current_hum) * self.drying_rate * dt
            else:
                # Heating from light source, humidity accumulation from mister/plants
                self.current_temp += self.heating_rate * dt
                self.current_hum += self.misting_rate * dt

            # Add raw sensor noise
            raw_temp = self.current_temp + random.uniform(-0.1, 0.1)
            raw_hum = self.current_hum + random.uniform(-0.2, 0.2)

            # 2. Moving Average Lowpass Filter: (previous + current) / 2
            filt_temp = (self.prev_temp + raw_temp) / 2.0
            filt_hum = (self.prev_hum + raw_hum) / 2.0
            self.prev_temp = filt_temp
            self.prev_hum = filt_hum

            # 3. Control Logic using Live Slider Values
            t_target = self.target_temp.get()
            h_target = self.target_hum.get()
            t_buf = self.temp_buffer.get()
            h_buf = self.hum_buffer.get()
            delay = self.short_cycle_delay.get()

            state_duration = current_time - self.last_state_change

            if self.fan_state and (filt_temp <= t_target - t_buf and filt_hum <= h_target - h_buf) and state_duration >= delay:
                self.fan_state = False
                self.last_state_change = current_time

            elif not self.fan_state and (filt_temp > t_target or filt_hum > h_target) and state_duration >= delay:
                self.fan_state = True
                self.last_state_change = current_time

            # Thread-safe appending to history
            with self.data_lock:
                self.time_history.append(current_time)
                self.temp_history.append(filt_temp)
                self.hum_history.append(filt_hum)
                self.fan_history.append(1.0 if self.fan_state else 0.0)

            time.sleep(dt)

    def _update_chart(self):
        """ Refreshes the GUI chart (~20 times per second) """
        if not self.is_running:
            return

        # Take a thread-safe snapshot of all history deques at once
        with self.data_lock:
            t = list(self.time_history)
            temps = list(self.temp_history)
            hums = list(self.hum_history)
            fans = list(self.fan_history)

        if t:
            window_size = 30.0  # Rolling 30-second time window
            current_time = t[-1]
            x_max = max(window_size, current_time)
            x_min = x_max - window_size

            # --- Temperature Plot ---
            self.ax_temp.clear()
            self.ax_temp.plot(t, temps, color='crimson', label='Temp (°C)')
            self.ax_temp.axhline(self.target_temp.get(), color='crimson', linestyle='--', alpha=0.6, label='Target')
            self.ax_temp.axhline(self.target_temp.get() - self.temp_buffer.get(), color='crimson', linestyle=':', alpha=0.4, label='Off Boundary')
            self.ax_temp.set_ylabel('Temp (°C)')
            self.ax_temp.set_xlim(x_min, x_max)
            self.ax_temp.legend(loc='upper left', fontsize='small')
            self.ax_temp.grid(True, alpha=0.3)

            # --- Humidity Plot ---
            self.ax_hum.clear()
            self.ax_hum.plot(t, hums, color='teal', label='Humidity (%)')
            self.ax_hum.axhline(self.target_hum.get(), color='teal', linestyle='--', alpha=0.6, label='Target')
            self.ax_hum.axhline(self.target_hum.get() - self.hum_buffer.get(), color='teal', linestyle=':', alpha=0.4, label='Off Boundary')
            
            # Overlay Fan state shading
            self.ax_hum.fill_between(t, 0, 100, where=[f == 1.0 for f in fans], color='gray', alpha=0.15)
            self.ax_hum.set_ylabel('Humidity (%)')
            self.ax_hum.set_xlabel('Time (Seconds)')
            self.ax_hum.set_xlim(x_min, x_max)
            self.ax_hum.legend(loc='upper left', fontsize='small')
            self.ax_hum.grid(True, alpha=0.3)

            # Update Readouts
            last_t = temps[-1]
            last_h = hums[-1]
            self.temp_readout.config(text=f"Filtered Temp: {last_t:.2f} °C")
            self.hum_readout.config(text=f"Filtered Hum: {last_h:.1f} %")
            if self.fan_state:
                self.fan_readout.config(text="Fan State: ON", foreground="green")
            else:
                self.fan_readout.config(text="Fan State: OFF", foreground="gray")

            self.canvas.draw_idle()

        # Schedule next tick if app is active
        if self.is_running:
            self.after_id = self.root.after(50, self._update_chart)

    def _on_close(self):
        """ Clean teardown sequence """
        self.is_running = False  # Stops physics loop & cancels future redraw ticks

        # Cancel any pending 'after' callbacks
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)

        # Release Matplotlib C-level GUI resources
        plt.close(self.fig)

        # Destroy Tkinter window
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeEnclosureSim(root)
    root.mainloop()