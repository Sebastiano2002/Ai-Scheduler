import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
import subprocess
import threading
import os
import glob
import sys
import queue
import csv

# Set appearance and theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class SmartSchedulerGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SmartScheduler Pipeline")
        self.geometry("1200x800")
        
        # Grid layout configuration
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Application state
        self.case_var = ctk.StringVar(value="A")
        self.is_running = False
        
        self.sequential_mode_active = False
        self.f3_done_in_session = False
        
        self.setup_ui()
        self.update_ui_state()
        
        self.log_queue = queue.Queue()
        self.check_queue()

    def setup_ui(self):
        # ----------------- Left Panel (Controls) -----------------
        self.left_panel = ctk.CTkFrame(self, width=280, corner_radius=10)
        self.left_panel.grid(row=0, column=0, rowspan=2, padx=15, pady=15, sticky="nsew")
        self.left_panel.grid_rowconfigure(7, weight=1) # The file list takes the remaining space

        # App Title
        self.app_label = ctk.CTkLabel(self.left_panel, text="SmartScheduler", font=ctk.CTkFont(size=20, weight="bold"))
        self.app_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Case Selection
        self.case_label = ctk.CTkLabel(self.left_panel, text="Seleziona Caso d'Uso", font=ctk.CTkFont(weight="bold"))
        self.case_label.grid(row=1, column=0, padx=20, pady=(10, 5), sticky="w")
        
        self.radio_a = ctk.CTkRadioButton(self.left_panel, text="Caso A", variable=self.case_var, value="A", command=self.on_case_changed)
        self.radio_a.grid(row=2, column=0, padx=20, pady=5, sticky="w")
        
        self.radio_b = ctk.CTkRadioButton(self.left_panel, text="Caso B", variable=self.case_var, value="B", command=self.on_case_changed)
        self.radio_b.grid(row=3, column=0, padx=20, pady=5, sticky="w")
        
        self.radio_all = ctk.CTkRadioButton(self.left_panel, text="Entrambi (A poi B)", variable=self.case_var, value="all", command=self.on_case_changed)
        self.radio_all.grid(row=4, column=0, padx=20, pady=5, sticky="w")

        # Buttons
        self.btn_run_all = ctk.CTkButton(self.left_panel, text="▶ Esegui tutta la pipeline", command=self.run_all, fg_color="#2E8B57", hover_color="#3CB371")
        self.btn_run_all.grid(row=5, column=0, padx=20, pady=(20, 10))

        self.buttons_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.buttons_frame.grid(row=6, column=0, padx=20, pady=5, sticky="ew")
        
        self.btn_phase_1 = ctk.CTkButton(self.buttons_frame, text="Fase 1 (Workers)", command=self.run_phase_1)
        self.btn_phase_1.pack(fill="x", pady=4)

        self.btn_phase_2 = ctk.CTkButton(self.buttons_frame, text="Fase 2 (Drafting)", command=self.run_phase_2)
        self.btn_phase_2.pack(fill="x", pady=4)

        self.btn_phase_3 = ctk.CTkButton(self.buttons_frame, text="Fase 3 (Verifica)", command=self.run_phase_3)
        self.btn_phase_3.pack(fill="x", pady=4)

        self.btn_phase_4 = ctk.CTkButton(self.buttons_frame, text="Fase 4 (Raffinamento)", command=self.run_phase_4)
        self.btn_phase_4.pack(fill="x", pady=4)
        
        self.btn_reset = ctk.CTkButton(self.buttons_frame, text="⟲ Reset / Ricomincia", command=self.reset_case, fg_color="#C27A23", hover_color="#DA9845")
        self.btn_reset.pack(fill="x", pady=(10, 4))
        
        self.btn_stop = ctk.CTkButton(self.buttons_frame, text="⏹ Ferma Esecuzione", command=self.stop_execution, fg_color="#8B0000", hover_color="#A52A2A", state="disabled")
        self.btn_stop.pack(fill="x", pady=4)

        # File List using Listbox
        file_header_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        file_header_frame.grid(row=7, column=0, padx=20, pady=(15, 0), sticky="ew")
        file_header_frame.grid_columnconfigure(0, weight=1)

        file_list_label = ctk.CTkLabel(file_header_frame, text="File Generati", font=ctk.CTkFont(weight="bold"))
        file_list_label.grid(row=0, column=0, sticky="w")
        
        self.btn_refresh_files = ctk.CTkButton(file_header_frame, text="🔄", width=28, height=24, command=self.refresh_file_list, fg_color="#444444", hover_color="#555555")
        self.btn_refresh_files.grid(row=0, column=1, sticky="e")
        
        self.file_listbox = tk.Listbox(self.left_panel, bg="#2B2B2B", fg="#FFFFFF", selectbackground="#1F6AA5", font=("Segoe UI", 10), borderwidth=0, highlightthickness=0)
        self.file_listbox.grid(row=8, column=0, padx=20, pady=(5, 20), sticky="nsew")
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)

        # ----------------- Right Panel (Viewers) -----------------
        self.right_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.right_panel.grid(row=0, column=1, padx=(0, 15), pady=15, sticky="nsew")
        
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(0, weight=1) # File viewer takes 50%
        self.right_panel.grid_rowconfigure(1, weight=1) # Log viewer takes 50%

        # TOP: File Viewer
        self.file_viewer_frame = ctk.CTkFrame(self.right_panel, corner_radius=10)
        self.file_viewer_frame.grid(row=0, column=0, pady=(0, 10), sticky="nsew")
        self.file_viewer_frame.grid_columnconfigure(0, weight=1)
        self.file_viewer_frame.grid_rowconfigure(1, weight=1)

        self.lbl_viewer = ctk.CTkLabel(self.file_viewer_frame, text="Visualizzatore File (Buffer)", font=ctk.CTkFont(weight="bold"))
        self.lbl_viewer.grid(row=0, column=0, padx=15, pady=(10, 0), sticky="w")

        self.file_viewer = ctk.CTkTextbox(self.file_viewer_frame, font=("Consolas", 12), text_color="#A6E3A1", wrap="none")
        self.file_viewer.grid(row=1, column=0, padx=15, pady=15, sticky="nsew")

        # BOTTOM: Log Viewer
        self.log_viewer_frame = ctk.CTkFrame(self.right_panel, corner_radius=10)
        self.log_viewer_frame.grid(row=1, column=0, sticky="nsew")
        self.log_viewer_frame.grid_columnconfigure(0, weight=1)
        self.log_viewer_frame.grid_rowconfigure(1, weight=1)

        self.lbl_logs = ctk.CTkLabel(self.log_viewer_frame, text="Terminale (Log)", font=ctk.CTkFont(weight="bold"))
        self.lbl_logs.grid(row=0, column=0, padx=15, pady=(10, 0), sticky="w")

        self.log_viewer = ctk.CTkTextbox(self.log_viewer_frame, font=("Consolas", 12), text_color="#CDD6F4", wrap="none")
        self.log_viewer.grid(row=1, column=0, padx=15, pady=15, sticky="nsew")

        # ----------------- Bottom Status Bar -----------------
        self.status_frame = ctk.CTkFrame(self, height=30, corner_radius=0, fg_color="transparent")
        self.status_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=15, pady=(0, 10))
        
        self.lbl_status = ctk.CTkLabel(self.status_frame, text="Pronto.", font=ctk.CTkFont(size=12, slant="italic"))
        self.lbl_status.pack(side="left")

        self.progress_bar = ctk.CTkProgressBar(self.status_frame, width=300)
        self.progress_bar.pack(side="right", padx=10)
        self.progress_bar.set(0)

    def on_case_changed(self):
        if self.is_running:
            return
        self.sequential_mode_active = False
        self.f3_done_in_session = False
        self.update_ui_state()

    def get_expected_files(self, case_label):
        labels = ["A", "B"] if case_label == "all" else [case_label]
        files = []
        for l in labels:
            files.extend([
                f"formalized_preferences_case_{l}.py",
                f"draft_code_case_{l}.txt",
                f"schedule_case_{l}.csv",
                f"final_code_case_{l}.txt",
                f"schedule_case_{l}_final.csv",
            ])
        return files

    def update_ui_state(self):
        if self.is_running:
            self.btn_run_all.configure(state="disabled")
            self.btn_phase_1.configure(state="disabled")
            self.btn_phase_2.configure(state="disabled")
            self.btn_phase_3.configure(state="disabled")
            self.btn_phase_4.configure(state="disabled")
            self.btn_reset.configure(state="disabled")
            self.btn_stop.configure(state="normal")
            return

        case_label = self.case_var.get()
        if case_label == "all":
            f1_done = os.path.exists("formalized_preferences_case_A.py") and os.path.exists("formalized_preferences_case_B.py")
            f2_done = os.path.exists("draft_code_case_A.txt") and os.path.exists("schedule_case_A.csv") and os.path.exists("draft_code_case_B.txt") and os.path.exists("schedule_case_B.csv")
            f4_done = os.path.exists("schedule_case_A_final.csv") and os.path.exists("schedule_case_B_final.csv")
        else:
            f1_done = os.path.exists(f"formalized_preferences_case_{case_label}.py")
            f2_done = os.path.exists(f"draft_code_case_{case_label}.txt") and os.path.exists(f"schedule_case_{case_label}.csv")
            f4_done = os.path.exists(f"schedule_case_{case_label}_final.csv")
        
        self.btn_phase_1.configure(state="normal" if not f1_done else "disabled")
        self.btn_phase_2.configure(state="normal" if (f1_done and not f2_done) else "disabled")
        self.btn_phase_3.configure(state="normal" if (f2_done and not f4_done) else "disabled")
        self.btn_phase_4.configure(state="normal" if (f2_done and self.f3_done_in_session and not f4_done) else "disabled")

        if self.sequential_mode_active or f1_done:
            self.btn_run_all.configure(state="disabled")
        else:
            self.btn_run_all.configure(state="normal")
            
        self.btn_reset.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.refresh_file_list()

    def reset_case(self):
        if self.is_running: return
        case_label = self.case_var.get()
        msg = f"Vuoi cancellare tutti i file generati per {'entrambi i casi' if case_label == 'all' else 'il Caso ' + case_label} e ricominciare?"
        if messagebox.askyesno("Conferma Reset", msg):
            for f in self.get_expected_files(case_label):
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except Exception as e:
                        print(f"Errore rimozione {f}: {e}")
            self.sequential_mode_active = False
            self.f3_done_in_session = False
            self.file_viewer.delete("0.0", "end")
            self.log_viewer.delete("0.0", "end")
            self.lbl_viewer.configure(text="Visualizzatore File")
            self.update_ui_state()
            self.lbl_status.configure(text=f"{'Entrambi i casi resettati' if case_label == 'all' else 'Caso ' + case_label + ' resettato'}.")

    def refresh_file_list(self):
        case_label = self.case_var.get()
        self.file_listbox.delete(0, tk.END)
        for f in self.get_expected_files(case_label):
            if os.path.exists(f):
                self.file_listbox.insert(tk.END, f)
        for f in glob.glob("*.log"):
             self.file_listbox.insert(tk.END, f)

    def on_file_select(self, event):
        if not self.file_listbox.curselection():
            return
        index = self.file_listbox.curselection()[0]
        filename = self.file_listbox.get(index)
        
        self.lbl_viewer.configure(text=f"File: {filename}")
        self.file_viewer.delete("0.0", "end")
        
        try:
            if filename.endswith('.csv'):
                with open(filename, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    data = list(reader)
                
                if not data:
                    self.file_viewer.insert("0.0", "Il file CSV è vuoto.")
                else:
                    col_widths = [max(len(str(item)) for item in col) for col in zip(*data)]
                    formatted_rows = []
                    for i, row in enumerate(data):
                        formatted_row = " | ".join(str(item).ljust(width) for item, width in zip(row, col_widths))
                        formatted_rows.append(formatted_row)
                        if i == 0:
                            separator = "-+-".join("-" * width for width in col_widths)
                            formatted_rows.append(separator)
                    self.file_viewer.insert("0.0", "\n".join(formatted_rows))
            else:
                with open(filename, "r", encoding="utf-8") as f:
                    content = f.read()
                self.file_viewer.insert("0.0", content)
        except Exception as e:
            self.file_viewer.insert("0.0", f"Impossibile leggere il file:\n{e}")

    # --- Runner Logic ---
    def start_execution(self, command, status_message, is_sequential=True, phase=None):
        if self.is_running: return
        self.is_running = True
        if is_sequential:
            self.sequential_mode_active = True
            
        self.update_ui_state()
        self.lbl_status.configure(text=status_message)
        self.progress_bar.start()
        
        # Svuota il terminale dei log ad ogni esecuzione
        self.log_viewer.delete("0.0", "end")
        self.log_viewer.insert("0.0", f"--- {status_message} ---\n\n")

        def run_thread():
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUNBUFFERED"] = "1"
            
            try:
                self.current_process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    bufsize=1,
                    universal_newlines=True
                )
                
                for line in iter(self.current_process.stdout.readline, ''):
                    self.log_queue.put(("log", line))
                self.current_process.stdout.close()
                return_code = self.current_process.wait()
                self.log_queue.put(("done", (return_code, phase)))
                
            except Exception as e:
                self.log_queue.put(("log", f"Eccezione durante l'esecuzione:\n{e}\n"))
                self.log_queue.put(("done", (-1, phase)))

        threading.Thread(target=run_thread, daemon=True).start()

    def stop_execution(self):
        if self.is_running and hasattr(self, 'current_process') and self.current_process:
            self.log_queue.put(("log", "\n[!] Interruzione forzata dall'utente...\n"))
            self.current_process.terminate()
            self.btn_stop.configure(state="disabled")

    def check_queue(self):
        try:
            while True:
                msg_type, data = self.log_queue.get_nowait()
                if msg_type == "log":
                    self.log_viewer.insert("end", data)
                    self.log_viewer.see("end")
                    
                    # Estrae l'avanzamento per aggiornare la status bar dinamicamente
                    line = data.strip()
                    if line.startswith("# PIPELINE | FASE") or line.startswith("# USE CASE") or line.startswith("PIPELINE COMPLETATA"):
                        self.lbl_status.configure(text=line.strip('# ='))
                    elif line.startswith("[*]") or line.startswith("[+]") or line.startswith("[-]"):
                        short_line = line if len(line) < 90 else line[:87] + "..."
                        self.lbl_status.configure(text=short_line)
                        
                elif msg_type == "done":
                    return_code, phase = data
                    self.is_running = False
                    self.progress_bar.stop()
                    self.progress_bar.set(1 if return_code == 0 else 0)
                    
                    if return_code == 0:
                        self.lbl_status.configure(text="Esecuzione completata con successo.")
                        if phase == 3:
                            self.f3_done_in_session = True
                    else:
                        self.lbl_status.configure(text=f"Esecuzione terminata con errori (codice: {return_code}).")
                    self.update_ui_state()
        except queue.Empty:
            pass
        finally:
            self.after(100, self.check_queue)

    def run_all(self):
        case_label = self.case_var.get()
        cmd = [sys.executable, "run_pipeline.py", "--case", case_label]
        msg = "Esecuzione completa pipeline Entrambi i Casi..." if case_label == "all" else f"Esecuzione completa pipeline Caso {case_label}..."
        self.start_execution(cmd, msg, is_sequential=False)

    def run_phase_1(self):
        case_label = self.case_var.get()
        cmd = [sys.executable, "Fase1_workers_agent.py", "--case", case_label]
        msg = "Esecuzione Fase 1 (Entrambi i Casi)..." if case_label == "all" else f"Esecuzione Fase 1 (Caso {case_label})...."
        self.start_execution(cmd, msg, phase=1)

    def run_phase_2(self):
        case_label = self.case_var.get()
        cmd = [sys.executable, "Fase2_drafting_agent.py", "--case", case_label]
        msg = "Esecuzione Fase 2 (Entrambi i Casi)..." if case_label == "all" else f"Esecuzione Fase 2 (Caso {case_label})...."
        self.start_execution(cmd, msg, phase=2)

    def run_phase_3(self):
        case_label = self.case_var.get()
        cmd = [sys.executable, "Fase3_verification_agent.py", "--case", case_label, "--from-csv"]
        msg = "Esecuzione Fase 3 (Entrambi i Casi)..." if case_label == "all" else f"Esecuzione Fase 3 (Caso {case_label})...."
        self.start_execution(cmd, msg, phase=3)

    def run_phase_4(self):
        case_label = self.case_var.get()
        cmd = [sys.executable, "Fase4_refinement_agent.py", "--case", case_label, "--from-draft"]
        msg = "Esecuzione Fase 4 (Entrambi i Casi)..." if case_label == "all" else f"Esecuzione Fase 4 (Caso {case_label})...."
        self.start_execution(cmd, msg, phase=4)

if __name__ == "__main__":
    app = SmartSchedulerGUI()
    app.mainloop()
