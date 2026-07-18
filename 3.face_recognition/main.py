import subprocess
import sys
import tkinter as tk
from tkinter import messagebox, ttk

SCRIPT_DIR = "."

processes = []


def run_register():
    """Launch register_cam.py in a subprocess and ask for the person's name."""

    name = _ask_name()

    if name is None:
        return

    _launch(["python", f"{SCRIPT_DIR}/register_cam.py", "--name", name])


def run_capture():
    """Launch webcam.py in a subprocess for live face recognition."""

    _launch(["python", f"{SCRIPT_DIR}/webcam.py"])


def _ask_name():
    """Open a small dialog to capture the person's name for registration."""

    dialog = tk.Toplevel(root)
    dialog.title("Register")
    dialog.resizable(False, False)
    dialog.transient(root)
    dialog.grab_set()

    tk.Label(
        dialog,
        text="Enter the person's name:",
        padx=10,
        pady=10
    ).pack()

    entry = ttk.Entry(dialog, width=30)
    entry.pack(padx=10, pady=(0, 10))
    entry.focus_set()

    result = {"value": None}

    def confirm():
        value = entry.get().strip()

        if not value:
            messagebox.showwarning(
                "Input required",
                "Please enter a name."
            )
            return

        result["value"] = value
        dialog.destroy()

    def cancel():
        dialog.destroy()

    button_frame = ttk.Frame(dialog)
    button_frame.pack(pady=(0, 10))

    ttk.Button(
        button_frame,
        text="OK",
        command=confirm
    ).pack(side="left", padx=5)

    ttk.Button(
        button_frame,
        text="Cancel",
        command=cancel
    ).pack(side="left", padx=5)

    dialog.bind("<Return>", lambda _event: confirm())
    dialog.bind("<Escape>", lambda _event: cancel())

    root.wait_window(dialog)

    return result["value"]


def _launch(command):
    """Run a subprocess command and report errors without blocking the GUI."""

    try:
        processes.append(subprocess.Popen(command))
    except Exception as error:  # noqa: BLE001
        messagebox.showerror(
            "Error",
            f"Failed to launch {command[-1]}:\n{error}"
        )


def exit_app():
    """Force-terminate any running subprocesses and close the application."""

    for process in processes:
        if process.poll() is None:
            process.kill()

    root.destroy()


root = tk.Tk()
root.title("Face Recognition")
root.resizable(False, False)

tk.Label(
    root,
    text="Face Recognition Application",
    font=("Helvetica", 16, "bold"),
    padx=20,
    pady=20
).pack()

button_frame = ttk.Frame(root)
button_frame.pack(pady=(0, 20))

ttk.Button(
    button_frame,
    text="Register",
    width=15,
    command=run_register
).pack(side="left", padx=10)

ttk.Button(
    button_frame,
    text="Capture",
    width=15,
    command=run_capture
).pack(side="left", padx=10)

ttk.Button(
    button_frame,
    text="Exit",
    width=15,
    command=exit_app
).pack(side="left", padx=10)


def _on_close():
    """Handle window-manager close (X button) by force-stopping subprocesses."""

    exit_app()


def _monitor():
    """Periodically remove finished processes and keep the GUI responsive."""

    processes[:] = [p for p in processes if p.poll() is None]

    root.after(500, _monitor)


root.protocol("WM_DELETE_WINDOW", _on_close)

_monitor()

root.mainloop()
