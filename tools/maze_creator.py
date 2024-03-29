from typing import Dict, List
import textwrap

import yaml
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk


class MazeEditor(tk.Tk):
    def __init__(self, map_size: List[int] = [15, 3]):
        super().__init__()
        self.title("Maze Editor")
        self.geometry("800x600")

        self.map_size: List[int] = map_size
        self.grid_buttons: Dict[List[int], tk.Button] = {}
        self.button_map = {}
        self.dragging = False

        self.selected_item = tk.StringVar(value="Wall")  # Default selection
        self.wall_texture = tk.StringVar(value="")  # Default texture
        self.initialize_ui()

    def initialize_ui(self):
        self.create_toolbar()
        self.create_grid_controls()
        self.reset_grid()

    def create_toolbar(self):
        toolbar = tk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        # Dropdown for item selection
        combobox = ttk.Combobox(
            toolbar,
            textvariable=self.selected_item,
            values=[
                "Open Space (0)",
                "Reset Location (R)",
                "Object Position (X)",
                "Wall",
            ],
            state="readonly",
        )
        combobox.pack(side=tk.LEFT)
        combobox.bind("<<ComboboxSelected>>", self.on_item_selected)

        # Text box for Wall texture
        tk.Label(toolbar, text="Texture:").pack(side=tk.LEFT)
        self.texture_entry = ttk.Entry(toolbar, textvariable=self.wall_texture)
        self.texture_entry.pack(side=tk.LEFT)

        # Create save button
        tk.Button(toolbar, text="Save", command=self.save_map).pack(side=tk.RIGHT)

        # Create reset button
        tk.Button(toolbar, text="Reset", command=self.reset_grid).pack(side=tk.RIGHT)

    def create_grid_controls(self):
        # fmt: off
        # Add Row Bottom
        tk.Button(self, text='+ Row', command=lambda: self.add_row(1)).pack(side=tk.BOTTOM, fill=tk.X)
        tk.Button(self, text='- Row', command=lambda: self.remove_row(-1)).pack(side=tk.BOTTOM, fill=tk.X)

        # Add Row Top
        tk.Button(self, text='+ Row', command=lambda: self.add_row(2)).pack(side=tk.TOP, fill=tk.X)
        tk.Button(self, text='- Row', command=lambda: self.remove_row(-2)).pack(side=tk.TOP, fill=tk.X)

        # Add Column Right
        tk.Button(self, text='+ Column', command=lambda: self.add_column(1)).pack(side=tk.RIGHT, fill=tk.Y)
        tk.Button(self, text='- Column', command=lambda: self.remove_column(-1)).pack(side=tk.RIGHT, fill=tk.Y)

        # Add Column Left
        tk.Button(self, text='+ Column', command=lambda: self.add_column(2)).pack(side=tk.LEFT, fill=tk.Y)
        tk.Button(self, text='- Column', command=lambda: self.remove_column(-2)).pack(side=tk.LEFT, fill=tk.Y)
        # fmt: on

    def create_grid(self, add_row: int = 0, add_col: int = 0):
        """
        Keyword Arguments:
            add_row (int): Whether to add a row. If 0, no row is added. If 1, row is
                added on the bottom. If 2, row is added on the top. If -1, row is
                removed from the bottom. If -2, row is removed from the top.
            add_col (int): Whether to add a column. If 0, no column is added. If 1,
                column is added on the right. If 2, column is added on the left. If -1,
                column is removed from the right. If -2, column is removed from the
                left.
        """
        if add_row not in (-2, -1, 0, 1, 2):
            raise ValueError("Invalid value for add_row")
        if add_col not in (-2, -1, 0, 1, 2):
            raise ValueError("Invalid value for add_col")

        if add_row in (1, 2):
            self.map_size[0] += 1
        elif add_row in (-1, -2):
            self.map_size[0] -= 1

        if add_col in (1, 2):
            self.map_size[1] += 1
        elif add_col in (-1, -2):
            self.map_size[1] -= 1

        # Store the previous states to be restored later
        current_states = {}
        for (row, col), btn in self.grid_buttons.items():
            if add_row == 2:
                row += 1
            if add_row == -2:
                row -= 1

            if add_col == 2:
                col += 1
            elif add_col == -2:
                col -= 1

            current_states[(row, col)] = btn.config()

        if hasattr(self, "grid_frame"):
            self.grid_frame.destroy()

        # Grid for the map
        self.grid_frame = tk.Frame(self)
        self.grid_frame.pack(fill=tk.BOTH, expand=True)

        self.grid_buttons.clear()
        self.button_map.clear()
        for row in range(self.map_size[0]):
            for col in range(self.map_size[1]):
                btn = tk.Label(
                    self.grid_frame, text="0", bg="white", relief="ridge", borderwidth=1
                )
                btn.grid(row=row, column=col, sticky="nsew")
                btn.bind("<Button-1>", self.start_drag)
                btn.bind("<B1-Motion>", self.update_drag)
                btn.bind("<ButtonRelease-1>", self.stop_drag)
                self.grid_buttons[(row, col)] = btn
                self.button_map[btn] = (row, col)

                # Restore previous states
                btn_config = current_states.get((row, col), {})
                for key, value in btn_config.items():
                    if key in ("text", "background"):
                        btn[key] = value[-1]

        # Making the grid scalable
        for row in range(self.map_size[0]):
            self.grid_frame.rowconfigure(row, weight=1)
        for col in range(self.map_size[1]):
            self.grid_frame.columnconfigure(col, weight=1)

    def reset_grid(self):
        self.grid_buttons: Dict[List[int], tk.Button] = {}
        self.button_map = {}
        self.create_grid()

    def start_drag(self, _: tk.Event):
        self.dragging = True
        self.update_btn()

    def update_drag(self, _: tk.Event):
        if self.dragging:
            self.update_btn()

    def stop_drag(self, _: tk.Event):
        self.dragging = False

    def update_btn(self):
        x, y = self.grid_frame.winfo_pointerxy()
        widget_under_cursor = self.grid_frame.winfo_containing(x, y)
        if widget_under_cursor is None or widget_under_cursor not in self.button_map:
            return

        self.update_grid(*self.button_map[widget_under_cursor])

    def update_grid(self, row, col):
        if (row, col) not in self.grid_buttons:
            return

        item = self.selected_item.get()
        if item == "Open Space (0)":
            self.grid_buttons[(row, col)].config(text="0", bg="white")
        elif item == "Reset Location (R)":
            self.grid_buttons[(row, col)].config(text="R", bg="yellow")
        elif item == "Object Position (X)":
            self.grid_buttons[(row, col)].config(text="X", bg="blue")
        elif item.startswith("Wall"):
            text = f"1:{self.wall_texture.get()}" if self.wall_texture.get() else "1"
            self.grid_buttons[(row, col)].config(text=text, bg="gray")

    def on_item_selected(self, _: tk.Event):
        state = "normal" if self.selected_item.get().startswith("Wall") else "disabled"
        self.texture_entry.config(state=state)

    def add_row(self, cmd: int):
        self.create_grid(add_row=cmd)

    def remove_row(self, cmd: int):
        if self.map_size[0] > 1:
            self.create_grid(add_row=cmd)

    def add_column(self, cmd: int):
        self.create_grid(add_col=cmd)

    def remove_column(self, cmd: int):
        if self.map_size[1] > 1:
            self.create_grid(add_col=cmd)

    def print_map(self) -> str:
        column_widths = []
        for col in range(self.map_size[1]):
            column_width = max(
                len(self.grid_buttons[(row, col)]["text"])
                for row in range(self.map_size[0])
            )
            column_widths.append(column_width)

        map_str = "[" + "\n"
        for row in range(self.map_size[0]):
            map_str += "  ["
            for col in range(self.map_size[1]):
                text: str = self.grid_buttons[(row, col)]["text"]
                if col < self.map_size[1] - 1:
                    map_str += text.rjust(column_widths[col]) + ", "
                else:
                    map_str += text.rjust(column_widths[col])
            map_str += "]," + "\n"
        map_str += "]"

        def str_representer(dumper: yaml.Dumper, data: str):
            style = None
            if "\n" in data:
                # Will use the | style for multiline strings.
                style = "|"
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)

        dumper = yaml.CDumper
        dumper.add_representer(str, str_representer)
        map = yaml.dump({"map": map_str}, Dumper=dumper)

        return map

    def save_map(self):
        # Create a new top-level window
        save_window = tk.Toplevel(self)
        save_window.title("Save Maze")
        save_window.geometry("300x500")

        # Create a text box
        save_entry = tk.Text(save_window)
        save_entry.pack(fill=tk.Y, expand=True)

        # Fill the text box with the map
        text = f"""\
defaults:
- /${{env/mazes}}@_here_

{self.print_map()}"""
        save_entry.insert(tk.END, textwrap.dedent(text))

        # Insert save button
        def save():
            file_path = filedialog.asksaveasfilename(
                initialdir="./configs/env/mazes/",
                defaultextension=".yaml",
                filetypes=[("YAML files", "*.yaml")],
            )
            if file_path:
                with open(file_path, "w") as f:
                    f.write(save_entry.get("1.0", tk.END))

                # Close the window
                save_window.destroy()

        save_button = tk.Button(save_window, text="Save", command=save)
        save_button.pack()

    @staticmethod
    def load(file_path: str):
        with open(file_path, "r") as f:
            maze = yaml.safe_load(f)

        map = yaml.safe_load(maze["map"])

        app = MazeEditor()
        app.map_size = [len(map), len(map[0])]
        app.create_grid()
        for row in range(app.map_size[0]):
            for col in range(app.map_size[1]):
                item = str(map[row][col])
                if "0" == item:
                    bg = "white"
                elif "R" == item:
                    bg = "yellow"
                elif "X" == item:
                    bg = "blue"
                else:
                    bg = "gray"
                app.grid_buttons[(row, col)].config(text=item, bg=bg)

        return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--map-size", type=int, nargs=2, help="Size of the map", default=[15, 3]
    )
    parser.add_argument(
        "--load", type=str, help="Load a maze from a file", default=None
    )

    args = parser.parse_args()

    if args.load is not None:
        app = MazeEditor.load(args.load)
    else:
        app = MazeEditor(args.map_size)
    app.mainloop()
