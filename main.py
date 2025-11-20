import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

BASE_DIR = Path(__file__).resolve().parent

USERS_FILE = BASE_DIR / "users.csv"       # user_id, username, password, age
MOVIES_FILE = BASE_DIR / "movies.csv"     # movie_id, title, category, year
RATINGS_FILE = BASE_DIR / "ratings.csv"   # user_id, movie_id, rating


def load_users() -> pd.DataFrame:
    if not USERS_FILE.exists():
        df = pd.DataFrame(columns=["user_id", "username", "password", "age"])
        df.to_csv(USERS_FILE, index=False)
        return df

    df = pd.read_csv(USERS_FILE)
    if df.empty:
        df = pd.DataFrame(columns=["user_id", "username", "password", "age"])
    if "user_id" not in df.columns:
        df["user_id"] = []
    df["user_id"] = pd.to_numeric(df.get("user_id", 0), errors="coerce").fillna(0).astype(int)
    df["age"] = pd.to_numeric(df.get("age", 0), errors="coerce").fillna(0).astype(int)
    df["username"] = df.get("username", "").astype(str)
    df["password"] = df.get("password", "").astype(str)
    return df

def save_users(df: pd.DataFrame) -> None:
    df.to_csv(USERS_FILE, index=False)

def load_movies() -> dict:
    if not MOVIES_FILE.exists():
        messagebox.showerror("Eroare", f"Nu există fișierul {MOVIES_FILE}")
        sys.exit(1)

    movies_df = pd.read_csv(MOVIES_FILE)

    required = {"movie_id", "title", "category", "year"}
    if not required.issubset(set(movies_df.columns)):
        messagebox.showerror("Eroare", "movies.csv nu conține coloanele necesare: movie_id,title,category,year")
        sys.exit(1)

    movies_df["movie_id"] = pd.to_numeric(movies_df["movie_id"], errors="coerce").astype("Int64")

    def parse_categories(series: pd.Series) -> pd.Series:
        def split_and_clean(value):
            s = str(value)
            for sep in (",", "/", "|", ";"):
                s = s.replace(sep, "|")
            parts = [p.strip() for p in s.split("|")]
            return [p for p in parts if p]

        return series.apply(split_and_clean)

    movies_df["categories"] = parse_categories(movies_df["category"])
    movies_df = movies_df.dropna(subset=["movie_id"])
    movies_df["movie_id"] = movies_df["movie_id"].astype(int)

    return movies_df.set_index("movie_id").to_dict("index")

def load_ratings() -> pd.DataFrame:
    if not RATINGS_FILE.exists():
        df = pd.DataFrame(columns=["user_id", "movie_id", "rating"])
        df.to_csv(RATINGS_FILE, index=False)
        return df

    df = pd.read_csv(RATINGS_FILE)
    if df.empty:
        return pd.DataFrame(columns=["user_id", "movie_id", "rating"])

    for col in ["user_id", "movie_id", "rating"]:
        if col not in df.columns:
            df[col] = np.nan

    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
    df["movie_id"] = pd.to_numeric(df["movie_id"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    df = df.dropna(subset=["user_id", "movie_id", "rating"])
    df["user_id"] = df["user_id"].astype(int)
    df["movie_id"] = df["movie_id"].astype(int)
    df["rating"] = df["rating"].astype(float)

    return df[["user_id", "movie_id", "rating"]]

def save_ratings(df: pd.DataFrame) -> None:
    df.to_csv(RATINGS_FILE, index=False)

def register_user(users_df: pd.DataFrame, username: str, password: str, age: int = 0):
    username = str(username).strip()
    password = str(password)

    if not username or not password:
        return None, None

    exists = users_df["username"].astype(str).str.lower().eq(username.lower())
    if exists.any():
        return None, None

    if users_df.empty:
        new_id = 1
    else:
        new_id = int(users_df["user_id"].max()) + 1

    new_row = {
        "user_id": new_id,
        "username": username,
        "password": password,
        "age": int(age) if age is not None else 0,
    }

    users_df = pd.concat([users_df, pd.DataFrame([new_row])], ignore_index=True)
    save_users(users_df)
    return users_df, new_id

def login_user(users_df: pd.DataFrame, username: str, password: str):
    username = str(username).strip()
    password = str(password)

    mask = users_df["username"].astype(str).str.lower().eq(username.lower())
    if not mask.any():
        return None

    row = users_df.loc[mask].iloc[0]
    if str(row["password"]) != password:
        return None

    return int(row["user_id"])

def cosine_similarity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    matrix = df.to_numpy(dtype=float)
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    sim = (matrix @ matrix.T) / (row_norms @ row_norms.T)
    sim = np.clip(sim, -1.0, 1.0)
    return pd.DataFrame(sim, index=df.index, columns=df.index)

def compute_recommendations(user_id: int,
                            ratings_df: pd.DataFrame,
                            movies_dict: dict,
                            top_n: int = 20):

    result = {"items": [], "message": None}

    if ratings_df is None or ratings_df.empty:
        result["message"] = "Nu există ratinguri."
        return result
    if not movies_dict:
        result["message"] = "Nu există filme."
        return result

    try:
        # Set de filme deja evaluate de utilizator (folosit și în fallback-uri)
        try:
            user_rated_ids = set(
                ratings_df.loc[ratings_df["user_id"] == user_id, "movie_id"].astype(int).tolist()
            )
        except Exception:
            user_rated_ids = set()
        user_item = ratings_df.pivot_table(
            index="user_id",
            columns="movie_id",
            values="rating"
        )
        user_item_filled = user_item.fillna(0.0)

        if user_id not in user_item_filled.index:
            # fallback: cele mai populare filme, EXCLUS cele deja văzute/rate de utilizator
            agg = ratings_df.groupby("movie_id")["rating"].agg(["mean", "count"]).reset_index()
            agg = agg.sort_values(by=["mean", "count"], ascending=[False, False])
            items = []
            for _, row in agg.iterrows():
                mid = int(row["movie_id"]) 
                if mid in user_rated_ids:
                    continue
                if mid in movies_dict:
                    items.append({
                        "movie_id": mid,
                        "score": float(row["mean"])
                    })
                if len(items) >= top_n:
                    break
            result["items"] = items
            return result

        sim_df = cosine_similarity_matrix(user_item_filled)
        sims = sim_df.loc[user_id].drop(user_id)
        sims = sims[sims > 0]
        if sims.empty:
            # fallback la popularitate, EXCLUS cele deja văzute/rate
            agg = ratings_df.groupby("movie_id")["rating"].agg(["mean", "count"]).reset_index()
            agg = agg.sort_values(by=["mean", "count"], ascending=[False, False])
            items = []
            for _, row in agg.iterrows():
                mid = int(row["movie_id"]) 
                if mid in user_rated_ids:
                    continue
                if mid in movies_dict:
                    items.append({
                        "movie_id": mid,
                        "score": float(row["mean"])
                    })
                if len(items) >= top_n:
                    break
            result["items"] = items
            return result

        user_rated = user_item.loc[user_id].dropna().index

        scores = {}
        for sim_uid, sim_val in sims.items():
            other_ratings = user_item.loc[sim_uid].dropna()
            for mid, r in other_ratings.items():
                if mid in user_rated:
                    continue
                if mid not in scores:
                    scores[mid] = {"weighted_sum": 0.0, "total_sim": 0.0}
                scores[mid]["weighted_sum"] += float(r) * float(sim_val)
                scores[mid]["total_sim"] += float(sim_val)

        if not scores:
            result["message"] = "Nu s-au putut genera recomandări."
            return result

        items = []
        for mid, vals in scores.items():
            # Siguranță: nu recomanda filme deja evaluate de utilizator
            try:
                if int(mid) in user_rated_ids:
                    continue
            except Exception:
                # dacă tipul nu se convertește corect, comparăm direct
                if mid in user_rated_ids:
                    continue
            if vals["total_sim"] == 0:
                continue
            score = vals["weighted_sum"] / vals["total_sim"]
            if mid in movies_dict:
                items.append({"movie_id": int(mid), "score": float(score)})

        items.sort(key=lambda x: x["score"], reverse=True)
        result["items"] = items[:top_n]
        return result

    except Exception as e:
        result["message"] = f"Eroare la recomandări: {e}"
        return result

def collect_all_categories(movies_dict: dict):
    cats = set()
    for info in movies_dict.values():
        lst = info.get("categories", [])
        for c in lst:
            c = str(c).strip()
            if c:
                cats.add(c)
    return cats

def stars_from_rating(r):
    if r is None or r == "" or (isinstance(r, float) and np.isnan(r)):
        return "☆☆☆☆☆"
    try:
        r = float(r)
    except Exception:
        return "☆☆☆☆☆"

    if r < 0:
        r = 0
    if r > 5:
        r = 5

    full = int(r)
    frac = r - full
    half = 1 if frac >= 0.5 else 0
    empty = 5 - full - half

    return "★" * full + "⯪" * half + "☆" * empty

def center_window(win: tk.Tk, width: int | None = None, height: int | None = None):
    """Center a Tk/Toplevel window on the current screen.
    If width and height are provided, use them; otherwise, measure the window's
    requested size after laying out widgets.
    """
    try:
        if width is not None and height is not None:
            w, h = int(width), int(height)
        else:
            win.update_idletasks()
            w = win.winfo_width() or win.winfo_reqwidth()
            h = win.winfo_height() or win.winfo_reqheight()

        screen_w = win.winfo_screenwidth()
        screen_h = win.winfo_screenheight()
        x = max((screen_w - w) // 2, 0)
        y = max((screen_h - h) // 2, 0)
        win.geometry(f"{w}x{h}+{x}+{y}")
    except Exception:
        pass

def palette():
    return {
        # Base palette
        "bg": "#0B0B0B",                    # global background (windows)
        "fg": "#EDEDED",                    # primary text

        # Window/root highlight (focus ring) colors
        "window_highlight_bg": "#0B0B0B",
        "window_highlight_fg": "#0B0B0B",

        # Label
        "label_bg": "#0B0B0B",
        "label_fg": "#EDEDED",

        # Frame
        "frame_bg": "#0B0B0B",

        # Button (classic Tk)
        "button_bg": "#0B0B0B",
        "button_fg": "#EDEDED",
        "button_active_bg": "#D98CB4",
        "button_active_fg": "#0B0B0B",
        "button_border_color": "#0B0B0B",
        "button_focus_color": "#0B0B0B",

        # Entry (classic Tk)
        "entry_bg": "#121212",
        "entry_fg": "#EDEDED",
        "entry_caret": "#D98CB4",
        "entry_select_bg": "#D98CB4",
        "entry_select_fg": "#0B0B0B",
        "entry_highlight_bg": "#0B0B0B",
        "entry_highlight_fg": "#0B0B0B",

        # Menu
        "menu_bg": "#121212",
        "menu_fg": "#EDEDED",
        "menu_active_bg": "#D98CB4",
        "menu_active_fg": "#0B0B0B",

        # Treeview (ttk)
        "tree_bg": "#121212",
        "tree_field_bg": "#0B0B0B",
        "tree_fg": "#EDEDED",
        "tree_sel_bg": "#D98CB4",
        "tree_sel_fg": "#0B0B0B",

        # Treeview Heading
        "heading_bg": "#151515",
        "heading_fg": "#EDEDED",
        "heading_hover_bg": "#D98CB4",
        "heading_hover_fg": "#0B0B0B",

        # Scrollbar
        "scroll_track": "#1A1A1A",
        "scroll_thumb": "#3A3A3A",
        "scroll_thumb_hover": "#D98CB4",
        "scroll_thumb_pressed": "#D98CB4",
        "scrollbar_width": 12,
    }

theme = False

def apply_oled_dark_mode(root: tk.Misc) -> None:
    global theme
    if theme: return
    try:
        colors = palette()
        try:
            if hasattr(root, "configure"):
                root.configure(bg=colors["bg"])  # type: ignore[arg-type]
        except Exception:
            pass

        root.option_add("*Background", colors["bg"])
        root.option_add("*Foreground", colors["fg"])
        root.option_add("*highlightBackground", colors["window_highlight_bg"])
        root.option_add("*highlightColor", colors["window_highlight_fg"])
        root.option_add("*highlightThickness", 0)
        root.option_add("*Label.background", colors["label_bg"])
        root.option_add("*Label.foreground", colors["label_fg"])
        root.option_add("*Label.relief", "flat")
        root.option_add("*Label.borderWidth", 0)
        root.option_add("*Frame.background", colors["frame_bg"])
        root.option_add("*Frame.relief", "flat")
        root.option_add("*Frame.borderWidth", 0)
        root.option_add("*Button.background", colors["button_bg"])
        root.option_add("*Button.foreground", colors["button_fg"])
        root.option_add("*Button.activeBackground", colors["button_active_bg"])
        root.option_add("*Button.activeForeground", colors["button_active_fg"])
        root.option_add("*Button.relief", "flat")
        root.option_add("*Button.borderWidth", 0)
        root.option_add("*Button.highlightBackground", colors["button_border_color"])
        root.option_add("*Button.highlightColor", colors["button_focus_color"])
        root.option_add("*Button.highlightThickness", 0)
        root.option_add("*Entry.background", colors["entry_bg"])
        root.option_add("*Entry.foreground", colors["entry_fg"])
        root.option_add("*Entry.insertBackground", colors["entry_caret"])
        root.option_add("*Entry.selectBackground", colors["entry_select_bg"])
        root.option_add("*Entry.selectForeground", colors["entry_select_fg"])
        root.option_add("*Entry.highlightBackground", colors["entry_highlight_bg"])
        root.option_add("*Entry.highlightColor", colors["entry_highlight_fg"])
        root.option_add("*Entry.highlightThickness", 0)
        root.option_add("*Entry.relief", "flat")
        root.option_add("*Entry.borderWidth", 0)
        root.option_add("*Menu.background", colors["menu_bg"])
        root.option_add("*Menu.foreground", colors["menu_fg"])
        root.option_add("*Menu.activeBackground", colors["menu_active_bg"])
        root.option_add("*Menu.activeForeground", colors["menu_active_fg"])
        root.option_add("*Menu.borderWidth", 0)
        root.option_add("*Menu.activeBorderWidth", 0)
        root.option_add("*Menu.relief", "flat")

        try:
            style = ttk.Style(root)
            try:
                style.theme_use('clam')
            except Exception:
                pass

            style.configure(
                'Treeview',
                background=colors["tree_bg"],
                fieldbackground=colors["tree_field_bg"],
                foreground=colors["tree_fg"],
                borderwidth=0,
                relief='flat'
            )
            style.configure(
                'Treeview.Heading',
                background=colors["heading_bg"],
                foreground=colors["heading_fg"],
                borderwidth=0,
                relief='flat'
            )

            try:
                style.configure('TButton', relief='flat', borderwidth=0, focusthickness=0,
                                background=colors["button_bg"],
                                foreground=colors["button_fg"])
                style.map('TButton',
                          relief=[('pressed', 'flat'), ('active', 'flat')],
                          background=[('active', colors["button_active_bg"]),
                                      ('pressed', colors["button_active_bg"])],
                          foreground=[('active', colors["button_active_fg"]),
                                      ('pressed', colors["button_active_fg"])])
            except Exception:
                pass
            try:
                style.map(
                    'Treeview.Heading',
                    background=[('active', colors["heading_hover_bg"]),
                                ('pressed', colors["heading_hover_bg"])],
                    foreground=[('active', colors["heading_hover_fg"]),
                                ('pressed', colors["heading_hover_fg"])]
                )
            except Exception:
                pass
            style.map(
                'Treeview',
                background=[('selected', colors["tree_sel_bg"])],
                foreground=[('selected', colors["tree_sel_fg"])],
                bordercolor=[('focus', ''), ('!focus', '')]
            )

            style.configure(
                'OLED.TScrollbar',
                gripcount=0,
                background=colors["scroll_thumb"],
                troughcolor=colors["scroll_track"],
                bordercolor=colors["scroll_track"],
                lightcolor=colors["scroll_thumb"],
                darkcolor=colors["scroll_thumb"],
                arrowsize=0,
                relief='flat'
            )
            style.map(
                'OLED.TScrollbar',
                background=[('active', colors["scroll_thumb_hover"]),
                            ('pressed', colors["scroll_thumb_pressed"])],
                troughcolor=[('!active', colors["scroll_track"])],
            )
            style.configure('OLED.Vertical.TScrollbar', width=colors["scrollbar_width"])
        except Exception:
            pass

        theme = True
    except Exception:
        # Non-critical; ignore if environment/theme doesn't support
        pass

def ui_login():
    users_df = load_users()

    def do_login():
        nonlocal users_df
        u = entry_user.get().strip()
        p = entry_pass.get().strip()
        user_id = login_user(users_df, u, p)
        if user_id is None:
            messagebox.showerror("Eroare", "Username sau parolă greșite.")
            return
        # Refolosește aceeași fereastră pentru a păstra tema aplicată o singură dată
        for child in win.winfo_children():
            try:
                child.destroy()
            except Exception:
                pass
        ui_main_window(int(user_id), existing_root=win, start_mainloop=False)

    def do_register():
        nonlocal users_df
        u = entry_user.get().strip()
        p = entry_pass.get().strip()

        if not u or not p:
            messagebox.showerror("Eroare", "Completează username și parolă.")
            return

        age = simpledialog.askinteger("Vârstă", "Introdu vârsta:", minvalue=0, maxvalue=120)
        if age is None:
            age = 0

        new_df, new_id = register_user(users_df, u, p, age)
        if new_id is None:
            messagebox.showerror("Eroare", "Username deja folosit.")
        else:
            users_df = new_df
            messagebox.showinfo("Succes", "Cont creat. Loghează-te.")

    win = tk.Tk()
    win.title("Login")
    win.geometry("300x200")
    apply_oled_dark_mode(win)
    center_window(win, 300, 200)
    tk.Label(win, text="Username").pack(pady=5)
    entry_user = tk.Entry(win)
    entry_user.pack()

    tk.Label(win, text="Parolă").pack(pady=5)
    entry_pass = tk.Entry(win, show="*")
    entry_pass.pack()

    def _enter_to_login(event=None):
        do_login()
        return "break"

    try:
        entry_pass.bind("<Return>", _enter_to_login)
        entry_pass.bind("<KP_Enter>", _enter_to_login)  # Numpad Enter
    except Exception:
        pass

    tk.Button(win, text="Login", width=20, command=do_login).pack(pady=8)
    tk.Button(win, text="Register", width=20, command=do_register).pack()

    win.mainloop()

def ui_main_window(user_id: int, existing_root: tk.Tk | None = None, start_mainloop: bool = True):
    movies_dict = load_movies()
    ratings_df_global = load_ratings()

    ratings_df_local = ratings_df_global

    win = existing_root if existing_root is not None else tk.Tk()
    try:
        _users_df = load_users()
        _row = _users_df.loc[_users_df["user_id"] == int(user_id)]
        if not _row.empty:
            _username = str(_row.iloc[0]["username"]) or f"user {user_id}"
        else:
            _username = f"user {user_id}"
    except Exception:
        _username = f"user {user_id}"
    win.title(f"Filme - {_username}")
    win.geometry("1000x650")
    center_window(win, 1000, 650)

    frame_list = tk.Frame(win)
    frame_list.pack(fill="both", expand=True)

    cols = ("ID", "Titlu", "Categorie", "Rating")
    tree = ttk.Treeview(frame_list, columns=cols, show="headings")
    for c in cols:
        tree.heading(c, text=c, anchor="center")
        tree.column(c, width=200, anchor="center")

    vscroll = ttk.Scrollbar(frame_list, orient="vertical", command=tree.yview, style="OLED.Vertical.TScrollbar")
    tree.configure(yscrollcommand=vscroll.set)

    frame_list.grid_rowconfigure(0, weight=1)
    frame_list.grid_columnconfigure(0, weight=1)
    tree.grid(row=0, column=0, sticky="nsew")
    vscroll.grid(row=0, column=1, sticky="ns")

    sort_state = {"col": None, "reverse": False}

    def _rating_value_from_text(txt: str) -> float:
        if not txt:
            return -1.0
        full = txt.count("★")
        half = txt.count("⯪")
        return full + 0.5 * half

    def sort_by(col_name: str):
        rows = []
        for iid in tree.get_children(""):
            vals = tree.item(iid)["values"]
            rows.append(tuple(vals))

        if col_name == "ID":
            key_fn = lambda v: int(v[0])
        elif col_name == "Titlu":
            key_fn = lambda v: str(v[1]).lower()
        elif col_name == "Categorie":
            key_fn = lambda v: str(v[2]).lower()
        elif col_name == "Rating":
            key_fn = lambda v: _rating_value_from_text(str(v[3]))
        else:
            key_fn = lambda v: v

        if sort_state["col"] == col_name:
            sort_state["reverse"] = not sort_state["reverse"]
        else:
            sort_state["col"] = col_name
            sort_state["reverse"] = False

        rows.sort(key=key_fn, reverse=sort_state["reverse"])

        tree.delete(*tree.get_children())
        for r in rows:
            tree.insert("", tk.END, values=r)

    for c in cols:
        tree.heading(c, text=c, command=lambda cn=c: sort_by(cn), anchor="center")

    def _update_rating_header_label():
        label = "Match" if current_view.get("name") == "recommended" else "Rating"
        tree.heading("Rating", text=label, command=lambda cn="Rating": sort_by(cn), anchor="center")

    def user_rating(mid):
        r = ratings_df_local.loc[
            (ratings_df_local["user_id"] == user_id) &
            (ratings_df_local["movie_id"] == mid)
        ]
        if r.empty:
            return None
        return float(r.iloc[0]["rating"])

    current_view = {"name": "all", "category": None}

    def refresh_all():
        tree.delete(*tree.get_children())
        for mid in sorted(movies_dict.keys()):
            d = movies_dict[mid]
            r = user_rating(mid)
            rating_text = stars_from_rating(r) if r is not None else ""
            tree.insert("", tk.END, values=(mid, d["title"], d["category"], rating_text))

    def refresh_watched():
        tree.delete(*tree.get_children())
        seen = set(
            ratings_df_local.loc[ratings_df_local["user_id"] == user_id, "movie_id"]
        )
        for mid in sorted(seen):
            if mid in movies_dict:
                d = movies_dict[mid]
                r = user_rating(mid)
                rating_text = stars_from_rating(r) if r is not None else ""
                tree.insert("", tk.END, values=(mid, d["title"], d["category"], rating_text))

    def refresh_unwatched():
        tree.delete(*tree.get_children())
        seen = set(
            ratings_df_local.loc[ratings_df_local["user_id"] == user_id, "movie_id"]
        )
        unseen_ids = [mid for mid in movies_dict.keys() if mid not in seen]
        for mid in sorted(unseen_ids):
            d = movies_dict[mid]
            tree.insert("", tk.END, values=(mid, d["title"], d["category"], ""))

    def refresh_recommended():
        res = compute_recommendations(user_id, ratings_df_local, movies_dict, top_n=20)
        items = res.get("items", [])
        tree.delete(*tree.get_children())
        for itm in sorted(items, key=lambda x: int(x["movie_id"])):
            mid = itm["movie_id"]
            score = itm["score"]
            if mid in movies_dict:
                d = movies_dict[mid]
                rating_text = stars_from_rating(score)
                tree.insert("", tk.END, values=(mid, d["title"], d["category"], rating_text))

    def refresh_by_category(cat):
        """Aplică filtrul de categorie peste ultima vizualizare de bază (toate/văzute/nevăzute/recomandate)."""
        cat = (cat or "").strip().lower()
        tree.delete(*tree.get_children())
        base = current_view.get("name", "all")

        eligible_ids = []
        if base == "all":
            eligible_ids = list(movies_dict.keys())
        elif base == "watched":
            seen = set(
                ratings_df_local.loc[ratings_df_local["user_id"] == user_id, "movie_id"]
            )
            eligible_ids = [mid for mid in movies_dict.keys() if mid in seen]
        elif base == "unwatched":
            seen = set(
                ratings_df_local.loc[ratings_df_local["user_id"] == user_id, "movie_id"]
            )
            eligible_ids = [mid for mid in movies_dict.keys() if mid not in seen]
        elif base == "recommended":
            res = compute_recommendations(user_id, ratings_df_local, movies_dict, top_n=20)
            items = res.get("items", [])
            eligible_ids = [itm["movie_id"] for itm in items if itm.get("movie_id") in movies_dict]
        else:
            eligible_ids = list(movies_dict.keys())

        filtered_ids = []
        for mid in eligible_ids:
            d = movies_dict[mid]
            cats = [c.lower() for c in d.get("categories", [])]
            if cat in cats:
                filtered_ids.append(mid)

        for mid in sorted(filtered_ids):
            d = movies_dict[mid]
            if base == "recommended":
                res = compute_recommendations(user_id, ratings_df_local, movies_dict, top_n=20)
                items = {itm["movie_id"]: itm.get("score") for itm in res.get("items", [])}
                score = items.get(mid, None)
                rating_text = stars_from_rating(score) if score is not None else ""
            else:
                r = user_rating(mid)
                rating_text = stars_from_rating(r) if r is not None else ""
            tree.insert("", tk.END, values=(mid, d["title"], d["category"], rating_text))

    def open_category_selector(event=None):
        """Afișează dropdown-ul categoriilor chiar deasupra butonului „Categorie”."""
        categories = sorted(collect_all_categories(movies_dict))
        def apply_category(cat_name: str):
            current_view["category"] = cat_name
            _update_rating_header_label()
            refresh_by_category(cat_name)

        menu = tk.Menu(win, tearoff=0)
        for cat in categories:
            menu.add_command(label=cat, command=lambda c=cat: apply_category(c))
        try:
            win.update_idletasks()
            try:
                menu.update_idletasks()
            except Exception:
                pass
            req_w = menu.winfo_reqwidth() if hasattr(menu, "winfo_reqwidth") else 0
            req_h = menu.winfo_reqheight() if hasattr(menu, "winfo_reqheight") else 0
            btn_x = btn_category.winfo_rootx()
            btn_y = btn_category.winfo_rooty()
            btn_w = btn_category.winfo_width()
            btn_h = btn_category.winfo_height()
            x = btn_x + max((btn_w - (req_w or 0)) // 2, 0)
            y = btn_y - (req_h or 0) - 2
            screen_w = win.winfo_screenwidth()
            screen_h = win.winfo_screenheight()
            if req_w:
                x = max(0, min(x, screen_w - req_w))
            if y < 0 or not req_h:
                y = btn_y + btn_h + 2
                if req_h:
                    y = min(y, screen_h - req_h)

            menu.tk_popup(x, y)
        finally:
            try:
                menu.grab_release()
            except Exception:
                pass

    def refresh_current_view():
        _update_rating_header_label()
        name = current_view["name"]
        cat = current_view.get("category")
        if cat:
            refresh_by_category(cat)
            return
        if name == "all":
            refresh_all()
        elif name == "watched":
            refresh_watched()
        elif name == "unwatched":
            refresh_unwatched()
        elif name == "recommended":
            refresh_recommended()
        else:
            refresh_all()

    def edit_rating(event):
        nonlocal ratings_df_local

        sel = tree.selection()
        if not sel:
            return

        vals = tree.item(sel[0])["values"]
        mid = int(vals[0])
        title = vals[1]

        rate_win = tk.Toplevel(win)
        rate_win.title(f"Rating: {title}")
        rate_win.resizable(False, False)
        tk.Label(rate_win, text=title, font=("Arial", 14)).pack(pady=10)

        frame_stars = tk.Frame(rate_win)
        frame_stars.pack()

        hover_value_label = tk.Label(rate_win, text="", font=("Arial", 11))
        hover_value_label.pack(pady=(6, 0))

        star_buttons = []

        def set_rating(r):
            nonlocal ratings_df_local
            exists = (
                (ratings_df_local["user_id"] == user_id) &
                (ratings_df_local["movie_id"] == mid)
            )
            if exists.any():
                ratings_df_local.loc[exists, "rating"] = float(r)
            else:
                ratings_df_local = pd.concat([
                    ratings_df_local,
                    pd.DataFrame([{
                        "user_id": user_id,
                        "movie_id": mid,
                        "rating": float(r)
                    }])
                ], ignore_index=True)

            save_ratings(ratings_df_local)
            ratings_df_local = load_ratings()
            refresh_current_view()
            rate_win.destroy()

        def render_hover(idx):
            full_count = (idx + 1) // 2
            val = (idx + 1) / 2
            hover_value_label.config(text=f"{val:.1f}/5")

            for i, btn in enumerate(star_buttons):
                star = i // 2
                if star < full_count:
                    btn.config(text="★")
                else:
                    btn.config(text="☆")

        def render_reset():
            for btn in star_buttons:
                btn.config(text="☆")
            hover_value_label.config(text="")

        for i in range(10):
            btn = tk.Button(
                frame_stars,
                text="☆",
                font=("Arial", 28),
                bd=0,
                relief="flat",
                padx=1,
                pady=0,
                command=lambda idx=i: set_rating((idx + 1) / 2)
            )
            btn.grid(row=0, column=i)
            btn.bind("<Enter>", lambda e, idx=i: render_hover(idx))
            btn.bind("<Leave>", lambda e: render_reset())
            star_buttons.append(btn)
        center_window(rate_win)

    tree.bind("<Double-1>", edit_rating)

    frame = tk.Frame(win)
    frame.pack(pady=10)

    def _set_view(name, cat=None):
        current_view["name"] = name
        current_view["category"] = cat
        _update_rating_header_label()
        refresh_current_view()

    tk.Button(frame, text="Toate", width=15, command=lambda: _set_view("all")).grid(row=0, column=0, padx=5)
    tk.Button(frame, text="Văzute", width=15, command=lambda: _set_view("watched")).grid(row=0, column=1, padx=5)
    tk.Button(frame, text="Nevăzute", width=15, command=lambda: _set_view("unwatched")).grid(row=0, column=2, padx=5)
    tk.Button(frame, text="Recomandate", width=15, command=lambda: _set_view("recommended")).grid(row=0, column=3, padx=5)
    btn_category = tk.Button(frame, text="Categorie", width=15, command=open_category_selector)
    btn_category.grid(row=0, column=4, padx=5)
    current_view["name"] = "all"
    current_view["category"] = None
    _update_rating_header_label()
    refresh_all()
    if start_mainloop and existing_root is None:
        win.mainloop()

if __name__ == "__main__":
    ui_login()