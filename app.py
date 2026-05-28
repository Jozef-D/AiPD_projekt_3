import sys
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from src import *

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



def load_wav_and_features(path, window="hann", use_fft=True):
    raw = open(path, "rb").read()
    sig, sr, ch, bps = read_wav(raw)

    sig = normalize_signal(sig)

    if use_fft:
        features = extract_features(sig, sample_rate=sr, window=window)
    else:
        sig_rms = trim_silence(sig)
        features = extract_frames(sig_rms)

    freqs, avg_spec = mean_fft_spectrum(sig, window=window, sample_rate=sr)

    return sig, sr, features, freqs, avg_spec



class App(tk.Tk):
    WINDOWS = ("hann", "hamming", "blackman", "rectangular")

    def __init__(self):
        super().__init__()
        self.title("Klasyfikator DTW")
        self.geometry("1500x950")
        self.resizable(True, True)

        self.database = []
        self.db_path = "Baza_nagran"
        self.wav_path = ""
        self.signal = None
        self.sample_rate = None
        self.features = None
        self.freqs = None
        self.avg_spec = None

        self._build_ui()
        self._auto_load_db()


    def _build_ui(self):
        top = tk.Frame(self, bd=1, relief=tk.GROOVE)
        top.pack(fill=tk.X, padx=8, pady=(8, 2))

        tk.Label(top, text="Plik WAV:").pack(side=tk.LEFT, padx=4)
        self.var_wav = tk.StringVar(value="(nie wybrano)")
        tk.Label(top, textvariable=self.var_wav, width=48, anchor="w",
                 relief=tk.SUNKEN).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Wybierz plik…", command=self._choose_wav).pack(side=tk.LEFT, padx=4)
        self.btn_classify = tk.Button(top, text="Klasyfikuj",
                                      command=self._classify, state=tk.DISABLED)
        self.btn_classify.pack(side=tk.LEFT, padx=4)

        tk.Label(top, text="k-NN k=").pack(side=tk.LEFT, padx=(18, 2))
        self.var_k = tk.IntVar(value=1)
        tk.Spinbox(top, from_=1, to=20, width=4,
                   textvariable=self.var_k).pack(side=tk.LEFT)

        # okno FFT
        tk.Label(top, text="  Okno FFT:").pack(side=tk.LEFT, padx=(18, 2))
        self.var_window = tk.StringVar(value="hann")
        cb = ttk.Combobox(top, textvariable=self.var_window,
                          values=self.WINDOWS, width=12, state="readonly")
        cb.pack(side=tk.LEFT)
        cb.bind("<<ComboboxSelected>>", self._on_window_changed)

        # tryb cech
        tk.Label(top, text="  Cechy:").pack(side=tk.LEFT, padx=(18, 2))
        self.var_feat_mode = tk.StringVar(value="FFT")
        for val in ("FFT", "RMS"):
            tk.Radiobutton(top, text=val, variable=self.var_feat_mode,
                           value=val, command=self._on_feat_mode_changed
                           ).pack(side=tk.LEFT)

        self.lbl_db_status = tk.Label(top, text="", fg="gray")
        self.lbl_db_status.pack(side=tk.LEFT, padx=12)

        # wykresy
        plot_frame = tk.LabelFrame(self, text="Analiza sygnału")
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.fig = plt.figure(figsize=(11.5, 6.5))
        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                               height_ratios=[1.6, 1],
                               hspace=0.45, wspace=0.32)

        self.ax_sig = self.fig.add_subplot(gs[0, :])
        self.ax_rms = self.fig.add_subplot(gs[1, 0])
        self.ax_fft = self.fig.add_subplot(gs[1, 1])

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_empty_plots()

        # wyniki klasyfikacji
        res_frame = tk.LabelFrame(self, text="Wyniki klasyfikacji")
        res_frame.pack(fill=tk.X, padx=8, pady=(2, 8))

        inner = tk.Frame(res_frame)
        inner.pack(padx=8, pady=6)

        def _lbl(row, text, var, color):
            tk.Label(inner, text=text, width=22, anchor="w").grid(
                row=row, column=0, padx=4, pady=2)
            tk.Label(inner, textvariable=var, width=20, anchor="w",
                     font=("TkDefaultFont", 11, "bold"), fg=color).grid(
                row=row, column=1, padx=4)

        self.var_word = tk.StringVar(value="—")
        self.var_speaker = tk.StringVar(value="—")
        self.var_dist = tk.StringVar(value="—")
        _lbl(0, "Rozpoznane słowo:", self.var_word, "navy")
        _lbl(1, "Zidentyfikowana osoba:", self.var_speaker, "darkgreen")
        _lbl(2, "Najlepszy dystans DTW:", self.var_dist, "black")

        tk.Label(inner, text="Top-k dopasowania:", width=22, anchor="w").grid(
            row=0, column=2, padx=(24, 4), pady=2)
        self.txt_topk = tk.Text(inner, width=52, height=4, state=tk.DISABLED,
                                relief=tk.SUNKEN, bd=1)
        self.txt_topk.grid(row=0, column=3, rowspan=3, padx=4, pady=2)

        self.var_status = tk.StringVar(value="Gotowy.")
        tk.Label(res_frame, textvariable=self.var_status, fg="gray",
                 anchor="w").pack(fill=tk.X, padx=8, pady=(0, 4))

    # obsługa bazy danych

    def _auto_load_db(self):
        self.lbl_db_status.config(text="Ładowanie bazy…", fg="orange")
        self.var_status.set("Ładowanie bazy danych…")
        self.update()

        def _worker():
            try:
                db = load_database(self.db_path)
                self.after(0, lambda db=db: self._on_db_loaded(db))
            except Exception as exc:
                msg = str(exc)
                self.after(0, lambda msg=msg: self._on_db_error(msg))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_db_loaded(self, db):
        self.database = db
        self.lbl_db_status.config(text=f"Baza: {len(db)} nagrań", fg="green")
        self.var_status.set(f"Baza gotowa: {len(db)} nagrań.")
        self._refresh_classify_btn()

    def _on_db_error(self, msg):
        self.lbl_db_status.config(text="Błąd bazy!", fg="red")
        self.var_status.set("Błąd ładowania bazy.")
        messagebox.showerror("Błąd bazy", msg)


    def _choose_wav(self):
        path = filedialog.askopenfilename(
            title="Wybierz plik WAV",
            filetypes=[("Pliki WAV", "*.wav"), ("Wszystkie", "*.*")]
        )
        if not path:
            return
        self.wav_path = path
        self.var_wav.set(os.path.basename(path))
        self._reload_features()

    def _reload_features(self):
        if not self.wav_path:
            return
        self.var_status.set("Wczytywanie i przetwarzanie pliku…")
        self.update()

        window   = self.var_window.get()
        use_fft  = (self.var_feat_mode.get() == "FFT")

        try:
            sig, sr, feats, freqs, avg_spec = load_wav_and_features(
                self.wav_path, window=window, use_fft=use_fft
            )
        except Exception as exc:
            messagebox.showerror("Błąd wczytywania WAV", str(exc))
            self.var_status.set("Błąd wczytywania pliku.")
            return

        self.signal = sig
        self.sample_rate = sr
        self.features = feats
        self.freqs = freqs
        self.avg_spec = avg_spec

        self._plot_signal()
        self._clear_results()
        self.var_status.set(
            f"Plik wczytany: {len(sig)} próbek, {sr} Hz | "
            f"okno: {window} | cechy: {self.var_feat_mode.get()}"
        )
        self._refresh_classify_btn()

    def _on_window_changed(self, _event=None):
        self._reload_features()

    def _on_feat_mode_changed(self):
        self._reload_features()

    def _refresh_classify_btn(self):
        ok = self.features is not None and bool(self.database)
        self.btn_classify.config(state=tk.NORMAL if ok else tk.DISABLED)


    def _classify(self):
        k = self.var_k.get()
        self.btn_classify.config(state=tk.DISABLED)
        self.var_status.set("Klasyfikacja (DTW)…")
        self.var_word.set("…")
        self.var_speaker.set("…")
        self.update()

        feats = self.features
        db = self.database
        mode = "rms" if self.var_feat_mode.get() == "RMS" else "mel"

        def _worker():
            try:
                result = classify(feats, db, k=k, mode=mode)
                self.after(0, lambda result=result: self._on_classified(result))
            except Exception as exc:
                msg = str(exc)
                self.after(0, lambda msg=msg: self._on_classify_error(msg))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_classified(self, result):
        self.var_word.set(result["predicted_word"])
        self.var_speaker.set(f"Speaker {result['predicted_speaker']}")
        self.var_dist.set(f"{result['best_dist']:.4f}")

        lines = [
            f"{i}. dist={e['dist']:.3f}  słowo={e['word']}"
            f"  mówca={e['speaker_id']}  take={e['take']}"
            for i, e in enumerate(result["top_k"], 1)
        ]
        self.txt_topk.config(state=tk.NORMAL)
        self.txt_topk.delete("1.0", tk.END)
        self.txt_topk.insert(tk.END, "\n".join(lines))
        self.txt_topk.config(state=tk.DISABLED)

        self.var_status.set("Klasyfikacja zakończona.")
        self.btn_classify.config(state=tk.NORMAL)

    def _on_classify_error(self, msg):
        self.var_status.set("Błąd klasyfikacji.")
        self.btn_classify.config(state=tk.NORMAL)
        messagebox.showerror("Błąd klasyfikacji", msg)


    def _draw_empty_plots(self):
        bg = "#f9f9f9"
        for ax, title in (
            (self.ax_sig, "Sygnał (amplituda) – po normalizacji"),
            (self.ax_rms, "Energia ramek (RMS)"),
            (self.ax_fft, "Uśrednione widmo FFT"),
        ):
            ax.clear()
            ax.set_facecolor(bg)
            ax.set_title(title, fontsize=9)
        self.ax_sig.set_ylabel("Amplituda")
        self.ax_rms.set_ylabel("Energia RMS");  self.ax_rms.set_xlabel("Czas [s]")
        self.ax_fft.set_ylabel("Amplituda");    self.ax_fft.set_xlabel("Częstotliwość [Hz]")
        self.fig.canvas.draw_idle()

    def _plot_signal(self):
        sig = self.signal
        sr = self.sample_rate
        freqs = self.freqs
        avg_spec = self.avg_spec
        window = self.var_window.get()

        t_sig = np.linspace(0, len(sig) / sr, num=len(sig))

        rms_feats = extract_frames(sig)
        frame_len, hop_len = 512, 256
        n_frames  = len(rms_feats)
        t_frames  = np.array(
            [(i * hop_len + frame_len // 2) / sr for i in range(n_frames)]
        )

        ax = self.ax_sig
        ax.clear()
        ax.set_facecolor("#f9f9f9")
        ax.plot(t_sig, sig, linewidth=0.4, color="steelblue")
        ax.axhline(0, color="black", linewidth=0.3)
        ax.set_xlim(t_sig[0], t_sig[-1])
        ax.set_ylim(-1.05, 1.05)
        ax.set_title("Sygnał (amplituda) – po normalizacji szczytowej", fontsize=9)
        ax.set_ylabel("Amplituda")

        ax = self.ax_rms
        ax.clear()
        ax.set_facecolor("#f9f9f9")
        ax.plot(t_frames, rms_feats, linewidth=1.0, color="darkorange")
        ax.fill_between(t_frames, rms_feats, alpha=0.25, color="darkorange")
        ax.set_title("Energia ramek (RMS)", fontsize=9)
        ax.set_ylabel("Energia RMS")
        ax.set_xlabel("Czas [s]")
        ax.set_xlim(t_sig[0], t_sig[-1])

        ax = self.ax_fft
        ax.clear()
        ax.set_facecolor("#f9f9f9")
        if freqs is not None and avg_spec is not None:
            ax.plot(freqs, avg_spec, linewidth=0.8, color="mediumseagreen")
            ax.fill_between(freqs, avg_spec, alpha=0.2, color="mediumseagreen")
            ax.set_xlim(0, freqs[-1])
            ax.set_title(f"Uśrednione widmo FFT  [okno: {window}]", fontsize=9)
        else:
            f2, s2 = mean_fft_spectrum(sig, window="hann", sample_rate=sr)
            ax.plot(f2, s2, linewidth=0.8, color="mediumseagreen")
            ax.fill_between(f2, s2, alpha=0.2, color="mediumseagreen")
            ax.set_xlim(0, f2[-1])
            ax.set_title("Uśrednione widmo FFT  [okno: hann, poglądowe]", fontsize=9)
        ax.set_ylabel("Amplituda")
        ax.set_xlabel("Częstotliwość [Hz]")

        self.fig.canvas.draw_idle()

    def _clear_results(self):
        self.var_word.set("—")
        self.var_speaker.set("—")
        self.var_dist.set("—")
        self.txt_topk.config(state=tk.NORMAL)
        self.txt_topk.delete("1.0", tk.END)
        self.txt_topk.config(state=tk.DISABLED)



if __name__ == "__main__":
    app = App()
    app.mainloop()