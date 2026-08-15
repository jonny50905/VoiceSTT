# -*- coding: utf-8 -*-
"""電影字幕式螢幕疊加層:只有文字浮在畫面上,其餘完全透明且滑鼠穿透。

由 live_subtitles.py 以子程序啟動,stdin 每行一個 JSON 事件:
  {"kind":"partial","text":"..."}          # 進行中句子(草稿,持續更新)
  {"kind":"final","seq":7,"text":"..."}    # 句子定稿(草稿品質)
  {"kind":"refined","seq":7,"text":"..."}  # 2-pass 修正,取代同 seq 的定稿文字
單獨驗收:python overlay.py --demo

實作:Tk 無邊框視窗 + 色鍵透明(背景色整塊摳掉)+ WS_EX_TRANSPARENT
(滑鼠事件穿透,連文字像素也不擋)+ WS_EX_NOACTIVATE(永不搶焦點)
+ WS_EX_TOOLWINDOW(不進工作列/Alt-Tab)。文字白字黑邊,任何底色都可讀。
"""
import argparse
import ctypes
import json
import queue
import sys
import threading
import time
import tkinter as tk

KEY = "#0a0b0c"  # 色鍵:此色像素全透明(選極罕見暗色,避免與字幕色衝突)
HOLD = 5.0  # 定稿字幕無後續時停留秒數


class Overlay:
    def __init__(self, font_size, bottom, width_ratio, read_stdin=True):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-transparentcolor", KEY)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.h = int(font_size * 6.5)
        self.w = sw
        self.root.geometry(f"{sw}x{self.h}+0+{sh - self.h - bottom}")
        self.canvas = tk.Canvas(self.root, bg=KEY, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.font = ("Microsoft JhengHei", font_size, "bold")
        self.wrap = int(sw * width_ratio)
        self.committed = None  # (seq, text, wall)
        self.partial = ""
        self.white_partial = None  # (utt, text):進行中句子的 Breeze 增量修正
        self.q = queue.Queue()
        self.dirty = True

        self.root.update_idletasks()
        user32 = ctypes.windll.user32
        hwnd = user32.GetParent(self.root.winfo_id()) or self.root.winfo_id()
        GWL_EXSTYLE = -20
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        # LAYERED | TRANSPARENT(滑鼠穿透) | NOACTIVATE | TOOLWINDOW
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | 0x80000 | 0x20 | 0x8000000 | 0x80)
        self.hwnd = hwnd
        print(f"dbg hwnd={hwnd} screen={sw}x{sh} geo={self.w}x{self.h}+0+{sh - self.h - bottom}",
              file=sys.stderr, flush=True)
        if read_stdin:  # demo/test 模式不讀 stdin:非互動環境 stdin 立即 EOF 會誤觸自我銷毀
            threading.Thread(target=self._stdin_reader, daemon=True).start()
        self.root.after(50, self._tick)
        self.root.after(2000, self._assert_topmost)

    def _stdin_reader(self):
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                self.q.put(json.loads(line))
            except json.JSONDecodeError:
                pass
        self.q.put({"kind": "eof"})

    def _assert_topmost(self):
        # 有些全螢幕/置頂應用會蓋上來,週期性重申最上層(不搶焦點)
        ctypes.windll.user32.SetWindowPos(self.hwnd, -1, 0, 0, 0, 0, 0x10 | 0x2 | 0x1)
        self.root.after(2000, self._assert_topmost)

    def _tick(self):
        while True:
            try:
                ev = self.q.get_nowait()
            except queue.Empty:
                break
            kind = ev.get("kind")
            if kind == "partial":
                self.partial = ev["text"]
            elif kind == "white_partial":
                self.white_partial = (ev.get("utt"), ev["text"], time.time())
            elif kind == "final":
                # 清掉該句「以及所有更舊句」的白字——過期快照的競態可能把死句貼回來
                if self.white_partial and (ev.get("utt") is None
                                           or (self.white_partial[0] or 0) <= (ev.get("utt") or 0)):
                    self.white_partial = None
                self.committed = (ev.get("seq"), ev["text"], time.time())
                self.partial = ""
            elif kind == "refined":
                if self.committed and self.committed[0] == ev.get("seq"):
                    self.committed = (self.committed[0], ev["text"], time.time())
                else:  # 對應定稿已被更新的句子蓋掉,仍以修正文字短暫顯示
                    self.committed = (ev.get("seq"), ev["text"], time.time())
            elif kind == "eof":
                self.root.destroy()
                return
            self.dirty = True
        if self.committed and time.time() - self.committed[2] > HOLD:
            self.committed = None
            self.dirty = True
        if self.white_partial and time.time() - self.white_partial[2] > 6.0:
            self.white_partial = None  # 增量白字超過 6s 沒更新=洩漏,自動清
            self.dirty = True
        if self.dirty:
            self._redraw()
            self.dirty = False
        self.root.after(50, self._tick)

    def _text_block(self, text, y_bottom, fill):
        """白字黑邊:黑色 8 向偏移打底,再疊主色。回傳文字塊頂端 y。"""
        x = self.w // 2
        ids = []
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (2, 2), (-2, 2), (2, -2)):
            ids.append(self.canvas.create_text(
                x + dx, y_bottom + dy, text=text, font=self.font, fill="black",
                width=self.wrap, anchor="s", justify="center"))
        main = self.canvas.create_text(x, y_bottom, text=text, font=self.font, fill=fill,
                                       width=self.wrap, anchor="s", justify="center")
        return self.canvas.bbox(main)[1]

    def _redraw(self):
        self.canvas.delete("all")
        y = self.h - 8
        if self.partial:
            y = self._text_block(self.partial, y, "#d8d8d8") - 6
        # 進行中句子的 Breeze 增量修正優先於上一句定稿
        if self.white_partial:
            self._text_block(self.white_partial[1], y, "white")
        elif self.committed:
            self._text_block(self.committed[1], y, "white")

    def run(self):
        self.root.mainloop()


def demo_feed():
    lines = [
        {"kind": "partial", "text": "那我們會"},
        {"kind": "partial", "text": "那我們會以這個例子"},
        {"kind": "partial", "text": "那我們會以這個例子做情境, 看 Orbit 怎麼"},
        {"kind": "final", "seq": 1, "text": "那我們會以這個例子做情境, 看 Orbit 怎麼陪漢神"},
        {"kind": "partial", "text": "走這個檔期"},
        {"kind": "refined", "seq": 1, "text": "那我們會以這個例子做情境,看 Orbit 怎麼陪漢神走這個檔期。"},
        {"kind": "partial", "text": "走這個檔期。首先是 Orbit 到底能帶給漢神什麼效益"},
        {"kind": "final", "seq": 2, "text": "首先是 Orbit 到底能帶給漢神什麼效益, 我們先一頁"},
        {"kind": "refined", "seq": 2, "text": "首先是 Orbit 到底能帶給漢神什麼效益,我們先用一頁說明。"},
    ]
    time.sleep(1)
    for ev in lines:
        print(json.dumps(ev, ensure_ascii=False), flush=True)
        time.sleep(1.2)
    time.sleep(6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--font-size", type=int, default=28)
    ap.add_argument("--bottom", type=int, default=70, help="距螢幕底部像素")
    ap.add_argument("--width-ratio", type=float, default=0.78)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--test", action="store_true", help="立即顯示靜態測試字幕(渲染診斷用)")
    args = ap.parse_args()

    if args.test:
        ov = Overlay(args.font_size, args.bottom, args.width_ratio, read_stdin=False)
        ov.committed = (0, "測試字幕:白字黑邊 Orbit subtitle test 一二三", time.time() + 3600)
        ov.partial = "進行中的草稿列 draft line"
        ov.dirty = True
        ov.run()
        return

    if args.demo:
        import subprocess
        feeder = subprocess.Popen(
            [sys.executable, "-c",
             "import overlay; overlay.demo_feed()"],
            stdout=subprocess.PIPE, cwd=str(__import__('pathlib').Path(__file__).parent),
            text=True, encoding="utf-8")
        ov = Overlay(args.font_size, args.bottom, args.width_ratio, read_stdin=False)

        def pump():
            for ln in feeder.stdout:
                ov.q.put(json.loads(ln))
            ov.q.put({"kind": "eof"})
        threading.Thread(target=pump, daemon=True).start()
        ov.run()
        return

    Overlay(args.font_size, args.bottom, args.width_ratio).run()


if __name__ == "__main__":
    main()
