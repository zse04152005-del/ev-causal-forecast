"""数据核查脚本的公共路径设置。

目录约定（相对项目根目录）：
- 输入：02_数据/raw/urbanev_github/data/（UrbanEV GitHub 版逐小时数据）
- 输出：02_数据/processed/audit/（所有 CSV、JSON、图都写到这里）
可用环境变量 URBANEV_DATA、AUDIT_OUT 覆盖。
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))          # 03_代码/scripts/audit -> 项目根目录
DATA = os.environ.get("URBANEV_DATA", os.path.join(ROOT, "02_数据", "raw", "urbanev_github", "data"))
OUT = os.environ.get("AUDIT_OUT", os.path.join(ROOT, "02_数据", "processed", "audit"))
os.makedirs(OUT, exist_ok=True)
os.chdir(OUT)                       # 各脚本用相对路径读写中间结果
if HERE not in sys.path:
    sys.path.insert(0, HERE)

def cjk_font(fm):
    """在 Linux / macOS / Windows 上找一个能显示中文的字体，返回字体名。"""
    cands = ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "/System/Library/Fonts/STHeiti Medium.ttc",
             "/System/Library/Fonts/Hiragino Sans GB.ttc", "/Library/Fonts/Arial Unicode.ttf",
             "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]
    for f in cands:
        if os.path.exists(f):
            try:
                fm.fontManager.addfont(f)
                return fm.FontProperties(fname=f).get_name()
            except Exception:
                continue
    return "sans-serif"
