# assets：需要跨机器完全一致的小文件

| 文件 | 内容 | 来源 |
|---|---|---|
| `zone_static.csv` | 275 个小区的面积、路网长度、几何质心、三类 POI 数、桩数、站数、**功能区编号** | 数据核查 `check10.py`（POI 转 GCJ-02 后落区；K-means k=3，按 POI 总密度从低到高编号 0/1/2） |

功能区由 K-means 得到，不同 sklearn 版本可能给出不同的编号，所以固定保存在这里，云端、Mac、Windows 共用。
需要重新生成时：删除本文件后运行 `python scripts/prepare_data.py`（约 30 秒，需要 scikit-learn 和 matplotlib）。
