#!/bin/bash
# 一键下载开放获取论文（共 21 篇）
# 用法：在 Mac 的"终端"中执行
#   cd ~/Desktop/EV充电因果预测sci论文/01_文献
#   bash 下载开放论文.sh
# 已存在的文件会跳过；下载失败的会在最后列出，可按《文献清单与下载链接.md》手动下载。
# 另有 2 篇开放获取论文（B4、F4，OSTI 公开版）和 7 篇需学校账号的论文，需要手动下载。

cd "$(dirname "$0")" || exit 1

FAILED=()

get() {
  local dir="$1" name="$2" url="$3"
  mkdir -p "$dir"
  if [ -s "$dir/$name" ]; then
    echo "已存在，跳过：$dir/$name"
    return
  fi
  echo "下载：$name"
  if curl -fL --retry 2 --connect-timeout 20 -A "Mozilla/5.0" -o "$dir/$name" "$url" \
     && head -c 4 "$dir/$name" | grep -q "%PDF"; then
    echo "  完成"
  else
    rm -f "$dir/$name"
    FAILED+=("$name  $url")
    echo "  失败"
  fi
  sleep 2
}

# A 对标论文
get A_对标论文 A1_Li_2025_UrbanEV.pdf                 https://www.nature.com/articles/s41597-025-04874-4.pdf
get A_对标论文 A2_Guo_2025_CHARGED.pdf                https://www.nature.com/articles/s41597-025-05584-7.pdf
get A_对标论文 A3_Qu_2024_PAG.pdf                     https://arxiv.org/pdf/2309.05259

# B EV 充电与定价实证
get B_EV充电与定价实证 B1_Bernard_2025_DynamicPricing_NBER.pdf https://www.nber.org/system/files/working_papers/w34600/w34600.pdf
get B_EV充电与定价实证 B3_Chen_2026_NoteChordVoice.pdf          https://arxiv.org/pdf/2608.14756

# C 时空预测
get C_时空预测 C1_Li_2018_DCRNN_ICLR.pdf              https://arxiv.org/pdf/1707.01926
get C_时空预测 C2_Yu_2018_STGCN_IJCAI.pdf             https://arxiv.org/pdf/1709.04875
get C_时空预测 C3_Wu_2019_GraphWaveNet_IJCAI.pdf      https://arxiv.org/pdf/1906.00121
get C_时空预测 C4_Bai_2020_AGCRN_NeurIPS.pdf          https://arxiv.org/pdf/2007.02842
get C_时空预测 C5_Liu_2023_STAEformer_CIKM.pdf        https://arxiv.org/pdf/2308.10425

# D 因果推断方法
get D_因果推断方法 D1_Chernozhukov_2018_DML.pdf        https://arxiv.org/pdf/1608.00060
get D_因果推断方法 D2_Wager_2018_CausalForest_JASA.pdf https://arxiv.org/pdf/1510.04342
get D_因果推断方法 D3_Hausman_2018_RDiT.pdf            https://www.nber.org/system/files/working_papers/w23602/w23602.pdf

# E 反事实预测与不确定性
get E_反事实预测与不确定性 E1_Bica_2020_CRN_ICLR.pdf                  https://arxiv.org/pdf/2002.04083
get E_反事实预测与不确定性 E2_Melnychuk_2022_CausalTransformer_ICML.pdf https://arxiv.org/pdf/2204.07258
get E_反事实预测与不确定性 F9_Gibbs_2021_ACI_NeurIPS.pdf              https://arxiv.org/pdf/2106.00170
get E_反事实预测与不确定性 F10_Xu_2021_EnbPI_ICML.pdf                 https://arxiv.org/pdf/2010.09107

# F 扩展阅读
get F_扩展阅读 F5_Callaway_2021_DID.pdf               https://arxiv.org/pdf/1803.09015
get F_扩展阅读 F6_Liu_2024_iTransformer_ICLR.pdf      https://arxiv.org/pdf/2310.06625
get F_扩展阅读 F7_Wang_2024_TimeXer_NeurIPS.pdf       https://arxiv.org/pdf/2402.19072
get F_扩展阅读 F8_Ansari_2024_Chronos.pdf             https://arxiv.org/pdf/2403.07815

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "全部下载完成。"
else
  echo "以下论文下载失败，请在浏览器中手动下载："
  for f in "${FAILED[@]}"; do echo "  $f"; done
fi
echo
echo "还需手动下载：B4、F4（OSTI 公开版）；A4、A5、B2、D4、F1、F2、F3（学校账号）。"
